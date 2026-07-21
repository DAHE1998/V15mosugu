/*
 * 01_skeleton.c — 切点列表 → 镜头骨架
 *
 * 从 00_scdet 的 raw_cuts.json 读取，只转发上游信息。
 * 不做视觉/语义/时间计算。
 *
 * 输出: 01_skeleton/skeleton.json
 *   - video, fps, width, height, total_frames
 *   - shots[].{id, range{start, end}}
 *
 * 编译: gcc -O2 -o 01_skeleton 01_skeleton.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#define LOG_TAG "[01]"
#define MAX_CUTS 65536
#define MAX_PATH 1024
#define MAX_VIDEO_PATH 4096

/* ── 简易 JSON 读取（只处理 00_scdet 的已知平铺结构） ── */

static char *read_file(const char *path, long *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    rewind(fp);
    char *buf = malloc(len + 1);
    if (!buf) { fclose(fp); return NULL; }
    fread(buf, 1, len, fp);
    fclose(fp);
    buf[len] = '\0';
    *out_len = len;
    return buf;
}

/* 提取字符串值: "key": "value" */
static int extract_string(const char *json, const char *key, char *out, int out_size) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return -1;
    p = strchr(p, ':');
    if (!p) return -1;
    p++;
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    if (*p != '"') return -1;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < out_size - 1) {
        if (*p == '\\' && *(p + 1)) p++;
        out[i++] = *p++;
    }
    out[i] = '\0';
    return 0;
}

/* 提取数值: "key": <number> */
static int extract_number(const char *json, const char *key, double *out) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return -1;
    p = strchr(p, ':');
    if (!p) return -1;
    p++;
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    *out = strtod(p, NULL);
    return 0;
}

/* 提取整数 */
static int extract_int(const char *json, const char *key, int *out) {
    double d;
    if (extract_number(json, key, &d) < 0) return -1;
    *out = (int)round(d);
    return 0;
}

/* 解析 cuts 数组: [n1, n2, n3, ...] */
static int parse_cuts(const char *json, int *cuts, int max_cuts) {
    const char *p = strstr(json, "\"cuts\"");
    if (!p) return 0;
    p = strchr(p, '[');
    if (!p) return 0;
    p++;
    int n = 0;
    while (*p && *p != ']' && n < max_cuts) {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')) p++;
        if (*p == ']') break;
        cuts[n++] = (int)strtol(p, (char **)&p, 10);
    }
    return n;
}

static int cmp_int(const void *a, const void *b) {
    return *(int *)a - *(int *)b;
}

/* ── main ── */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "%s Usage: 01_skeleton <work_dir>\n", LOG_TAG);
        return 1;
    }

    char work[MAX_PATH];
    snprintf(work, sizeof(work), "%s", argv[1]);

    /* 构建路径 */
    char cuts_path[MAX_PATH * 2];
    char out_dir[MAX_PATH * 2];
    char out_path[MAX_PATH * 2];

    snprintf(cuts_path, sizeof(cuts_path), "%s/00_scdet/raw_cuts.json", work);

    /* 尝试旧路径 */
    {
        struct stat st;
        if (stat(cuts_path, &st) != 0) {
            snprintf(cuts_path, sizeof(cuts_path), "%s/00_change_cuda/raw_cuts.json", work);
            if (stat(cuts_path, &st) != 0) {
                fprintf(stderr, "%s Error: raw_cuts.json not found in 00_scdet/ or 00_change_cuda/\n", LOG_TAG);
                return 1;
            }
        }
    }

    /* 读输入 JSON */
    long len;
    char *json = read_file(cuts_path, &len);
    if (!json) {
        fprintf(stderr, "%s Error: cannot read %s\n", LOG_TAG, cuts_path);
        return 1;
    }

    /* 提取上游字段 — 直接 d["key"]，无保底，缺字段直接崩 */
    char video[MAX_VIDEO_PATH] = {0};
    if (extract_string(json, "video", video, sizeof(video)) < 0) {
        fprintf(stderr, "%s Error: missing 'video'\n", LOG_TAG);
        free(json);
        return 1;
    }

    double fps_val;
    if (extract_number(json, "fps", &fps_val) < 0) {
        fprintf(stderr, "%s Error: missing 'fps'\n", LOG_TAG);
        free(json);
        return 1;
    }

    int width, height, total_frames;
    if (extract_int(json, "width", &width) < 0) {
        fprintf(stderr, "%s Error: missing 'width'\n", LOG_TAG);
        free(json);
        return 1;
    }
    if (extract_int(json, "height", &height) < 0) {
        fprintf(stderr, "%s Error: missing 'height'\n", LOG_TAG);
        free(json);
        return 1;
    }
    if (extract_int(json, "total_frames", &total_frames) < 0) {
        fprintf(stderr, "%s Error: missing 'total_frames'\n", LOG_TAG);
        free(json);
        return 1;
    }

    /* 解析 cuts */
    int cuts[MAX_CUTS];
    int n_cuts = parse_cuts(json, cuts, MAX_CUTS);
    free(json);

    /* 过滤有效切点 + 排序 + 去重 */
    int valid[MAX_CUTS];
    int n_valid = 0;
    for (int i = 0; i < n_cuts; i++) {
        if (cuts[i] > 0 && cuts[i] < total_frames) {
            valid[n_valid++] = cuts[i];
        }
    }
    qsort(valid, n_valid, sizeof(int), cmp_int);

    int deduped[MAX_CUTS];
    int n_dedup = 0;
    for (int i = 0; i < n_valid; i++) {
        if (n_dedup == 0 || valid[i] != deduped[n_dedup - 1]) {
            deduped[n_dedup++] = valid[i];
        }
    }

    /* 构建镜头区间 */
    int bounds[MAX_CUTS + 2];
    int n_bounds = 0;
    bounds[n_bounds++] = 0;
    for (int i = 0; i < n_dedup; i++) {
        bounds[n_bounds++] = deduped[i];
    }
    bounds[n_bounds++] = total_frames;
    int n_shots = n_bounds - 1;

    /* 创建输出目录 */
    snprintf(out_dir, sizeof(out_dir), "%s/01_skeleton", work);
    mkdir(out_dir, 0755);

    /* 写 skeleton.json */
    snprintf(out_path, sizeof(out_path), "%s/skeleton.json", out_dir);
    FILE *fp = fopen(out_path, "w");
    if (!fp) {
        fprintf(stderr, "%s Error: cannot write %s\n", LOG_TAG, out_path);
        return 1;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"video\": \"%s\",\n", video);
    fprintf(fp, "  \"fps\": %.10g,\n", fps_val);
    fprintf(fp, "  \"width\": %d,\n", width);
    fprintf(fp, "  \"height\": %d,\n", height);
    fprintf(fp, "  \"total_frames\": %d,\n", total_frames);
    fprintf(fp, "  \"shots\": [\n");

    for (int i = 0; i < n_shots; i++) {
        int sf = bounds[i];
        int ef = bounds[i + 1] - 1;
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"id\": %d,\n", i);
        fprintf(fp, "      \"range\": {\n");
        fprintf(fp, "        \"start\": %d,\n", sf);
        fprintf(fp, "        \"end\": %d\n", ef);
        fprintf(fp, "      }\n");
        fprintf(fp, "    }%s\n", (i < n_shots - 1) ? "," : "");
    }

    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");

    fclose(fp);
    printf("%s -> %s (%d shots)\n", LOG_TAG, out_path, n_shots);

    return 0;
}
