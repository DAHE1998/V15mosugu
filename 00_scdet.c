/*
 * 00_scdet.c — GPU 场景切点检测
 *
 * 全程 GPU: NVDEC CUDA 硬解 → Vulkan compute shader
 * 编译: gcc -O2 -o 00_scdet 00_scdet.c -lm
 * 用法: ./00_scdet <video.mp4> [work_dir]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#define MAX_CUTS 65536
#define SCDET_THR 10
#define MIN_GAP   2

typedef struct {
    char path[1024];
    int  width, height, total_frames;
    double duration, fps;
} VideoInfo;

/* ── ffprobe 辅助 ────────────────────────────────────────────── */

static int probe_int(const char *video, const char *entries, int *val) {
    char cmd[2048], buf[128];
    snprintf(cmd, sizeof(cmd),
        "ffprobe -v error -select_streams v:0 -show_entries %s -of csv=p=0 \"%s\"",
        entries, video);
    FILE *fp = popen(cmd, "r");
    if (!fp) return -1;
    if (!fgets(buf, sizeof(buf), fp)) { pclose(fp); return -1; }
    pclose(fp);
    *val = atoi(buf);
    return 0;
}

static int probe_float(const char *video, const char *entries, double *val) {
    char cmd[2048], buf[128];
    snprintf(cmd, sizeof(cmd),
        "ffprobe -v error -show_entries %s -of csv=p=0 \"%s\"",
        entries, video);
    FILE *fp = popen(cmd, "r");
    if (!fp) return -1;
    if (!fgets(buf, sizeof(buf), fp)) { pclose(fp); return -1; }
    pclose(fp);
    *val = atof(buf);
    return 0;
}

static int probe_str(const char *video, const char *entries, char *out, size_t n) {
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "ffprobe -v error -select_streams v:0 -show_entries %s -of csv=p=0 \"%s\"",
        entries, video);
    FILE *fp = popen(cmd, "r");
    if (!fp) return -1;
    if (!fgets(out, n, fp)) { pclose(fp); return -1; }
    size_t len = strlen(out);
    while (len > 0 && (out[len-1] == '\n' || out[len-1] == '\r')) out[--len] = 0;
    pclose(fp);
    return 0;
}

/* ── 获取视频元信息 ───────────────────────────────────────────── */

static void get_video_info(const char *video, VideoInfo *vi) {
    snprintf(vi->path, sizeof(vi->path), "%s", video);
    probe_int(video, "stream=width",  &vi->width);
    probe_int(video, "stream=height", &vi->height);
    probe_int(video, "stream=nb_frames", &vi->total_frames);
    probe_float(video, "format=duration", &vi->duration);

    if (vi->total_frames <= 0)
        vi->total_frames = (int)(vi->duration * 24.0);

    char fps_str[64] = {0};
    probe_str(video, "stream=r_frame_rate", fps_str, sizeof(fps_str));
    char *slash = strchr(fps_str, '/');
    if (slash) {
        double num = atof(fps_str);
        double den = atof(slash + 1);
        vi->fps = (den > 0) ? num / den : 24.0;
    } else {
        vi->fps = atof(fps_str);
        if (vi->fps <= 0) vi->fps = vi->total_frames / vi->duration;
    }
}

/* ── scdet_vulkan 全 GPU 检测 ─────────────────────────────────── */

static int run_scdet(const char *video, double fps, double *times, int *n_times) {
    char cmd[4096], buf[256];
    int count = 0;

    snprintf(cmd, sizeof(cmd),
        "ffmpeg -init_hw_device vulkan=vk -filter_hw_device vk "
        "-hwaccel cuda -i \"%s\" "
        "-vf \"hwupload,scdet_vulkan=threshold=%d\" "
        "-f null - 2>&1",
        video, SCDET_THR);

    FILE *fp = popen(cmd, "r");
    if (!fp) { perror("popen"); return -1; }

    while (fgets(buf, sizeof(buf), fp) && count < MAX_CUTS) {
        char *p = strstr(buf, "scd.time:");
        if (!p) continue;
        p += 9;
        while (*p == ' ') p++;
        times[count++] = atof(p);
    }
    pclose(fp);
    *n_times = count;
    return 0;
}

/* ── qsort 比较函数 ───────────────────────────────────────────── */

static int cmp_int(const void *a, const void *b) {
    return *(const int *)a - *(const int *)b;
}

/* ── JSON 输出 ────────────────────────────────────────────────── */

static void write_raw_cuts(const char *path, const VideoInfo *vi,
                            const int *cuts, int n) {
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    fprintf(fp, "{\n");
    fprintf(fp, "  \"video\": \"%s\",\n", vi->path);
    fprintf(fp, "  \"total_frames\": %d,\n", vi->total_frames);
    fprintf(fp, "  \"duration\": %.6f,\n", vi->duration);
    fprintf(fp, "  \"fps\": %.10g,\n", vi->fps);
    fprintf(fp, "  \"width\": %d,\n", vi->width);
    fprintf(fp, "  \"height\": %d,\n", vi->height);
    fprintf(fp, "  \"cuts\": [\n");
    for (int i = 0; i < n; i++)
        fprintf(fp, "    %d%s\n", cuts[i], i < n-1 ? "," : "");
    fprintf(fp, "  ]\n}\n");
    fclose(fp);
}

static void write_events(const char *path, const VideoInfo *vi,
                          const int *cuts, int n) {
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    fprintf(fp, "{\n");
    fprintf(fp, "  \"video\": \"%s\",\n", vi->path);
    fprintf(fp, "  \"total_frames\": %d,\n", vi->total_frames);
    fprintf(fp, "  \"duration\": %.6f,\n", vi->duration);
    fprintf(fp, "  \"fps\": %.10g,\n", vi->fps);
    fprintf(fp, "  \"width\": %d,\n", vi->width);
    fprintf(fp, "  \"height\": %d,\n", vi->height);
    fprintf(fp, "  \"cuts\": [");
    for (int i = 0; i < n; i++)
        fprintf(fp, "%d%s", cuts[i], i < n-1 ? "," : "");
    fprintf(fp, "],\n");
    fprintf(fp, "  \"events\": []\n}\n");
    fclose(fp);
}

/* ── main ─────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <video.mp4> [work_dir]\n", argv[0]);
        return 1;
    }

    const char *video = argv[1];

    /* 计算 work_dir */
    char work_dir[1024];
    if (argc >= 3) {
        snprintf(work_dir, sizeof(work_dir), "%s", argv[2]);
    } else {
        const char *base = strrchr(video, '/');
        const char *name = base ? base + 1 : video;
        const char *dot  = strrchr(name, '.');
        char vid_name[256];
        if (dot) { size_t n = dot - name; memcpy(vid_name, name, n); vid_name[n] = 0; }
        else snprintf(vid_name, sizeof(vid_name), "%s", name);

        if (base) {
            size_t d = base - video;
            memcpy(work_dir, video, d); work_dir[d] = 0;
            snprintf(work_dir + d, sizeof(work_dir) - d, "/%s_v15mosugu", vid_name);
        } else {
            snprintf(work_dir, sizeof(work_dir), "%s_v15mosugu", vid_name);
        }
    }

    char out_dir[2048];
    snprintf(out_dir, sizeof(out_dir), "%s/00_scdet", work_dir);
    mkdir(work_dir, 0755);
    mkdir(out_dir, 0755);

    /* 视频信息 */
    VideoInfo vi = {0};
    get_video_info(video, &vi);
    printf("[00] %dx%d  %dfr  %.1fs  %.4ffps\n",
           vi.width, vi.height, vi.total_frames, vi.duration, vi.fps);
    fflush(stdout);

    /* scdet_vulkan */
    double times[MAX_CUTS];
    int n_raw = 0;
    time_t t0 = time(NULL);
    run_scdet(video, vi.fps, times, &n_raw);
    time_t dt = time(NULL) - t0;
    printf("  scdet: %d detections (%lds)\n", n_raw, dt);
    fflush(stdout);

    /* 帧号转换 + -1 校正 */
    int raw_frames[MAX_CUTS], n_rf = 0;
    for (int i = 0; i < n_raw; i++) {
        int f = (int)round(times[i] * vi.fps);
        int c = f - 1;
        if (c > 1 && c < vi.total_frames) raw_frames[n_rf++] = c;
    }

    /* 排序 & 去重 */
    qsort(raw_frames, n_rf, sizeof(int), cmp_int);

    int cuts[MAX_CUTS], n = 0;
    for (int i = 0; i < n_rf; i++)
        if (i == 0 || raw_frames[i] - raw_frames[i-1] >= MIN_GAP)
            cuts[n++] = raw_frames[i];

    printf("  -> %d cuts (after -1 correction + dedup)\n", n);
    printf("  first 10:");
    for (int i = 0; i < 10 && i < n; i++) printf(" %d", cuts[i]);
    printf("\n");
    fflush(stdout);

    /* 输出 */
    char path_r[2048], path_e[2048];
    snprintf(path_r, sizeof(path_r), "%s/raw_cuts.json", out_dir);
    snprintf(path_e, sizeof(path_e), "%s/events.json",   out_dir);
    write_raw_cuts(path_r, &vi, cuts, n);
    write_events(path_e, &vi, cuts, n);
    printf("  -> %s\n  -> %s\n", path_r, path_e);

    return 0;
}
