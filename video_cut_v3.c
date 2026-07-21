/*
 * video_cut_v3.c — C translation of segment_cli_v2.py
 *
 * 逐段独立 ffmpeg 子进程。每段新鲜 NVENC，零状态污染。
 * 架构与已验证的 Python 版本完全一致。
 *
 * Compile: gcc -o video_cut video_cut_v3.c -O2 -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/wait.h>

/* ── JSON ── */
typedef struct { int id, start, end; } Shot;

static char *read_file(const char *path, long *len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    *len = ftell(fp); rewind(fp);
    char *buf = malloc(*len + 1);
    if (!buf) { fclose(fp); return NULL; }
    fread(buf, 1, *len, fp); fclose(fp);
    buf[*len] = '\0'; return buf;
}

static int extract_str(const char *j, const char *k, char *out, int sz) {
    char s[64]; snprintf(s, sizeof(s), "\"%s\"", k);
    const char *p = strstr(j, s); if (!p) return -1;
    p = strchr(p, ':'); if (!p) return -1;
    p++; while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != '"') return -1; p++;
    int i = 0;
    while (*p && *p != '"' && i < sz - 1) {
        if (*p == '\\' && *(p+1)) p++; out[i++] = *p++;
    }
    out[i] = '\0'; return 0;
}

static int parse_shots(const char *j, Shot *s, int max) {
    const char *p = strstr(j, "\"shots\""); if (!p) return 0;
    p = strchr(p, '['); if (!p) return 0; p++;
    int n = 0;
    while (*p && *p != ']' && n < max) {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')) p++;
        if (*p != '{') break;
        int d = 0; const char *e = p;
        while (*e) { if (*e == '{') d++; if (*e == '}') { d--; if (d == 0) break; } e++; }
        if (!*e) break;
        int olen = (int)(e - p + 1); char obj[4096];
        if (olen >= (int)sizeof(obj)) { p = e + 1; continue; }
        strncpy(obj, p, olen); obj[olen] = '\0';
        int id = -1, st = -1, ed = -1;
        char *k;
        k = strstr(obj, "\"id\"");    if (k) { k = strchr(k,':'); if (k) id = (int)strtol(k+1,NULL,10); }
        k = strstr(obj, "\"start\""); if (k) { k = strchr(k,':'); if (k) st = (int)strtol(k+1,NULL,10); }
        k = strstr(obj, "\"end\"");   if (k) { k = strchr(k,':'); if (k) ed = (int)strtol(k+1,NULL,10); }
        if (id >= 0 && st >= 0 && ed >= 0 && ed >= st) s[n++] = (Shot){id, st, ed};
        p = e + 1;
    }
    return n;
}

static int shot_cmp(const void *a, const void *b) {
    return ((const Shot *)a)->start - ((const Shot *)b)->start;
}

/* ── FPS probe ── */
static double probe_fps(const char *video_path) {
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "ffprobe -v quiet -select_streams v:0 "
        "-show_entries stream=avg_frame_rate "
        "-of default=noprint_wrappers=1:nokey=1 '%s'", video_path);
    FILE *fp = popen(cmd, "r");
    if (!fp) return 30.0;
    char buf[64]; fgets(buf, sizeof(buf), fp);
    pclose(fp);
    int num = 30000, den = 1001;
    sscanf(buf, "%d/%d", &num, &den);
    return (double)num / den;
}

/* ── spawn one ffmpeg segment ── */
static int spawn_segment(const Shot *s, const char *video,
                          const char *out_dir, double fps,
                          int max_workers) {
    int trim_start = s->start + 1;
    int trim_end   = s->end - 1;
    int n_frames = trim_end - trim_start + 1;
    double t_start = trim_start / fps;

    char out_path[4096];
    snprintf(out_path, sizeof(out_path), "%s/segment_%04d.mp4",
             out_dir, s->id);

    char sn[32], st[32];
    snprintf(sn, sizeof(sn), "%d", n_frames);
    snprintf(st, sizeof(st), "%.6f", t_start);

    /* -ss before -i → GPU keyframe seek, fast. ±1 frame at boundaries OK. */
    char cmd[8192];
    snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -hide_banner "
        "-hwaccel cuda -hwaccel_output_format cuda "
        "-ss %s -i '%s' "
        "-frames:v %s "
        "-c:v h264_nvenc -preset p1 -cq 26 "
        "-c:a aac -b:a 128k '%s'",
        st, video, sn, out_path);

    int ret = system(cmd);
    int ok = (ret == 0);

    /* Verify frame count */
    if (ok) {
        char vfy[4096];
        snprintf(vfy, sizeof(vfy),
            "ffprobe -v quiet -count_frames -select_streams v:0 "
            "-show_entries stream=nb_read_frames "
            "-of csv=p=0 '%s'", out_path);
        FILE *fp = popen(vfy, "r");
        int got = -1;
        if (fp) { fscanf(fp, "%d", &got); pclose(fp); }
        if (got != n_frames) {
            fprintf(stderr, "  [%3d] FRAME MISMATCH: got=%d exp=%d\n",
                    s->id, got, n_frames);
            ok = 0;
        }
    }

    printf("  [%3d] segment_%04d.mp4  %4dfr  %s\n",
           s->id, s->id, n_frames, ok ? "OK" : "ERR");
    fflush(stdout);
    return ok;
}

/* ── main ── */
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <skeleton.json>\n", argv[0]);
        return 1;
    }

    long json_len;
    char *json = read_file(argv[1], &json_len);
    if (!json) { fprintf(stderr, "Cannot read JSON\n"); return 1; }

    char video_path[4096] = {0};
    extract_str(json, "video", video_path, sizeof(video_path));

    Shot shots[65536];
    int n_shots = parse_shots(json, shots, 65536);
    free(json);

    if (n_shots == 0) { fprintf(stderr, "No shots\n"); return 1; }
    qsort(shots, n_shots, sizeof(Shot), shot_cmp);

    char out_dir[4096];
    { const char *s = strrchr(argv[1], '/');
      if (s) { int d = (int)(s - argv[1]); strncpy(out_dir, argv[1], d); out_dir[d]='\0'; }
      else strcpy(out_dir, "."); }

    double fps = probe_fps(video_path);

    printf("Video: %s\n", video_path);
    printf("Shots: %d  FPS: %.4f\n", n_shots, fps);
    printf("Output: %s/\n", out_dir);
    printf("Workers: 5  Encoder: h264_nvenc (per-shot fresh process)\n\n");

    /* Parallel execution — matching Python ThreadPoolExecutor(max_workers=5) */
    #define MAX_PROCS 5
    int running = 0;
    int ok = 0, fail = 0;
    int next = 0;

    while (next < n_shots || running > 0) {
        /* Launch up to MAX_PROCS */
        while (running < MAX_PROCS && next < n_shots) {
            pid_t pid = fork();
            if (pid == 0) {
                /* Child: run ffmpeg for one segment */
                int ret = spawn_segment(&shots[next], video_path, out_dir, fps, 5);
                _exit(ret ? 0 : 1);
            }
            if (pid > 0) {
                running++;
                next++;
            }
        }

        /* Wait for one child to finish */
        if (running > 0) {
            int status;
            pid_t pid = wait(&status);
            if (pid > 0) {
                running--;
                if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
                    ok++;
                else
                    fail++;
            }
        }
    }

    printf("\nDone: %d OK  %d FAIL  (total %d)\n", ok, fail, n_shots);
    return fail > 0 ? 1 : 0;
}
