#!/usr/bin/env python3
"""
segment_refiner_gpu — 全 GPU 版（对接骨架）

读取 03_skeleton/skeleton.json 的 shots[]，
逐镜头选取最佳代表帧，输出 segment_refiner/refined.json。

数据流:
  03_skeleton/skeleton.json (shots[].representative_frame = null)
    → 本模块填代表帧
    → segment_refiner/refined.json (shots[].representative_frame = 帧号)

核心计算(GPU):
  Phase 1: 一次 ffmpeg NVDEC 解码全部帧 224×224 → CPU uint8
  Phase 2: 分批 GPU interpolate 160×90 → 全局 MAFD 曲线
  Phase 3: 逐镜头候选帧 → GPU Laplacian → GPU 三层评分 → 选代表帧
"""

import json, os, sys, time, subprocess as sp
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

# ── 参数 ────────────────────────────────────────────────────────
MAFD_H, MAFD_W = 90, 160
PEAK_DISTANCE = 30
PEAK_PROMINENCE_PCT = 60
PEAK_HEIGHT_PCT = 40
MIN_SUBSEG_FRAMES = 15
MAX_SUBSEG_PER_SEGMENT = 3
CENTER_RANGE = 15
STABLE_WINDOW_RATIO = 0.06
MIN_STABLE_WINDOW = 5
MAX_STABLE_WINDOW = 31
GPU_BATCH = 1000


# ── 静默 ffmpeg ─────────────────────────────────────────────────
def _run_ffmpeg(cmd):
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=64 * 1024 * 1024)
    return proc


# ── 单次解码全视频 224×224 → CPU uint8 ──────────────────────────
def _decode_all_frames(video, total_frames, width=224, height=224):
    """一次 ffmpeg 解码全视频 224×224 → CPU uint8 array (N, H, W, 3)"""
    vf = (
        "scale='if(gt(iw,ih),-2,{w})':'if(gt(iw,ih),{h},-2)',"
        "crop={w}:{h}"
    ).format(w=width, h=height)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-hwaccel", "cuda",
        "-i", video,
        "-vf", vf,
        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-vframes", str(total_frames), "-",
    ]

    expected = total_frames * height * width * 3
    t0 = time.time()
    proc = _run_ffmpeg(cmd)

    CHUNK = 64 * 1024 * 1024
    chunks = []
    while True:
        buf = proc.stdout.read(CHUNK)
        if not buf:
            break
        chunks.append(buf)
    proc.stdout.close()
    proc.wait()

    raw = b"".join(chunks)
    del chunks
    read_time = time.time() - t0

    if len(raw) < expected:
        raw = raw + b"\x00" * (expected - len(raw))

    arr = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(total_frames, height, width, 3).copy()
    del raw
    print(f"  decode: {total_frames}fr, {arr.nbytes/1e6:.0f}MB, {read_time:.1f}s")
    return arr


# ── 分批 GPU 算全局 MAFD ────────────────────────────────────────
def _compute_global_mafd(all_frames, device="cuda"):
    """分批: 224×224 → GPU → interpolate 160×90 → MAFD"""
    n = all_frames.shape[0]
    mafd_chunks = []

    t0 = time.time()
    for i in range(0, n - 1, GPU_BATCH):
        end = min(i + GPU_BATCH + 1, n)
        chunk = all_frames[i:end]

        tensor = torch.from_numpy(chunk).to(device, dtype=torch.float32, non_blocking=True) / 255.0
        tensor = tensor.permute(0, 3, 1, 2)
        small = F.interpolate(tensor, size=(MAFD_H, MAFD_W), mode='bilinear', align_corners=False)
        diff = torch.abs(small[1:] - small[:-1]).mean(dim=(1, 2, 3))
        mafd_chunks.append(diff.cpu())

        del tensor, small, diff
        torch.cuda.empty_cache()

    mafd = torch.cat(mafd_chunks).numpy()
    elapsed = time.time() - t0
    print(f"  MAFD: {len(mafd)} values, {elapsed:.1f}s, mean={mafd.mean():.4f}")
    return mafd


# ── 自适应平滑 ───────────────────────────────────────────────────
def smooth_signal(signal, n_frames):
    window = max(MIN_STABLE_WINDOW, min(MAX_STABLE_WINDOW,
                                        round(n_frames * STABLE_WINDOW_RATIO)))
    if window < 2:
        return signal
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode='same')


# ── 内部转折检测 ─────────────────────────────────────────────────
def detect_stable_zones(raw_mafd, n_frames):
    arr = np.array(raw_mafd, dtype=np.float32)
    if len(arr) < 2 * PEAK_DISTANCE:
        return [(0, n_frames - 1, 0.0)]

    smoothed = smooth_signal(arr, n_frames)
    nonzero = smoothed[smoothed > 0]
    if len(nonzero) == 0:
        return [(0, n_frames - 1, 0.0)]

    prom = float(np.percentile(nonzero, PEAK_PROMINENCE_PCT))
    height = float(np.percentile(nonzero, PEAK_HEIGHT_PCT))
    if prom < 1e-6:
        prom = float(nonzero.max() * 0.1) if nonzero.max() > 0 else 1e-6

    peaks, _ = find_peaks(smoothed, distance=PEAK_DISTANCE, prominence=prom, height=height)
    if len(peaks) == 0:
        return [(0, n_frames - 1, 0.0)]

    max_idx = len(arr) - 1
    boundaries = sorted(set([0] + peaks.tolist() + [max_idx]))

    zones = []
    for i in range(len(boundaries) - 1):
        sf = boundaries[i]
        ef = min(boundaries[i + 1], max_idx)
        dur = ef - sf + 1
        trans_score = float(smoothed[sf:ef + 1].max())
        if dur >= MIN_SUBSEG_FRAMES:
            zones.append((sf, ef, trans_score))
        else:
            if zones:
                zones[-1] = (zones[-1][0], ef, zones[-1][2])
            else:
                zones.append((sf, ef, 0.0))

    return zones[:MAX_SUBSEG_PER_SEGMENT]


# ── GPU Laplacian (替代 cv2.Laplacian) ──────────────────────────
def _gpu_laplacian_var(frames_gpu):
    """
    GPU 离散 Laplacian 方差。替代 cv2.Laplacian(gray, CV_64F).var()。

    frames_gpu: (K, 3, H, W) float32, [0..1]
    返回: (K,) float32, 每帧的 Laplacian 方差 ≈ sharpness
    """
    gray = 0.299 * frames_gpu[:, 0] + 0.587 * frames_gpu[:, 1] + 0.114 * frames_gpu[:, 2]

    lap = (
        4.0 * gray[:, 1:-1, 1:-1]
        - gray[:, :-2, 1:-1] - gray[:, 2:, 1:-1]
        - gray[:, 1:-1, :-2] - gray[:, 1:-1, 2:]
    )

    b = lap.reshape(lap.shape[0], -1)
    mean_lap = b.mean(dim=1, keepdim=True)
    var_lap = ((b - mean_lap) ** 2).mean(dim=1)
    return var_lap


# ── GPU 单镜头处理 ──────────────────────────────────────────────
def process_shot_gpu(shot_id, seg_start, seg_end,
                     all_frames_224, full_mafd, device="cuda"):
    """
    全 GPU 处理一个 shot。
    候选帧批量传 GPU → GPU Laplacian → GPU 三层评分。
    """
    t0 = time.time()
    n_frames = seg_end - seg_start + 1
    print(f"  [shot {shot_id}] {seg_start}-{seg_end} ({n_frames}fr)", end="", flush=True)

    seg_mafd = full_mafd[seg_start:seg_end]
    zones = detect_stable_zones(seg_mafd, n_frames)

    best_frames = []
    for rel_sf, rel_ef, trans_score in zones:
        abs_sf = seg_start + rel_sf
        abs_ef = seg_start + rel_ef
        zone_len = rel_ef - rel_sf + 1
        zone_mafd = seg_mafd[rel_sf:rel_ef + 1]

        # stable_center
        best_rel = int(np.argmin(zone_mafd))
        best_abs = abs_sf + best_rel

        # 候选帧列表 (全局帧号)
        candidates = []
        for offset in range(-CENTER_RANGE, CENTER_RANGE + 1):
            cf = best_rel + offset
            if 0 <= cf < zone_len:
                candidates.append(abs_sf + cf)
        candidates = sorted(set(candidates))

        if len(candidates) == 1:
            best_frames.append(candidates[0])
            continue

        # ── 全 GPU 三层评分 ──
        cand_np = all_frames_224[candidates]
        cand_gpu = torch.from_numpy(cand_np).to(device, dtype=torch.float32, non_blocking=True)
        del cand_np
        cand_gpu = cand_gpu.permute(0, 3, 1, 2) / 255.0

        # Stability (0.5)
        cand_rel_list = [c - abs_sf for c in candidates]
        cand_mafd_idx = [min(r, len(zone_mafd) - 1) for r in cand_rel_list]
        stab_vals = torch.tensor([zone_mafd[i] for i in cand_mafd_idx],
                                  device=device, dtype=torch.float32)
        stab = 1.0 / (1.0 + stab_vals)

        # Sharpness (0.3)
        sharp_raw = _gpu_laplacian_var(cand_gpu)
        sharp = sharp_raw / (sharp_raw.max() + 1e-6)

        # Center distance (0.2)
        dists = torch.tensor([abs(c - best_abs) for c in candidates],
                              device=device, dtype=torch.float32)
        dists = 1.0 / (1.0 + dists)

        # 加权总分
        scores = 0.5 * stab + 0.3 * sharp + 0.2 * dists
        best_idx = int(scores.argmax().cpu())
        best_frames.append(candidates[best_idx])

        del cand_gpu, stab, sharp_raw, sharp, dists, scores

    torch.cuda.empty_cache()
    elapsed = time.time() - t0
    print(f" -> {len(zones)} zones, rep_frames={best_frames} ({elapsed:.1f}s)")

    # 有多个 zone 时取第一个为代表帧（主镜头代表帧）
    return best_frames[0] if best_frames else seg_start


# ── 主流程 ──────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <output_dir>")
        print(f"  output_dir 应包含 03_skeleton/skeleton.json 和视频文件")
        sys.exit(1)

    output = sys.argv[1]
    skeleton_path = os.path.join(output, "03_skeleton", "skeleton.json")

    if not os.path.isfile(skeleton_path):
        print(f"错误: 找不到骨架文件 {skeleton_path}")
        print(f"请先运行 03_skeleton.py <output_dir>")
        sys.exit(1)

    with open(skeleton_path) as f:
        skeleton = json.load(f)

    shots = skeleton["shots"]
    total_frames = skeleton["total_frames"]
    video_path = skeleton["video"]
    fps = skeleton.get("fps", 30.0)

    print(f"[segment_refiner GPU] {len(shots)} shots, {total_frames} frames, {fps:.2f}fps")
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    # ═══ Phase 1: 一次解码 ═══
    print()
    print("Phase 1/3: decoding all frames 224×224...")
    all_frames = _decode_all_frames(video_path, total_frames)

    # ═══ Phase 2: 全局 MAFD (分批 GPU) ═══
    print()
    print("Phase 2/3: global MAFD on GPU...")
    full_mafd = _compute_global_mafd(all_frames, device=device)

    # ═══ Phase 3: 逐镜头 GPU 处理 ═══
    print()
    print("Phase 3/3: processing shots (GPU scoring)...")
    for shot in shots:
        sf = shot["range"]["start"]
        ef = shot["range"]["end"]
        rep = process_shot_gpu(shot["id"], sf, ef, all_frames, full_mafd, device)
        shot["representative_frame"] = rep

    del all_frames, full_mafd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 输出 refined.json（骨架 + 代表帧） ──
    refined = {
        "video": skeleton["video"],
        "total_frames": skeleton["total_frames"],
        "duration": skeleton.get("duration"),
        "fps": skeleton.get("fps"),
        "width": skeleton.get("width"),
        "height": skeleton.get("height"),
        "shots": shots,
    }

    out_dir = os.path.join(output, "segment_refiner")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "refined.json")
    with open(out_path, "w") as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)

    filled = sum(1 for s in shots if s["representative_frame"] is not None)
    elapsed = time.time() - t_start
    print(f"\nDone: {filled}/{len(shots)} shots filled ({elapsed:.1f}s)")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
