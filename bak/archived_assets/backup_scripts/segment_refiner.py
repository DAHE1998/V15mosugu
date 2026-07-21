#!/usr/bin/env python3
"""
segment_refiner v3 — 一次解码全量处理 (Claude 方案 C)

流程:
  1. 一次 ffmpeg 解码全部 24573 帧 224×224 → CPU uint8 array
  2. 分批 GPU → interpolate 160×90 → 全局 MAFD 曲线
  3. 从 MAFD 曲线按 cuts.json 切 segments → find_peaks → zones
  4. 从 CPU 内存取候选帧 224×224 → Laplacian → 三层评分
"""

import json, os, sys, time, subprocess as sp
import numpy as np
import cv2
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
GPU_BATCH = 1000  # 每批送 GPU 的帧数


# ── 静默 ffmpeg 辅助 ────────────────────────────────────────────
def _run_ffmpeg(cmd):
    """执行 ffmpeg，只返回 stdout，错误静默丢弃。大量数据时使用 pipe。"""
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=64 * 1024 * 1024)
    return proc


# ── 单次解码全视频 224×224 → CPU uint8 ──────────────────────────
def _decode_all_frames(video, total_frames, width=224, height=224):
    """
    一次 ffmpeg 解码全视频到 224×224 → CPU numpy uint8 array (N, H, W, 3)

    关键: scale + crop 保持短边 resize + center crop，无黑边
    """
    vf = (
        "scale='if(gt(iw,ih),-2,{w})':'if(gt(iw,ih),{h},-2)',"
        "crop={w}:{h}"
    ).format(w=width, h=height)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-hwaccel", "cuda",
        "-i", video,
        "-vf", vf,
        "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-vframes", str(total_frames),
        "-",  # stdout
    ]

    expected = total_frames * height * width * 3
    t0 = time.time()
    proc = _run_ffmpeg(cmd)

    # 分段读取大 pipe，避免一次性分配 3.7GB
    CHUNK = 64 * 1024 * 1024  # 64MB chunks
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
        print(f"  WARNING: got {len(raw)}/{expected} bytes, padding zeros")
        raw = raw + b"\x00" * (expected - len(raw))

    arr = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(total_frames, height, width, 3).copy()
    del raw
    print(f"  decode: {total_frames}fr, {arr.nbytes/1e6:.0f}MB, {read_time:.1f}s")
    return arr


# ── 从 CPU 全帧 array 批量计算全局 MAFD ─────────────────────────
def _compute_global_mafd(all_frames, device="cuda"):
    """
    分批 GPU 处理: 224×224 → interpolate 160×90 → MAFD

    all_frames: (N, H, W, 3) uint8 CPU
    返回: (N-1,) float32 numpy, mafd[i] = |frame_i+1 - frame_i|.mean()
    """
    n = all_frames.shape[0]
    mafd_chunks = []

    t0 = time.time()
    for i in range(0, n - 1, GPU_BATCH):
        end = min(i + GPU_BATCH + 1, n)  # +1 因为需要 diff pair
        chunk = all_frames[i:end]  # (B+1, H, W, 3) uint8

        tensor = torch.from_numpy(chunk).to(device, dtype=torch.float32, non_blocking=True) / 255.0
        tensor = tensor.permute(0, 3, 1, 2)  # (B+1, 3, H, W)
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
    """find_peaks 切分 stable zones。返回 [(sf, ef, score), ...]"""
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

    zones = zones[:MAX_SUBSEG_PER_SEGMENT]
    return zones


# ── 单段处理（从 CPU 全帧 array + 全局 MAFD） ─────────────────
def process_segment_v3(idx, seg_start, seg_end,
                       all_frames_224, full_mafd,
                       device="cuda"):
    """
    处理一个 segment。不用 ffmpeg，从预加载的 CPU array 和 MAFD 计算。

    all_frames_224: (N, 224, 224, 3) uint8 CPU
    full_mafd: (N-1,) float32 numpy
    """
    t0 = time.time()
    n_frames = seg_end - seg_start + 1
    print(f"  [seg {idx}] {seg_start}-{seg_end} ({n_frames}fr)", end="", flush=True)

    # 切 segment 的 MAFD
    seg_mafd = full_mafd[seg_start:seg_end]  # n_frames-1

    # find_peaks
    zones = detect_stable_zones(seg_mafd, n_frames)

    # 对每个 zone 选代表帧
    result = []
    for rel_sf, rel_ef, trans_score in zones:
        abs_sf = seg_start + rel_sf
        abs_ef = seg_start + rel_ef
        zone_len = rel_ef - rel_sf + 1

        # zone 内的 MAFD
        zone_mafd = seg_mafd[rel_sf:rel_ef + 1]

        # stable_center = MAFD 最低点
        best_rel = int(np.argmin(zone_mafd))
        best_abs = abs_sf + best_rel  # 全局帧号

        # 候选帧: stable_center +/- CENTER_RANGE
        candidates = []
        for offset in range(-CENTER_RANGE, CENTER_RANGE + 1):
            cf = best_rel + offset
            if 0 <= cf < zone_len:
                candidates.append(abs_sf + cf)
        candidates = sorted(set(candidates))

        if len(candidates) == 1:
            result.append({
                "start_frame": abs_sf,
                "end_frame": abs_ef,
                "transition_score": round(float(trans_score), 4),
                "representative_frame": candidates[0],
            })
            continue

        # 三层评分（全 CPU，从预加载 array 取值）
        # stability (0.5)
        cand_rel = [c - abs_sf for c in candidates]  # relative to zone start
        # clamp: zone_mafd 长度 = zone_len - 1
        cand_mafd_idx = [min(r, len(zone_mafd) - 1) for r in cand_rel]
        stab = np.array([float(zone_mafd[i]) for i in cand_mafd_idx])
        stab = 1.0 / (1.0 + stab)

        # sharpness (0.3) — 从 224×224 array 取帧算 Laplacian
        sharp = np.array([
            cv2.Laplacian(
                cv2.cvtColor(all_frames_224[c], cv2.COLOR_RGB2GRAY),
                cv2.CV_64F
            ).var()
            for c in candidates
        ])
        sharp = sharp / (sharp.max() + 1e-6)

        # center_distance (0.2)
        dists = np.array([abs(c - best_abs) for c in candidates])
        dists = 1.0 / (1.0 + dists)

        # 加权总分
        scores = 0.5 * stab + 0.3 * sharp + 0.2 * dists
        best_candidate = candidates[int(np.argmax(scores))]

        result.append({
            "start_frame": abs_sf,
            "end_frame": abs_ef,
            "transition_score": round(float(trans_score), 4),
            "representative_frame": best_candidate,
        })

    elapsed = time.time() - t0
    print(f" -> {len(zones)} zones ({elapsed:.1f}s)")
    return result


# ── 主流程 ──────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video.mp4> [output_dir]")
        sys.exit(1)

    video = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(video), os.path.splitext(os.path.basename(video))[0] + "_v15mosugu")

    # 读 cuts.json
    cuts_path = os.path.join(output, "00_cuts", "cuts.json")
    if not os.path.exists(cuts_path):
        cuts_path = os.path.join(output, "01_raw_cuts", "raw_cuts.json")
    with open(cuts_path) as f:
        data = json.load(f)

    cuts = sorted(set(data["cuts"]))
    total_frames = data["total_frames"]
    video_path = data["video"]
    fps = data.get("fps", 30.0)

    # 构建 segments
    segments = []
    prev = 0
    for c in cuts:
        if c > prev:
            segments.append((len(segments), prev, c))
        prev = c + 1
    if prev < total_frames:
        segments.append((len(segments), prev, total_frames - 1))

    print(f"[segment_refiner v3] {len(segments)} segments, {total_frames} frames, {fps:.2f}fps")
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    # ═══ Phase 1: 一次解码全视频 224×224 ═══
    print()
    print("Phase 1/3: decoding all frames at 224×224 (short-side resize + center crop)...")
    all_frames = _decode_all_frames(video_path, total_frames, width=224, height=224)

    # ═══ Phase 2: 分批 GPU 算全局 MAFD ═══
    print()
    print("Phase 2/3: computing global MAFD curve...")
    full_mafd = _compute_global_mafd(all_frames, device=device)

    # ═══ Phase 3: 逐段处理（零 ffmpeg） ═══
    print()
    print("Phase 3/3: processing segments...")
    all_segments = []
    for idx, sf, ef in segments:
        zones = process_segment_v3(idx, sf, ef, all_frames, full_mafd, device)
        all_segments.append({
            "segment_id": idx,
            "start_frame": sf,
            "end_frame": ef,
            "sub_events": zones,
        })

    # 释放
    del all_frames, full_mafd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 输出
    result = {
        "video": os.path.abspath(video_path),
        "total_frames": total_frames,
        "n_segments": len(all_segments),
        "segments": all_segments,
    }
    out_dir = os.path.join(output, "keyframes")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "keyframes.json")
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total_zones = sum(len(s["sub_events"]) for s in all_segments)
    elapsed = time.time() - t_start
    print(f"\nDone: {len(all_segments)} segments, {total_zones} zones ({elapsed:.1f}s)")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
