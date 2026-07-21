#!/usr/bin/env python3
"""
02 — 镜头代表帧精选（MAFD 稳定段提取）

读取 01_skeleton/skeleton.json 的 shots[]，
逐镜头用全 GPU 方式解码、MAFD 分析、稳定段提取、评分精选。

输出 per shot:
  - representative_frame: 最佳稳定段的最高分帧（int，向后兼容）
  - key_frames: 所有稳定段的代表帧列表

数据流:
  01_skeleton/skeleton.json → 02_select_frames/skeleton.json

核心计算:
  Phase 1: 一次 ffmpeg NVDEC 解码全部帧 224×224 → CPU uint8
  Phase 2: 分批 GPU interpolate 160×90 → 全局 MAFD 曲线
  Phase 3: 逐镜头 MAFD 低谷区 → 稳定段 → 评分 → key_frames
"""

import json, os, sys, time, subprocess as sp
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

# ── 参数 ────────────────────────────────────────────────────────
MAFD_H, MAFD_W = 90, 160
GPU_BATCH = 1000
CENTER_RANGE = 15


# ── 静默 ffmpeg ─────────────────────────────────────────────────
def _run_ffmpeg(cmd):
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=64 * 1024 * 1024)
    return proc


# ── 单次解码全视频 224×224 → CPU uint8 ──────────────────────────
def _decode_all_frames(video, total_frames, width=224, height=224):
    vf = (
        "scale='if(gt(iw,ih),-2,{w})':'if(gt(iw,ih),{h},-2)',"
        "crop={w}:{h}"
    ).format(w=width, h=height)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-hwaccel", "cuda",
        "-i", video,
        "-vf", vf,
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24",
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


# ── GPU Laplacian ───────────────────────────────────────────────
def _gpu_laplacian_var(frames_gpu):
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


# ── MAFD 稳定段检测 ────────────────────────────────────────────
def detect_stable_regions(seg_mafd, min_stable_frames):
    """
    GPT 规范实现：
    1. gaussian_filter1d(sigma=3) 平滑
    2. threshold = percentile(smooth, 35%)
    3. 稳定段 = smooth < threshold 的连续区域
    4. 丢弃长度 < min_stable_frames 的段

    返回: [(start, end), ...] 稳定段列表（帧序号从 0 开始，相对于 shot 起点）
    """
    if len(seg_mafd) < 2:
        return [(0, max(0, len(seg_mafd) - 1))]

    smooth = gaussian_filter1d(seg_mafd.astype(np.float64), sigma=3)
    threshold = float(np.percentile(smooth, 35))

    # 找连续稳定区域
    stable_mask = smooth < threshold
    regions = []
    i = 0
    while i < len(stable_mask):
        if stable_mask[i]:
            start = i
            while i < len(stable_mask) and stable_mask[i]:
                i += 1
            end = i - 1
            length = end - start + 1
            if length >= min_stable_frames:
                regions.append((start, end))
        else:
            i += 1

    # 如果没有稳定段，退回用最低 MAFD 的周围区域作为唯一稳定段
    if not regions:
        best = int(np.argmin(seg_mafd))
        half = max(min_stable_frames // 2, 1)
        start = max(0, best - half)
        end = min(len(seg_mafd) - 1, best + half)
        regions = [(start, end)]

    return regions


# ── 内容去重（同 shot 内多个 key_frame 画面雷同）──────────────────
def dedup_by_content(key_frame_scores, frames_224, mse_threshold=50):
    """
    MSE < mse_threshold（224x224 RGB uint8 尺度）视为内容雷同，
    同一雷同组内只保留评分最高的一帧。

    注意：这里按分数取代，而不是简单地"保留先遍历到的那个"——
    因为 key_frame_scores 在调用处是按 stable region 的时间顺序排列的，
    如果只按顺序去重，会系统性地偏向保留时间靠前的稳定段，
    即使后面画质更好的稳定段分数更高。
    """
    kept = []  # [fn, score]
    for fn, score in key_frame_scores:
        fn_img = frames_224[fn].astype(np.float32)
        dup_idx = None
        for i, (k_fn, k_score) in enumerate(kept):
            mse = float(np.mean((fn_img - frames_224[k_fn].astype(np.float32)) ** 2))
            if mse < mse_threshold:
                dup_idx = i
                break
        if dup_idx is None:
            kept.append([fn, score])
        elif score > kept[dup_idx][1]:
            kept[dup_idx] = [fn, score]
    return [(fn, score) for fn, score in kept]


# ── GPU 单镜头处理 ──────────────────────────────────────────────
def process_shot_gpu(shot_id, seg_start, seg_end,
                     all_frames_224, full_mafd, device, fps):
    t0 = time.time()
    n_frames = seg_end - seg_start + 1
    dur_s = n_frames / fps
    min_stable = int(0.5 * fps)
    if dur_s < 2:       max_kf = 2
    elif dur_s < 30:    max_kf = 4
    else:               max_kf = 8
    print(f"  [shot {shot_id}] {seg_start}-{seg_end} ({n_frames}fr)", end="", flush=True)

    seg_mafd = full_mafd[seg_start:seg_end]
    regions = detect_stable_regions(seg_mafd, min_stable)
    regions = regions[:max_kf]

    key_frame_scores = []
    for rel_sf, rel_ef in regions:
        abs_sf = seg_start + rel_sf
        abs_ef = seg_start + rel_ef
        zone_len = rel_ef - rel_sf + 1
        zone_mafd = seg_mafd[rel_sf:rel_ef + 1]

        # 候选帧: MAFD 最低点附近 ±CENTER_RANGE
        best_rel = int(np.argmin(zone_mafd))
        best_abs = abs_sf + best_rel

        candidates = []
        for offset in range(-CENTER_RANGE, CENTER_RANGE + 1):
            cf = best_rel + offset
            if 0 <= cf < zone_len:
                candidates.append(abs_sf + cf)
        candidates = sorted(set(candidates))

        if len(candidates) == 1:
            key_frame_scores.append((candidates[0], 0.0))
            continue

        # GPU 评分
        cand_np = all_frames_224[candidates]
        cand_gpu = torch.from_numpy(cand_np).to(device, dtype=torch.float32, non_blocking=True)
        del cand_np
        cand_gpu = cand_gpu.permute(0, 3, 1, 2) / 255.0

        # 稳定性分: 越低 MAFD 越高分
        cand_rel_list = [c - abs_sf for c in candidates]
        cand_mafd_idx = [min(r, len(zone_mafd) - 1) for r in cand_rel_list]
        stab_vals = torch.tensor([zone_mafd[i] for i in cand_mafd_idx],
                                  device=device, dtype=torch.float32)
        stab_max = stab_vals.max()
        stab = 1.0 - (stab_vals / (stab_max + 1e-6))

        # 锐度分: Laplacian variance
        sharp_raw = _gpu_laplacian_var(cand_gpu)
        sharp = sharp_raw / (sharp_raw.max() + 1e-6)

        # 亮度有效分: 避开全黑/全白帧
        brightness = cand_gpu.mean(dim=(1, 2, 3))
        brightness_valid = torch.clamp(
            1.0 - torch.abs(brightness - 0.5) * 3.0,
            0.0, 1.0
        )

        # 综合评分 (GPT 规范)
        scores = 0.5 * stab + 0.3 * sharp + 0.2 * brightness_valid
        best_idx = int(scores.argmax().cpu())
        key_frame_scores.append((int(candidates[best_idx]), float(scores[best_idx].cpu())))

        del cand_gpu, stab, sharp_raw, sharp, brightness, brightness_valid, scores

    torch.cuda.empty_cache()

    if not key_frame_scores:
        key_frame_scores = [((seg_start + seg_end) // 2, 0.0)]

    # --- 内容去重：同一 shot 内多个稳定段选出的帧如果画面雷同，只留分高的 ---
    key_frame_scores = dedup_by_content(key_frame_scores, all_frames_224, mse_threshold=50)
    # --- 结束 ---

    # 按帧号排序（保持时间升序）
    key_frame_scores.sort(key=lambda x: x[0])
    key_frames = [f for f, _ in key_frame_scores]

    # representative_frame = 评分最高的帧（不是帧号最小的）
    rep_frame = max(key_frame_scores, key=lambda x: x[1])[0]

    elapsed = time.time() - t0
    print(f" -> {len(regions)} stable regions, {len(key_frames)} key_frames ({elapsed:.1f}s)")
    return rep_frame, key_frames


# ── 主流程 ──────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <output_dir>")
        print(f"  output_dir 应包含 01_skeleton/skeleton.json")
        sys.exit(1)

    output = sys.argv[1]
    skeleton_path = os.path.join(output, "01_skeleton", "skeleton.json")

    if not os.path.isfile(skeleton_path):
        print(f"错误: 找不到 {skeleton_path}")
        print(f"请先运行 01_skeleton.py")
        sys.exit(1)

    with open(skeleton_path) as f:
        skeleton = json.load(f)

    shots = skeleton["shots"]
    total_frames = skeleton["total_frames"]
    video_path = skeleton["video"]
    fps = skeleton["fps"]

    if not os.path.isfile(video_path):
        candidate = os.path.join(output, os.path.basename(video_path))
        if os.path.isfile(candidate):
            video_path = candidate

    print(f"[02] {len(shots)} shots, {total_frames} frames, {fps:.2f}fps")
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    # ═══ Phase 1: 一次解码 ═══
    print()
    print("Phase 1/3: decoding all frames 224×224...")
    all_frames = _decode_all_frames(video_path, total_frames)

    # ═══ Phase 2: 全局 MAFD ═══
    print()
    print("Phase 2/3: global MAFD on GPU...")
    full_mafd = _compute_global_mafd(all_frames, device=device)

    # ═══ Phase 3: 逐镜头稳定段提取 ═══
    print()
    print("Phase 3/3: extracting stable key_frames (GPU scoring)...")
    for shot in shots:
        sf = shot["range"]["start"]
        ef = shot["range"]["end"]
        rep_frame, key_frames = process_shot_gpu(
            shot["id"], sf, ef, all_frames, full_mafd, device, fps)
        shot["representative_frame"] = rep_frame
        shot["key_frames"] = key_frames

    del all_frames, full_mafd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 更新骨架 ──
    skeleton["shots"] = shots

    out_dir = os.path.join(output, "02_select_frames")
    os.makedirs(out_dir, exist_ok=True)

    skel_path = os.path.join(out_dir, "skeleton.json")
    with open(skel_path, "w") as f:
        json.dump(skeleton, f, ensure_ascii=False, indent=2)

    # 独立输出（断点检查用）
    refined = {
        "step": "02_select_frames",
        "n_shots": len(shots),
        "shots": [{"id": s["id"], "range": s["range"],
                    "representative_frame": s["representative_frame"],
                    "key_frames": s.get("key_frames", [])}
                  for s in shots],
    }
    ref_path = os.path.join(out_dir, "refined.json")
    with open(ref_path, "w") as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)

    total_kf = sum(len(s.get("key_frames", [1])) for s in shots)
    elapsed = time.time() - t_start
    print(f"\nDone: {len(shots)} shots, {total_kf} total key_frames ({elapsed:.1f}s)")
    print(f"  skeleton: {skel_path}")
    print(f"  refined:  {ref_path}")

    return skeleton


if __name__ == "__main__":
    main()