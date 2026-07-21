#!/usr/bin/env python3
"""segment_cli_v2 — 逐段独立 NVENC，5线程并行，纯帧号驱动
   基于 segment_cli_orig.py 架构，适配 skeleton.json + FFmpeg 8.1"""

import json, subprocess, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_shot(shot, video, out_dir, fps, total):
    sid = shot["id"]
    sf = shot["range"]["start"]
    ef = shot["range"]["end"]
    n_frames = ef - sf + 1

    # 帧号→时间戳，仅用于 -ss 定位（帧数由 -frames:v 精确控制）
    t_start = sf / fps
    out = os.path.join(out_dir, f"segment_{sid:04d}.mp4")

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-ss", f"{t_start:.6f}",
        "-i", video,
        "-frames:v", str(n_frames),
        "-fps_mode", "passthrough",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-cq", "26",
        "-forced-idr", "1",
        "-an",
        out
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0

    # Verify
    if ok and os.path.exists(out):
        r2 = subprocess.run(["ffprobe", "-v", "quiet", "-count_frames",
            "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0", out], capture_output=True, text=True)
        try:
            got = int(r2.stdout.strip())
            if got != n_frames:
                ok = False
                print(f"  [{sid:3d}] FRAME MISMATCH: got={got} exp={n_frames}", flush=True)
        except:
            pass

    status = "OK" if ok else f"ERR"
    print(f"  [{sid:3d}] segment_{sid:04d}.mp4  {n_frames:4d}fr  {status}", flush=True)
    return ok

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <skeleton.json>", file=sys.stderr)
        sys.exit(1)

    skel_path = sys.argv[1]
    with open(skel_path) as f:
        skel = json.load(f)

    video = skel["video"]
    shots = skel["shots"]
    out_dir = os.path.dirname(skel_path) or "."

    # FPS
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", video],
        capture_output=True, text=True, check=True)
    num, den = map(int, r.stdout.strip().split("/"))
    fps = num / den

    print(f"Video: {video}")
    print(f"Shots: {len(shots)}  FPS: {num}/{den} = {fps:.6f}")
    print(f"Output: {out_dir}/")
    print(f"Workers: 5  Encoder: h264_nvenc (per-shot fresh instance)")
    print()

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(process_shot, s, video, out_dir, fps, len(shots)): s for s in shots}
        results = []
        for f in as_completed(futures):
            results.append(f.result())

    ok = sum(results)
    fail = len(results) - ok
    print(f"\nDone: {ok} OK  {fail} FAIL  (total {len(results)})")
    if fail > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
