#!/usr/bin/env python3
"""segment_cli — GPU切段，-ss前置，5线程并发"""
import json, subprocess, os, sys
from concurrent.futures import ThreadPoolExecutor

def process_segment(idx, sf, ef, video, out_dir, fps):
    n = ef - sf + 1
    t = sf / fps
    out = f"{out_dir}/seg_{idx:04d}_f{sf:05d}-{ef:05d}.mp4"
    cmd = ["ffmpeg","-y","-hide_banner","-hwaccel","cuda",
           "-ss",f"{t:.6f}","-i",video,"-frames:v",str(n),
           "-c:v","h264_nvenc","-preset","p4","-cq","26",
           "-c:a","aac","-b:a","128k",out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0
    status = "OK" if ok else f"ERR: {r.stderr[-200:]}"
    print(f"  [{idx:4d}/{n_total}] {os.path.basename(out)} ({n}fr) {status}", flush=True)
    return ok

if __name__ == "__main__":
    video = sys.argv[1]; cuts_path = sys.argv[2]; out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    d = json.load(open(cuts_path))
    raw_cuts = sorted(d["cuts"]); total = d["total_frames"]; fps = float(d["fps"])
    # 去重：相邻切点间隔 ≤1 帧的只保留第一个（scdet 连续帧误检）
    cuts = []
    for i, c in enumerate(raw_cuts):
        if i == 0 or c - raw_cuts[i-1] > 1:
            cuts.append(c)
    if len(cuts) != len(raw_cuts):
        print(f"[filter] {len(raw_cuts)} -> {len(cuts)} cuts (removed {len(raw_cuts)-len(cuts)} adjacent duplicates)")
    segs = []; prev = 0
    for c in cuts:
        if c > prev: segs.append((len(segs), prev, c-1))
        prev = c
    if prev < total: segs.append((len(segs), prev, total-1))
    global n_total; n_total = len(segs)
    print(f"Total: {n_total} segments, fps={fps}, max_workers=5, encoder=h264_nvenc")
    with ThreadPoolExecutor(max_workers=5) as pool:
        fs = [pool.submit(process_segment, i, s, e, video, out_dir, fps) for i,s,e in segs]
        rs = [f.result() for f in fs]
    print(f"\nDone: {sum(rs)}/{n_total} segments OK")
