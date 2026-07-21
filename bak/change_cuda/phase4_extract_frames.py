#!/usr/bin/env python3
"""
Phase 4: 事件帧图片抽取（GPU 加速）
====================================

读取 change_cuda_main 输出的 events.json，用 FFmpeg GPU 精确抽取代表帧。

用法:
  python3 phase4_extract_frames.py <events.json> <video.mp4> [--output-dir <dir>]

输出:
  <output_dir>/event_001/
    ├── frame_903.png        # middle 帧
    ├── frame_691.png        # medoid 帧（如果有）
    └── event_info.txt       # 该事件的时间窗口信息

依赖:
  - ffmpeg（需支持 h264_cuvid / CUDA）
"""

import json
import os
import sys
import subprocess
import argparse
from pathlib import Path


def extract_frames(events_json: str, video: str, output_dir: str = None):
    with open(events_json) as f:
        data = json.load(f)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(events_json), "event_frames")

    video_name = os.path.basename(data.get("video", video))
    fps = data.get("fps", 30.0)

    os.makedirs(output_dir, exist_ok=True)

    all_frame_indices = []      # all rep frames across events
    event_frame_map = {}        # frame_index -> event_id list
    frame_meta = {}             # frame_index -> (event_time_range, is_middle, is_medoid)

    for evt in data.get("events", []):
        eid = evt["id"]
        rep_frames = evt.get("representative_frames", [evt.get("center_frame", 0)])
        start_t = evt.get("start_time", 0)
        end_t = evt.get("end_time", 0)
        center_t = evt.get("center_time", 0)

        for i, fn in enumerate(rep_frames):
            if fn not in all_frame_indices:
                all_frame_indices.append(fn)
            if fn not in event_frame_map:
                event_frame_map[fn] = []
                frame_meta[fn] = {}
            event_frame_map[fn].append(eid)
            # 标记是 middle(0) 还是 medoid(1+)
            role = "middle" if i == 0 else "medoid"
            frame_meta[fn][eid] = {
                "role": role,
                "start_time": start_t,
                "end_time": end_t,
                "center_time": center_t,
            }

    if not all_frame_indices:
        print("[Phase 4] No representative frames to extract.")
        return

    # 用 FFmpeg select filter 批量提取所有代表帧
    # 模式: select='eq(n,FRAME1)+eq(n,FRAME2)+...'
    all_frame_indices.sort()
    select_parts = [f"eq(n\\,{fn})" for fn in all_frame_indices]
    select_expr = "+".join(select_parts)

    # GPU 加速提取
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-c:v", "h264_cuvid",
        "-i", video,
        "-vf", f"select='{select_expr}'",
        "-vsync", "0",
        "-q:v", "2",
        f"{output_dir}/%d.png",
        "-loglevel", "error",
    ]

    print(f"[Phase 4] Extracting {len(all_frame_indices)} frames from {len(data.get('events', []))} events...")
    print(f"  Frames: {all_frame_indices[:5]}... ({len(all_frame_indices)} total)")
    print(f"  FFmpeg: {' '.join(cmd[:6])} ... -vf select=... -vsync 0 {output_dir}/%d.png")
    sys.stdout.flush()

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[Phase 4] ERROR: ffmpeg failed with code {result.returncode}")
        print(f"  stderr: {result.stderr[:500]}")
        return

    # 重命名文件: ffmpeg 输出 1.png,2.png... 对应 all_frame_indices[0], all_frame_indices[1]...
    extracted = list(sorted(Path(output_dir).glob("*.png")))
    for i, fpath in enumerate(extracted):
        if i >= len(all_frame_indices):
            break
        fn = all_frame_indices[i]
        new_name = f"frame_{fn}.png"
        new_path = os.path.join(output_dir, new_name)
        os.rename(str(fpath), new_path)

    print(f"[Phase 4] Extracted {len(all_frame_indices)} frames to {output_dir}/")

    # 写 summary JSON
    summary = {
        "video": video_name,
        "fps": fps,
        "n_events": len(data.get("events", [])),
        "n_frames_extracted": len(all_frame_indices),
        "frames": {},
    }

    for fn in all_frame_indices:
        fname = f"frame_{fn}.png"
        fpath = os.path.join(output_dir, fname)
        exists = os.path.exists(fpath)
        size = os.path.getsize(fpath) if exists else 0
        summary["frames"][str(fn)] = {
            "file": fname,
            "exists": exists,
            "size": size,
            "events": event_frame_map.get(fn, []),
        }

    summary_path = os.path.join(output_dir, "extracted_frames.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Phase 4] Summary: {summary_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Extract event representative frames")
    parser.add_argument("events_json", help="events.json from change_cuda_main")
    parser.add_argument("video", help="Source video file")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: next to events.json)")
    args = parser.parse_args()

    extract_frames(args.events_json, args.video, args.output_dir)
