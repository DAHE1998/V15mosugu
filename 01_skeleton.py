#!/usr/bin/env python3
"""01 — 事件 → 骨架（镜头级）

从 change_cuda 输出构建 shots[] 骨架。
优先读 events，如果 events 为空则从 raw_cuts 推导镜头边界。

输出:
  01_skeleton/skeleton.json
  ├── video, total_frames, fps, duration
  └── shots[]
        ├── id, range{start,end}, head, tail
        └── representative_frame
"""
import json
import sys
import os

LOG_TAG = "[01]"


def build_from_cuts(d):
    """从 raw_cuts 的切点列表构建 shots。"""
    video = d["video"]
    tf = d["total_frames"]
    fps = d["fps"]
    cuts = sorted(set(d.get("cuts", d.get("events", []))))
    # 过滤无效切点
    valid = sorted({c for c in cuts if 0 < c < tf})
    bounds = [0] + valid + [tf]

    shots = []
    for i in range(len(bounds) - 1):
        sf = bounds[i]
        ef = bounds[i + 1] - 1
        head = sorted(set(range(sf, min(sf + 3, ef + 1))))
        tail = sorted(set(range(max(ef - 2, sf), ef + 1)))
        shots.append({
            "id": i,
            "range": {"start": sf, "end": ef},
            "head": head,
            "tail": tail,
            "representative_frame": (sf + ef) // 2,
        })
    return video, tf, fps, shots


def build_from_events(d):
    """从 events 数组构建 shots。"""
    video = d["video"]
    tf = d["total_frames"]
    fps = d["fps"]
    shots = []
    for evt in d["events"]:
        eid = evt["id"]
        sf = evt["start_frame"]
        ef = evt["end_frame"]
        head = sorted(set(range(sf, min(sf + 3, ef + 1))))
        tail = sorted(set(range(max(ef - 2, sf), ef + 1)))
        rep = evt.get("representative_frames", [None])[0]
        shots.append({
            "id": eid,
            "range": {"start": sf, "end": ef},
            "head": head,
            "tail": tail,
            "representative_frame": rep,
        })
    return video, tf, fps, shots


def main():
    if len(sys.argv) < 2:
        print(f"{LOG_TAG} 用法: 01_skeleton.py <work_dir>")
        print(f"{LOG_TAG}   work_dir = base_dir/video_name/")
        sys.exit(1)

    work = sys.argv[1]
    out_dir = os.path.join(work, "01_skeleton")
    os.makedirs(out_dir, exist_ok=True)

    # ── 尝试读 events.json → 若 events>0 则用 events 建 ──
    events_path = os.path.join(work, "00_scdet", "events.json")
    cuts_path = os.path.join(work, "00_scdet", "raw_cuts.json")
    # 兼容旧版 00_change_cuda 路径
    if not os.path.isfile(events_path):
        events_path = os.path.join(work, "00_change_cuda", "events.json")
        cuts_path = os.path.join(work, "00_change_cuda", "raw_cuts.json")

    video = None
    tf = None
    fps = None
    shots = []

    if os.path.isfile(events_path):
        with open(events_path) as f:
            d = json.load(f)
        if d.get("events") and len(d["events"]) > 0:
            video, tf, fps, shots = build_from_events(d)
            print(f"{LOG_TAG} {len(shots)} shots from {len(d['events'])} events")

    # ── 无 events → 从 raw_cuts 建 ──
    if not shots and os.path.isfile(cuts_path):
        with open(cuts_path) as f:
            d = json.load(f)
        video, tf, fps, shots = build_from_cuts(d)
        print(f"{LOG_TAG} {len(shots)} shots from {len(d.get('cuts', []))} cuts")

    if not shots:
        print(f"{LOG_TAG} 错误: 无 events 也无 cuts，无法建骨架")
        sys.exit(1)

    # ── 输出 ──
    skeleton = {
        "video": video,
        "total_frames": tf,
        "fps": fps,
        "duration": d.get("duration"),
        "width": d.get("width"),
        "height": d.get("height"),
        "shots": shots,
    }

    out_path = os.path.join(out_dir, "skeleton.json")
    with open(out_path, "w") as f:
        json.dump(skeleton, f, ensure_ascii=False, indent=2)

    filled = sum(1 for s in shots if s["representative_frame"] is not None)
    print(f"{LOG_TAG} -> {out_path} ({len(shots)} shots, {filled} with rep_frame)")


if __name__ == "__main__":
    main()
