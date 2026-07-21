#!/usr/bin/env python3
"""
selector/scheduler.py — Candidate Scheduler (Section 8)

输入: change_curve.json
输出: candidates.json

8.1 Peak Candidate: find_peaks → change_peak
8.2 Stable Candidate: score < low_threshold for 5-10s → stable_sample
"""

import json
import os
import sys
import numpy as np
from scipy.signal import find_peaks


def find_peak_candidates(curve, distance=30, prominence=None, height=None):
    """
    8.1 Peak Candidate。

    Args:
        curve: list of float
        distance: 最小帧间隔 (默认30 = ~1s @ 30fps)
        prominence: 自适应 (None = 使用 curve 的 75th percentile)
        height: 最小高度 (None = 使用 curve 的 50th percentile)

    Returns:
        list of {"frame": int, "reason": "change_peak", "score": float}
    """
    arr = np.array(curve, dtype=np.float32)

    # 自适应阈值 (Section 8.1: "不要固定阈值")
    if prominence is None:
        prominence = float(np.percentile(arr[arr > 0], 75))
    if height is None:
        height = float(np.percentile(arr[arr > 0], 50))

    peaks, props = find_peaks(arr, distance=distance,
                               prominence=prominence, height=height)

    return [
        {"frame": int(pk), "reason": "change_peak", "score": round(float(arr[pk]), 6)}
        for pk in peaks
    ]


def find_stable_candidates(curve, fps=25.0, low_threshold=None,
                           min_dur_sec=5.0, max_dur_sec=10.0):
    """
    8.2 Stable Candidate。

    找连续 score < low_threshold 的区间，每 5-10 秒取一个代表帧。

    Args:
        curve: list of float
        fps: 帧率
        low_threshold: 低分阈值 (None = 自适应: 25th percentile)
        min_dur_sec: 最短稳定区间 (秒)
        max_dur_sec: 最长稳定区间 (秒)

    Returns:
        list of {"frame": int, "reason": "stable_sample", "score": float}
    """
    arr = np.array(curve, dtype=np.float32)
    n = len(arr)

    if low_threshold is None:
        low_threshold = float(np.percentile(arr[arr > 0], 25))

    min_frames = int(min_dur_sec * fps)
    max_frames = int(max_dur_sec * fps)

    candidates = []

    # 找连续低分区段
    i = 0
    while i < n:
        if arr[i] < low_threshold:
            start = i
            while i < n and arr[i] < low_threshold:
                i += 1
            end = i - 1
            dur = end - start + 1

            if dur >= min_frames:
                # 在稳定区间内均匀采样
                step = min(max_frames, dur) // 2  # 最多每 max_frames/2 取一个
                if step < 1:
                    step = 1
                for j in range(start, end + 1, step):
                    candidates.append({
                        "frame": j,
                        "reason": "stable_sample",
                        "score": round(float(arr[j]), 6)
                    })
        else:
            i += 1

    return candidates


def schedule_candidates(curve, fps=25.0, peak_params=None):
    """
    主函数: 生成所有候选。

    Args:
        curve: change curve (list of float)
        fps: 帧率
        peak_params: dict with distance/prominence/height

    Returns:
        dict with candidates and stats
    """
    if peak_params is None:
        peak_params = {}

    print(f"[scheduler] scheduling candidates from {len(curve)} frames ...")

    # 8.1 Peak candidates
    peaks = find_peak_candidates(
        curve,
        distance=peak_params.get("distance", 30),
        prominence=peak_params.get("prominence"),
        height=peak_params.get("height"),
    )
    print(f"  peak candidates: {len(peaks)}")

    # 8.2 Stable candidates
    stable = find_stable_candidates(curve, fps=fps)
    print(f"  stable candidates: {len(stable)}")

    # 合并去重
    all_candidates = peaks + stable
    seen = set()
    unique = []
    for c in all_candidates:
        if c["frame"] not in seen:
            seen.add(c["frame"])
            unique.append(c)

    unique.sort(key=lambda x: x["frame"])
    print(f"  total unique: {len(unique)}")

    return {
        "total_frames": len(curve),
        "fps": fps,
        "peak_candidates": len(peaks),
        "stable_candidates": len(stable),
        "total_candidates": len(unique),
        "candidates": unique,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("curve_json")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.curve_json) as f:
        data = json.load(f)

    curve = data["curve"]
    fps = args.fps or data.get("fps", 25.0)

    result = schedule_candidates(curve, fps=fps)

    out = args.out or args.curve_json.replace("change_curve.json", "candidates.json")
    with open(out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  -> {out}")
