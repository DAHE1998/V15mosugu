#!/usr/bin/env python3
"""
short_merge — D50后短段合并: <2s 的 shot 合并到视觉最近的相邻 shot

纯视觉判断，基于 DINO shot_visual_graph。
"""

import json, os, sys
import numpy as np

MIN_DUR_S = 2.0
MIN_MERGE_COS = 0.3  # 两边都低于此值 → 不合并，保留独立


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <output_dir>")
        sys.exit(1)

    output = sys.argv[1]

    d50_path = os.path.join(output, "d50_dedup", "skeleton.json")
    graph_path = os.path.join(output, "dino_cluster", "shot_visual_graph.npy")

    with open(d50_path) as f:
        skeleton = json.load(f)
    graph = np.load(graph_path)

    shots = skeleton["shots"]
    fps = skeleton["fps"]
    n = len(shots)
    min_fr = int(MIN_DUR_S * fps)

    print(f"[short_merge] {n} shots  min={MIN_DUR_S}s ({min_fr}fr)")

    # ── 初始每个 shot 一个 scene ──
    scenes = [[i] for i in range(n)]

    # ── 迭代合并 ──
    changed = True
    while changed:
        changed = False
        new = []
        i = 0
        while i < len(scenes):
            sc = scenes[i]
            sf = shots[sc[0]]["range"]["start"]
            ef = shots[sc[-1]]["range"]["end"]

            if (ef - sf + 1) >= min_fr:
                new.append(sc)
                i += 1
                continue

            # 短 scene，比左右邻居视觉相似度
            left_sim  = float(graph[sc[0], scenes[i-1][-1]]) if i > 0 else -1
            right_sim = float(graph[sc[-1], scenes[i+1][0]]) if i < len(scenes)-1 else -1

            # 两边都不像 → 保留为独立过渡段
            max_sim = max(left_sim, right_sim)
            if max_sim < MIN_MERGE_COS:
                new.append(sc)
                i += 1
                continue

            if left_sim >= right_sim and i > 0:
                new[-1].extend(sc)
                changed = True
            elif right_sim > left_sim and i < len(scenes)-1:
                sc.extend(scenes[i+1])
                new.append(sc)
                i += 1  # skip next
                changed = True
            else:
                new.append(sc)
            i += 1
        scenes = new

    # ── 输出 ──
    proto_scenes = []
    for ci, shot_ids in enumerate(scenes):
        sf = shots[shot_ids[0]]["range"]["start"]
        ef = shots[shot_ids[-1]]["range"]["end"]
        proto_scenes.append({
            "id": ci, "shot_ids": shot_ids, "n_shots": len(shot_ids),
            "range": {"start": sf, "end": ef},
            "duration_s": round((ef - sf + 1) / fps, 1),
        })

    skeleton["proto_scenes"] = proto_scenes

    out_dir = os.path.join(output, "short_merge")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "skeleton.json"), "w") as f:
        json.dump(skeleton, f, ensure_ascii=False, indent=2)

    sizes = [sc["n_shots"] for sc in proto_scenes]
    print(f"  scenes: {len(proto_scenes)}  merged: {n - len(proto_scenes)}  "
          f"max={max(sizes)}  avg={np.mean(sizes):.1f}")
    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    main()
