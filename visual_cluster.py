#!/usr/bin/env python3
"""
visual_cluster — 第一轮纯视觉聚类: shot → proto-scene

基于 DINO 相邻 shot 余弦相似度，在相似度骤降处切分，
将 shot 合并为 proto-scene。纯视觉信号，不引入文本/ASR。

输入:  d50_dedup/skeleton.json (去重后的 shots)
       dino_cluster/skeleton.json (含 adjacent similarity)
输出:  visual_cluster/skeleton.json (+ proto_scenes[])
"""

import json, os, sys
import numpy as np

CUT_THR = 0.75  # 相邻 shot 相似度低于此值 → 切分
MIN_SCENE_FRAMES = 15  # 最短 scene（帧），短于此时合并到相邻


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <output_dir> [cut_threshold]")
        sys.exit(1)

    output = sys.argv[1]
    cut_thr = float(sys.argv[2]) if len(sys.argv) > 2 else CUT_THR

    d50_path = os.path.join(output, "d50_dedup", "skeleton.json")

    with open(d50_path) as f:
        skeleton = json.load(f)

    shots = skeleton["shots"]
    fps = skeleton["fps"]
    n = len(shots)

    # ── 收集相邻 shot 相似度 ──
    # 从 dino_cluster skeleton 读取 visual_cluster.next_similarity
    dino_path = os.path.join(output, "dino_cluster", "skeleton.json")
    with open(dino_path) as f:
        dino_sk = json.load(f)

    # dino_sk shots 含有 visual_cluster.next_similarity
    boundaries = []
    for i, s in enumerate(dino_sk["shots"][:n]):
        vc = s.get("visual_cluster", {})
        sim = vc.get("next_similarity")
        if sim is not None and sim < cut_thr:
            boundaries.append(i)

    print(f"[visual_cluster] {n} shots  cut_thr={cut_thr}  "
          f"boundaries={len(boundaries)}")

    # ── 按边界分组 ──
    scenes = []
    prev = 0
    for bi in boundaries:
        if bi + 1 > prev:
            scenes.append(list(range(prev, bi + 1)))
        prev = bi + 1
    if prev < n:
        scenes.append(list(range(prev, n)))

    # ── 合并过短 scene ──
    min_fr = int(MIN_SCENE_FRAMES)
    merged = []
    for sc in scenes:
        sf = shots[sc[0]]["range"]["start"]
        ef = shots[sc[-1]]["range"]["end"]
        dur_fr = ef - sf + 1

        if dur_fr < min_fr and len(merged) > 0:
            # 合并到上一个 scene
            merged[-1].extend(sc)
        else:
            merged.append(sc)

    # ── 构建输出 ──
    proto_scenes = []
    for ci, shot_ids in enumerate(merged):
        sf = shots[shot_ids[0]]["range"]["start"]
        ef = shots[shot_ids[-1]]["range"]["end"]
        dur = (ef - sf + 1) / fps

        proto_scenes.append({
            "id": ci,
            "shot_ids": shot_ids,
            "n_shots": len(shot_ids),
            "range": {"start": sf, "end": ef},
            "duration_s": round(dur, 1),
        })

    skeleton["proto_scenes"] = proto_scenes

    out_dir = os.path.join(output, "visual_cluster")
    os.makedirs(out_dir, exist_ok=True)

    skel_path = os.path.join(out_dir, "skeleton.json")
    with open(skel_path, "w") as f:
        json.dump(skeleton, f, ensure_ascii=False, indent=2)

    # 摘要
    summary = {
        "step": "visual_cluster",
        "cut_threshold": cut_thr,
        "n_shots": n,
        "n_boundaries": len(boundaries),
        "n_proto_scenes": len(proto_scenes),
        "proto_scenes": proto_scenes,
    }
    with open(os.path.join(out_dir, "visual_cluster.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    sizes = [sc["n_shots"] for sc in proto_scenes]
    single = sum(1 for s in sizes if s == 1)
    print(f"  proto-scenes: {len(proto_scenes)}  "
          f"singletons={single}  max_size={max(sizes)}  "
          f"avg_size={np.mean(sizes):.1f}")
    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    main()
