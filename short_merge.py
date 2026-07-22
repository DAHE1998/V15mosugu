#!/usr/bin/env python3
"""
short_merge — 短镜头合并。

Phase 1: 连续 <MIN_DUR_S 的镜头合并成组（快剪蒙太奇）
Phase 2: 剩下的孤立短镜头合并到 DINO 视觉最近邻
"""

import json, os, sys
import numpy as np

MIN_DUR_S = 4.0
MIN_MERGE_COS = 0.3


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <output_dir>")
        sys.exit(1)

    output = sys.argv[1]
    with open(os.path.join(output, "d50_dedup", "skeleton.json")) as f:
        skeleton = json.load(f)
    graph = np.load(os.path.join(output, "dino_cluster", "shot_visual_graph.npy"))

    shots = skeleton["shots"]
    fps = skeleton["fps"]
    n = len(shots)
    print(f"[short_merge] {n} shots  min={MIN_DUR_S}s")

    # ── Phase 1: 连续短镜头成组 ──
    groups = []
    i = 0
    while i < n:
        dur = (shots[i]["range"]["end"] - shots[i]["range"]["start"] + 1) / fps
        if dur < MIN_DUR_S:
            g = [i]
            j = i + 1
            while j < n:
                d2 = (shots[j]["range"]["end"] - shots[j]["range"]["start"] + 1) / fps
                if d2 < MIN_DUR_S:
                    g.append(j)
                    j += 1
                else:
                    break
            groups.append(g)
            i = j
        else:
            groups.append([i])
            i += 1

    n_grp = len(groups)
    n_merged = n - n_grp
    print(f"  Phase 1: {n} shots → {n_grp} groups (merged {n_merged} consecutive short)")

    # ── Phase 1.5: 大组内人物变化点切分 ──
    # 读取人脸链判断主导人物变化
    face_map = {}
    fc_path = os.path.join(output, "face_continuity", "person_chains.json")
    if os.path.isfile(fc_path):
        with open(fc_path) as f:
            fc = json.load(f)
        for c in fc["chains"]:
            if c["n_shots"] >= 2:
                for sid in c["shots"]:
                    face_map.setdefault(sid, []).append(c["person_id"])

    split_groups = []
    for g in groups:
        if len(g) <= 3:
            split_groups.append(g)
            continue
        # 找组内人物变化点
        sub = [g[0]]
        for k in range(1, len(g)):
            prev_faces = set(face_map.get(g[k-1], []))
            curr_faces = set(face_map.get(g[k], []))
            # 去除无处不在的旁白脸
            prev_dom = prev_faces
            curr_dom = curr_faces
            # 人物变化: (a)两面都有不同主导人物 + DINO低 (b)一面有脸一面无 + DINO极低
            different = (prev_dom and curr_dom and not (prev_dom & curr_dom))
            lost_face = (prev_dom and not curr_dom) or (curr_dom and not prev_dom)
            dino_gap = float(graph[g[k-1], g[k]])
            face_change = (different and dino_gap < 0.6) or (lost_face and dino_gap < 0.4)
            if face_change:
                split_groups.append(sub)
                sub = [g[k]]
            else:
                sub.append(g[k])
        split_groups.append(sub)

    if len(split_groups) > len(groups):
        print(f"  Phase 1.5: split {len(split_groups)-len(groups)} groups at face changes → {len(split_groups)} groups")
        groups = split_groups

    # ── Phase 2: 孤立短镜头合并到最近邻 ──
    changed = True
    while changed:
        changed = False
        new = []
        i = 0
        while i < len(groups):
            g = groups[i]
            sf = shots[g[0]]["range"]["start"]
            ef = shots[g[-1]]["range"]["end"]
            dur = (ef - sf + 1) / fps

            if dur >= MIN_DUR_S:
                new.append(g)
                i += 1
                continue

            left_sim = float(graph[g[0], groups[i-1][-1]]) if i > 0 else -1
            right_sim = float(graph[g[-1], groups[i+1][0]]) if i < len(groups)-1 else -1
            max_sim = max(left_sim, right_sim)

            if max_sim < MIN_MERGE_COS:
                new.append(g)
                i += 1
                continue

            if left_sim >= right_sim and i > 0:
                new[-1].extend(g)
                changed = True
            elif right_sim > left_sim and i < len(groups)-1:
                g.extend(groups[i+1])
                new.append(g)
                i += 1
                changed = True
            else:
                new.append(g)
            i += 1
        groups = new

    # ── 输出 ──
    proto_scenes = []
    for ci, shot_ids in enumerate(groups):
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
    print(f"  scenes: {len(proto_scenes)}  max={max(sizes)}  avg={np.mean(sizes):.1f}")
    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    main()
