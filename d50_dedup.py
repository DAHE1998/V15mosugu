#!/usr/bin/env python3
"""
4.5 — D50 段内去重（DINO embedding 余弦聚类）

读取 dino_cluster 的 key_frame_embeddings.npz，
逐 shot 做段内余弦贪心聚类，合并语义雷同帧，
只保留每个群落的 centroid 最近帧。

数据流:
  dino_cluster/skeleton.json → d50_dedup/skeleton.json

配置:
  COS_THRESHOLD = 0.95  (余弦相似度 > 此值 → 判为同群 → 去重)
"""

import json, os, sys, time
import numpy as np

COS_THRESHOLD = 0.95


def dedup_shot(key_frames, embeddings, frame_to_idx, threshold):
    """段内余弦贪心聚类去重，返回 (kept_frames, killed_frames)"""
    if len(key_frames) <= 1:
        return list(key_frames), []

    indices = np.array([frame_to_idx[f] for f in key_frames])
    sub = embeddings[indices]
    nrm = np.linalg.norm(sub, axis=1, keepdims=True)
    nrm[nrm == 0] = 1
    sub = sub / nrm

    # 贪心聚类
    clusters = []
    assigned = set()
    for i in range(len(key_frames)):
        if i in assigned:
            continue
        group = [i]
        for j in range(i + 1, len(key_frames)):
            if j in assigned:
                continue
            if float(sub[i].dot(sub[j])) > threshold:
                group.append(j)
                assigned.add(j)
        assigned.add(i)
        clusters.append(group)

    # 每群取离 centroid 最近的帧 → kept; 其余 → killed
    kept = []
    killed = []
    for grp in clusters:
        grp_emb = sub[grp]
        centroid = grp_emb.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        sims = grp_emb.dot(centroid)
        best_local = int(np.argmax(sims))
        for gi_idx, gi in enumerate(grp):
            if gi_idx == best_local:
                kept.append(key_frames[gi])
            else:
                killed.append(key_frames[gi])

    return sorted(kept), sorted(killed)


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <output_dir> [cos_threshold]")
        print(f"  output_dir 应包含 dino_cluster/skeleton.json")
        sys.exit(1)

    output = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else COS_THRESHOLD

    in_dir = os.path.join(output, "dino_cluster")
    skel_in = os.path.join(in_dir, "skeleton.json")
    emb_path = os.path.join(in_dir, "key_frame_embeddings.npz")

    if not os.path.isfile(skel_in):
        print(f"错误: 找不到 {skel_in}")
        sys.exit(1)
    if not os.path.isfile(emb_path):
        print(f"错误: 找不到 {emb_path}")
        print(f"  请确保 dino_cluster 输出了 key_frame_embeddings.npz")
        sys.exit(1)

    with open(skel_in) as f:
        skeleton = json.load(f)

    data = np.load(emb_path)
    embeddings = data["embeddings"].astype(np.float32)
    frame_ids = data["frame_ids"]
    frame_to_idx = {int(fn): i for i, fn in enumerate(frame_ids)}

    shots = skeleton["shots"]
    print(f"[d50] D50 段内去重  cos>{threshold}  {len(shots)} shots")

    total_kf = 0
    total_killed = 0
    deduped_shots = 0

    for s in shots:
        kf = s.get("key_frames", [])
        total_kf += len(kf)

        if len(kf) <= 1:
            continue

        # 确保帧号在 embedding 中
        valid_kf = [f for f in kf if f in frame_to_idx]
        if len(valid_kf) < 2:
            continue

        kept, killed = dedup_shot(valid_kf, embeddings, frame_to_idx, threshold)
        s["key_frames"] = kept

        if killed:
            s["_d50_killed"] = killed
            s["_d50_n_regions"] = len(kept)
            total_killed += len(killed)
            deduped_shots += 1
            print(f"  shot {s['id']:3d}: {len(kf)}→{len(kept)}  -{len(killed)}"
                  f"  kept={kept}  killed={killed}")

    # 清理临时字段（可选保留用于调试）
    # for s in shots: s.pop("_d50_killed", None); s.pop("_d50_n_regions", None)

    out_dir = os.path.join(output, "d50_dedup")
    os.makedirs(out_dir, exist_ok=True)

    skel_out = os.path.join(out_dir, "skeleton.json")
    with open(skel_out, "w") as f:
        json.dump(skeleton, f, ensure_ascii=False, indent=2)

    # 独立摘要
    summary = {
        "step": "d50_dedup",
        "cos_threshold": threshold,
        "total_key_frames": total_kf,
        "killed": total_killed,
        "kept": total_kf - total_killed,
        "deduped_shots": deduped_shots,
    }
    summary_path = os.path.join(out_dir, "dedup_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {total_kf}→{total_kf - total_killed} kf"
          f"  -{total_killed} ({deduped_shots} shots affected)")
    print(f"  skeleton: {skel_out}")
    print(f"  summary:  {summary_path}")


if __name__ == "__main__":
    main()
