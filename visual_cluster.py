#!/usr/bin/env python3
"""
visual_cluster — 第一轮纯视觉聚类: shot → proto-scene

基于 DINO shot_visual_graph (N×N 余弦相似度矩阵)，
Union-Find Connected Components，只用视觉信号。

输入:  d50_dedup/skeleton.json (去重后的 shots)
       dino_cluster/shot_visual_graph.npy (N×N)
输出:  visual_cluster/skeleton.json (+ proto_scenes[])
"""

import json, os, sys
import numpy as np

VIS_THR = 0.70   # 建边余弦阈值
WINDOW  = 30     # 时间窗口（只连窗口内的 shot pair）
MIN_SCENE_FRAMES = 15  # 最短场景帧数，短于此合并到相邻


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1
        return True


def cc_cluster(graph, n, thr, window):
    uf = UnionFind(n)
    edges = 0
    for i in range(n):
        for j in range(i + 1, min(n, i + window)):
            if float(graph[i, j]) >= thr:
                uf.union(i, j)
                edges += 1
    comps = {}
    for i in range(n):
        root = uf.find(i)
        comps.setdefault(root, []).append(i)
    return list(comps.values()), edges


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <output_dir> [vis_thr] [window]")
        sys.exit(1)

    output = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else VIS_THR
    window = int(sys.argv[3]) if len(sys.argv) > 3 else WINDOW

    d50_path = os.path.join(output, "d50_dedup", "skeleton.json")
    graph_path = os.path.join(output, "dino_cluster", "shot_visual_graph.npy")

    with open(d50_path) as f:
        skeleton = json.load(f)
    graph = np.load(graph_path)

    shots = skeleton["shots"]
    fps = skeleton["fps"]
    n = len(shots)

    print(f"[visual_cluster] {n} shots  thr={thr}  window={window}")

    # ── CC 聚类 ──
    comps, edges = cc_cluster(graph, n, thr, window)
    comps.sort(key=lambda c: c[0])
    print(f"  edges={edges}  components={len(comps)}")

    # ── 构建 proto-scenes ──
    proto_scenes = []
    for ci, comp in enumerate(comps):
        shot_ids = sorted(comp)
        sf = shots[shot_ids[0]]["range"]["start"]
        ef = shots[shot_ids[-1]]["range"]["end"]
        dur = (ef - sf + 1) / fps

        # 组件内平均相似度
        sims = [float(graph[i, j])
                for a in range(len(shot_ids))
                for b in range(a + 1, len(shot_ids))
                for i in [shot_ids[a]] for j in [shot_ids[b]]]
        mean_sim = float(np.mean(sims)) if sims else 1.0

        proto_scenes.append({
            "id": ci,
            "shot_ids": shot_ids,
            "n_shots": len(shot_ids),
            "range": {"start": sf, "end": ef},
            "duration_s": round(dur, 1),
            "mean_visual_sim": round(mean_sim, 4),
        })

    # ── 合并过短 scene ──
    min_fr = int(MIN_SCENE_FRAMES)
    merged = []
    for sc in proto_scenes:
        dur_fr = sc["range"]["end"] - sc["range"]["start"] + 1
        if dur_fr < min_fr and merged:
            # 合并到上一个
            prev = merged[-1]
            prev["shot_ids"].extend(sc["shot_ids"])
            prev["range"]["end"] = sc["range"]["end"]
            prev["duration_s"] = round(
                (prev["range"]["end"] - prev["range"]["start"] + 1) / fps, 1)
            prev["n_shots"] = len(prev["shot_ids"])
        else:
            merged.append(sc)

    skeleton["proto_scenes"] = merged

    out_dir = os.path.join(output, "visual_cluster")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "skeleton.json"), "w") as f:
        json.dump(skeleton, f, ensure_ascii=False, indent=2)

    summary = {
        "step": "visual_cluster",
        "vis_threshold": thr,
        "window": window,
        "n_shots": n,
        "n_components": len(comps),
        "n_proto_scenes_before_merge": len(proto_scenes),
        "n_proto_scenes": len(merged),
        "proto_scenes": merged,
    }
    with open(os.path.join(out_dir, "visual_cluster.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    sizes = [sc["n_shots"] for sc in merged]
    single = sum(1 for s in sizes if s == 1)
    print(f"  proto-scenes: {len(merged)}  "
          f"singletons={single}  "
          f"max_size={max(sizes) if sizes else 0}  "
          f"avg_size={np.mean(sizes):.1f}")
    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    main()
