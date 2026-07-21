#!/usr/bin/env python3
"""
visual_cluster — 第一轮纯视觉聚类: shot → proto-scene

基于 DINO shot_visual_graph，只用视觉信号做 Connected Components，
提炼出 proto-scene。不引入文本/ASR 任何信号。

输入:  d50_dedup/skeleton.json (去重后的 shots)
       dino_cluster/shot_visual_graph.npy (N×N 余弦相似度)
输出:  visual_cluster/skeleton.json (+ proto_scenes[])

参数:
  VIS_THR    = 0.85  建边余弦阈值
  WINDOW     = 30    时间窗口（只连接相邻 30 个 shot 内的 pair）
  MIN_SCENE  = 1     最小 scene shot 数
"""

import json, os, sys
import numpy as np

VIS_THR = 0.85
WINDOW = 30
MIN_SCENE = 1


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


def build_components(graph, n, vis_thr, window):
    """在视觉图上建 CC。只连接窗口内且相似度达标的 pair。"""
    uf = UnionFind(n)
    edges = 0
    for i in range(n):
        for j in range(i + 1, min(n, i + window)):
            if graph[i, j] >= vis_thr:
                uf.union(i, j)
                edges += 1
    return uf, edges


def collect_scenes(uf, n):
    """从 UF 收集 component → scene 列表。"""
    comps = {}
    for i in range(n):
        root = uf.find(i)
        comps.setdefault(root, []).append(i)
    return list(comps.values())


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <output_dir> [vis_thr] [window]")
        sys.exit(1)

    output = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else VIS_THR
    window = int(sys.argv[3]) if len(sys.argv) > 3 else WINDOW

    skel_in = os.path.join(output, "d50_dedup", "skeleton.json")
    graph_path = os.path.join(output, "dino_cluster", "shot_visual_graph.npy")

    with open(skel_in) as f:
        skeleton = json.load(f)

    graph = np.load(graph_path)
    shots = skeleton["shots"]
    n = len(shots)

    print(f"[visual_cluster] {n} shots  vis_thr={thr}  window={window}")

    # ── CC 聚类 ──
    uf, edges = build_components(graph, n, thr, window)
    comps = collect_scenes(uf, n)
    comps.sort(key=lambda c: c[0])

    print(f"  edges={edges}  components={len(comps)}")

    # ── 构建 scene 列表 ──
    scenes = []
    for ci, comp in enumerate(comps):
        shot_ids = sorted(comp)
        sf = shots[shot_ids[0]]["range"]["start"]
        ef = shots[shot_ids[-1]]["range"]["end"]
        nfr = ef - sf + 1
        dur = nfr / skeleton["fps"]

        # 计算组件内平均相似度
        sims = []
        for i in range(len(shot_ids)):
            for j in range(i + 1, len(shot_ids)):
                sims.append(float(graph[shot_ids[i], shot_ids[j]]))
        mean_sim = float(np.mean(sims)) if sims else 1.0

        scenes.append({
            "id": ci,
            "shot_ids": shot_ids,
            "n_shots": len(shot_ids),
            "range": {"start": sf, "end": ef},
            "duration_s": round(dur, 1),
            "mean_visual_sim": round(mean_sim, 4),
        })

    # ── 统计 ──
    sizes = [sc["n_shots"] for sc in scenes]
    print(f"  scenes: {len(scenes)}  "
          f"size: min={min(sizes)}  max={max(sizes)}  "
          f"avg={np.mean(sizes):.1f}  median={np.median(sizes):.0f}")

    single = sum(1 for s in sizes if s == 1)
    print(f"  singletons: {single} ({100*single/len(scenes):.0f}%)")

    # ── 填入骨架 ──
    skeleton["scenes"] = scenes
    # 标记来源
    for s in shots:
        s["visual_cluster"] = {"method": "cc_visual_only",
                               "vis_thr": thr, "window": window}

    out_dir = os.path.join(output, "visual_cluster")
    os.makedirs(out_dir, exist_ok=True)

    skel_out = os.path.join(out_dir, "skeleton.json")
    with open(skel_out, "w") as f:
        json.dump(skeleton, f, ensure_ascii=False, indent=2)

    # 独立输出
    cluster_out = {
        "step": "visual_cluster",
        "vis_thr": thr,
        "window": window,
        "n_shots": n,
        "n_scenes": len(scenes),
        "singletons": single,
        "scenes": scenes,
    }
    cluster_path = os.path.join(out_dir, "visual_cluster.json")
    with open(cluster_path, "w") as f:
        json.dump(cluster_out, f, ensure_ascii=False, indent=2)

    print(f"  -> {out_dir}/")
    return skeleton


if __name__ == "__main__":
    main()
