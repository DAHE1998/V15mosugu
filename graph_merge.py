#!/usr/bin/env python3
"""04 — Scene Merge: 纯视觉 Connected Component + 序号连续性 + 三级验证。

输入:  dino_cluster/skeleton.json (shots[] + range)
       dino_cluster/shot_visual_graph.npy   (N×N, 仅此一个信号源)
输出:  06_merge/skeleton.json  (+ scenes[])
       06_merge/graph_stats.json

v3.0 改动:
  - 删除 TEXT_W, 纯视觉建边 (VIS_W=1.0)
  - 连续性检查改 shot 序号间隔: 序号间隔 != 1 一律拆开
  - 删除帧距 MAX_GAP 阈值
  - 新增 C1-C5 assert
  - 不读取任何 asr / text_graph 文件

用法:
  python graph_merge.py <work_dir>
  MERGE_THR=0.78 THR_VIS=0.60 python graph_merge.py <work_dir>
"""
import json, sys, os, time
import numpy as np
import torch

assert torch.cuda.is_available(), "CUDA 不可用，检查 torch 安装"

# ── 参数 ──────────────────────────────────────────────────────────
MERGE_THR = float(os.environ.get(MERGE_THR, 0.78))
THR_VIS   = float(os.environ.get(THR_VIS, 0.60))
WINDOW    = int(os.environ.get(WINDOW, 30))
LOG_TAG   = "[04sm]"


def _triple_check(members, vis_sim):
    """
    三级验证:
      mean(sim) > 0.70
      AND percentile(sim, 10%) > 0.55

    全部通过 -> 保留 (-1, mean_sim)
    任一不通过 -> 在最弱点拆分 (split_at, mean_sim)
    """
    n = len(members)
    if n < 2:
        return -1, 1.0

    weights = []
    min_w = 1.0
    split_at = -1

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = members[i], members[j]
            w = float(vis_sim[si, sj])
            weights.append(w)
            if w < min_w:
                min_w = w
                split_at = max(i, j)

    if not weights:
        return -1, 1.0

    mean_sim = float(np.mean(weights))
    p10_sim = float(np.percentile(weights, 10))

    if mean_sim > 0.70 and p10_sim > 0.55:
        return -1, mean_sim
    else:
        return split_at, mean_sim


def _split_by_continuity(members, vis_sim):
    """
    按 shot 序号连续性拆解 component。
    序号间隔 != 1 的位置一律拆开（硬约束）。
    然后对每个连续区间跑三级验证。
    """
    members.sort()

    # 第一阶段: shot 序号连续性检查 (硬约束)
    groups = []
    cur = [members[0]]
    for m in members[1:]:
        if m - cur[-1] != 1:          # 序号间隔 != 1 -> 拆开
            groups.append(cur)
            cur = [m]
        else:
            cur.append(m)
    groups.append(cur)

    # 第二阶段: 三级验证 (mean, p10)
    scenes = []
    for g in groups:
        if len(g) < 2:
            scenes.append(g)
            continue
        split_at, _ = _triple_check(g, vis_sim)
        if split_at < 0:
            scenes.append(g)
        else:
            left = g[:split_at]
            right = g[split_at:]
            if left:
                scenes.append(left)
            if right:
                scenes.append(right)

    scenes.sort(key=lambda x: x[0])
    return scenes


def connected_component_merge(vis_graph, shots):
    """Union Find + WINDOW + 序号连续性 + 三级验证。"""
    N = vis_graph.shape[0]
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    n_edges = 0

    # WINDOW 内纯视觉建边
    for i in range(N):
        for j in range(i + 1, min(N, i + WINDOW + 1)):
            vis_sim = float(vis_graph[i, j])
            if vis_sim >= MERGE_THR:
                union(i, j)
                n_edges += 1

    # Group by component
    comps = {}
    for i in range(N):
        r = find(i)
        comps.setdefault(r, []).append(i)

    # 每个 component 拆解 -> scene
    all_scenes = []
    for root, members in comps.items():
        sub_scenes = _split_by_continuity(members, vis_graph)
        all_scenes.extend(sub_scenes)

    all_scenes.sort(key=lambda x: x[0])

    # 构建 scene 对象
    result = []
    for group in all_scenes:
        min_s = shots[group[0]]["range"]["start"]
        max_e = shots[group[0]]["range"]["end"]
        for idx in group:
            s = shots[idx]
            if s["range"]["start"] < min_s:
                min_s = s["range"]["start"]
            if s["range"]["end"] > max_e:
                max_e = s["range"]["end"]

        result.append({
            "id": len(result),
            "shot_ids": group,
            "range": {"start": min_s, "end": max_e},
            "representative_frame": shots[group[0]].get("representative_frame"),
        })

    return result, n_edges


def _assert_scenes(scenes, shots):
    """C1-C5 出口断言。"""
    scenes.sort(key=lambda s: s["range"]["start"])

    # C1: 无重叠
    for i in range(len(scenes) - 1):
        assert scenes[i]["range"]["end"] <= scenes[i + 1]["range"]["start"], \
            f"C1 失败: scene[{i}] end={scenes[i]['range']['end']} > scene[{i+1}] start={scenes[i+1]['range']['start']}"

    # C2: shot 序号连续性（无跳号）
    all_shot_ids = []
    for s in scenes:
        all_shot_ids.extend(s["shot_ids"])
    assert sorted(all_shot_ids) == list(range(max(all_shot_ids) + 1)), \
        f"C2 失败: 存在跳号，shot_ids 不连续"

    # C3: shot 唯一归属
    assert len(set(all_shot_ids)) == len(all_shot_ids), \
        f"C3 失败: 存在重复归属的 shot"

    # C4: range 与 shot_ids 一致
    for s in scenes:
        shot_starts = [shots[sid]["range"]["start"] for sid in s["shot_ids"]]
        shot_ends = [shots[sid]["range"]["end"] for sid in s["shot_ids"]]
        assert s["range"]["start"] == min(shot_starts), \
            f"C4 失败: scene[{s['id']}] range.start 与 shot_ids 不一致"
        assert s["range"]["end"] == max(shot_ends), \
            f"C4 失败: scene[{s['id']}] range.end 与 shot_ids 不一致"

    # C5: 所有 shot 恰好属于一个 scene
    assert sorted(all_shot_ids) == list(range(len(shots))), \
        f"C5 失败: 存在孤儿 shot 或遗漏"

    print(f"  C1-C5 pass: {len(scenes)} scenes, {len(all_shot_ids)} shots")


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <work_dir>")
        sys.exit(1)

    work = sys.argv[1]
    in_path = os.path.join(work, "dino_cluster", "skeleton.json")
    with open(in_path) as f:
        skeleton = json.load(f)

    shots = skeleton["shots"]
    N = len(shots)
    print(f"{LOG_TAG} {N} shots  MERGE_THR={MERGE_THR} WINDOW={WINDOW}")

    vis_graph_path = os.path.join(work, "dino_cluster", "shot_visual_graph.npy")

    if not os.path.isfile(vis_graph_path):
        print(f"{LOG_TAG} visual graph 文件缺失, 请先跑 04")
        sys.exit(1)

    vis_graph = np.load(vis_graph_path).astype(np.float32)
    print(f"{LOG_TAG} loaded visual graph: {vis_graph.shape}")

    t0 = time.time()
    scenes, n_edges = connected_component_merge(vis_graph, shots)
    dt = time.time() - t0

    # C1-C5 出口断言
    _assert_scenes(scenes, shots)

    n_scenes = len(scenes)
    sizes = [len(s["shot_ids"]) for s in scenes]
    if sizes:
        print(f"{LOG_TAG} {N} shots -> {n_scenes} scenes  ({n_edges} edges, {dt*1000:.0f}ms)")
        print(f"{LOG_TAG} sizes: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.1f}")
    else:
        print(f"{LOG_TAG} {N} shots -> {n_scenes} scenes (empty)")

    out_dir = os.path.join(work, "graph_merge")
    os.makedirs(out_dir, exist_ok=True)

    skeleton["scenes"] = scenes
    with open(os.path.join(out_dir, "skeleton.json"), "w") as f:
        json.dump(skeleton, f, ensure_ascii=False, indent=2)

    stats = {
        "method": "cc_continuity_triple_check",
        "n_shots": N, "n_scenes": n_scenes, "n_edges": n_edges,
        "params": {
            "MERGE_THR": MERGE_THR, "THR_VIS": THR_VIS,
            "WINDOW": WINDOW,
        },
        "scene_sizes": sizes,
        "time_ms": round(dt * 1000),
    }
    with open(os.path.join(out_dir, "graph_stats.json"), "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"{LOG_TAG} -> {out_dir}/")


if __name__ == "__main__":
    main()
