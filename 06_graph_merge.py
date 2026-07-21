#!/usr/bin/env python3
"""06_graph_merge — Connected Component + triple check(mean/p10/gap) + MAX_GAP 切分。

输入:  05_text_cluster/skeleton.json
       04_dino_cluster/shot_visual_graph.npy   (N×N)
       05_text_cluster/shot_text_graph.npy      (N×N)
输出:  06_merge/skeleton.json  (+ scenes[])
       06_merge/graph_stats.json

用法:
  python 06_graph_merge.py <work_dir>
  MERGE_THR=0.78 VIS_W=0.7 TEXT_W=0.3 python 06_graph_merge.py <work_dir>"""
import json, sys, os, time, re
import numpy as np

# ── 参数 ──────────────────────────────────────────────────────────
MERGE_THR = float(os.environ.get("MERGE_THR", "0.78"))
THR_VIS   = float(os.environ.get("THR_VIS", "0.60"))
SPLIT_THR = float(os.environ.get("SPLIT_THR", "0.65"))
VIS_W     = float(os.environ.get("VIS_W", "0.8"))
TEXT_W    = float(os.environ.get("TEXT_W", "0.2"))
WINDOW    = int(os.environ.get("WINDOW", "30"))
MAX_GAP   = int(os.environ.get("MAX_GAP", "20"))
LOG_TAG   = "[06g]"


def _empty_asr(text):
    """ASR 是否为无意义内容."""
    t = (text or "").strip()
    if not t:
        return True
    meaningful = re.sub(r'[^一-鿿\w]', '', t)
    return len(meaningful) == 0


def _triple_check(members, vis_sim, text_sim, asr_empty):
    """
    GPT 三级判断:
      mean(sim) > 0.70
      AND percentile(sim, 10%) > 0.55
      AND max_gap < 30

    全部通过 → 保留 (-1, mean_sim)
    任一不通过 → 在最弱点拆分 (split_at, mean_sim)
    """
    n = len(members)
    if n < 2:
        return -1, 1.0

    weights = []
    min_w = 1.0
    split_at = -1
    max_gap = 0

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = members[i], members[j]
            vs = float(vis_sim[si, sj])
            ts = float(text_sim[si, sj])
            if asr_empty[si] or asr_empty[sj]:
                w = 1.0 * vs + 0.0 * ts
            else:
                w = VIS_W * vs + TEXT_W * ts
            weights.append(w)
            if w < min_w:
                min_w = w
                split_at = max(i, j)
        # max_gap between consecutive sorted members
        if i > 0:
            gap = members[i] - members[i - 1]
            if gap > max_gap:
                max_gap = gap

    if not weights:
        return -1, 1.0

    mean_sim = float(np.mean(weights))
    p10_sim = float(np.percentile(weights, 10))

    if mean_sim > 0.70 and p10_sim > 0.55 and max_gap < 30:
        return -1, mean_sim
    else:
        return split_at, mean_sim


def _split_component(members, vis_sim, text_sim, asr_empty):
    """
    递归拆解 component：MST 验证 → MAX_GAP → 递归。
    返回 scene 列表（每个 scene 是一个 shot 索引列表）。
    """
    members.sort()

    # 第一阶段：MAX_GAP 切分
    groups = []
    cur = [members[0]]
    for m in members[1:]:
        if m - cur[-1] > MAX_GAP:
            groups.append(cur)
            cur = [m]
        else:
            cur.append(m)
    groups.append(cur)

    # 第二阶段：GPT 三级验证 (mean, p10, max_gap)
    scenes = []
    for g in groups:
        if len(g) < 2:
            scenes.append(g)
            continue
        split_at, _ = _triple_check(g, vis_sim, text_sim, asr_empty)
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


def connected_component_merge(vis_graph, text_graph, shots):
    """Connected Component + WINDOW + Union Find + 内部验证 (MST + MAX_GAP)。"""
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

    asr_empty = [_empty_asr(s.get("asr_text")) for s in shots]
    n_edges = 0

    # ── WINDOW 内建边 ──
    for i in range(N):
        for j in range(i + 1, min(N, i + WINDOW + 1)):
            vis_sim = float(vis_graph[i, j])
            text_sim = float(text_graph[i, j])

            if asr_empty[i] or asr_empty[j]:
                weight = 1.0 * vis_sim + 0.0 * text_sim
                thr = THR_VIS
            else:
                weight = VIS_W * vis_sim + TEXT_W * text_sim
                thr = MERGE_THR

            if weight >= thr:
                union(i, j)
                n_edges += 1

    # ── Group by component ──
    comps = {}
    for i in range(N):
        r = find(i)
        comps.setdefault(r, []).append(i)

    # ── 每个 component 拆解 → scene ──
    all_scenes = []
    for root, members in comps.items():
        sub_scenes = _split_component(members, vis_graph, text_graph, asr_empty)
        all_scenes.extend(sub_scenes)

    all_scenes.sort(key=lambda x: x[0])

    # ── 构建 scene 对象 ──
    result = []
    for group in all_scenes:
        texts = []
        min_s = shots[group[0]]["range"]["start"]
        max_e = shots[group[0]]["range"]["end"]
        for idx in group:
            s = shots[idx]
            t = (s.get("asr_text") or "").strip()
            if t:
                texts.append(t.rstrip("。"))
            if s["range"]["start"] < min_s:
                min_s = s["range"]["start"]
            if s["range"]["end"] > max_e:
                max_e = s["range"]["end"]

        result.append({
            "id": len(result),
            "shot_ids": group,
            "range": {"start": min_s, "end": max_e},
            "representative_frame": shots[group[0]].get("representative_frame"),
            "asr_text": "。".join(texts),
        })

    return result, n_edges


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <work_dir>")
        sys.exit(1)

    work = sys.argv[1]
    in_path = os.path.join(work, "05_text_cluster", "skeleton.json")
    with open(in_path) as f:
        skeleton = json.load(f)

    shots = skeleton["shots"]
    N = len(shots)
    print(f"{LOG_TAG} {N} shots  MERGE_THR={MERGE_THR} WINDOW={WINDOW} "
          f"VIS_W={VIS_W} TEXT_W={TEXT_W} MAX_GAP={MAX_GAP} SPLIT_THR={SPLIT_THR}")

    vis_graph_path = os.path.join(work, "04_dino_cluster", "shot_visual_graph.npy")
    text_graph_path = os.path.join(work, "05_text_cluster", "shot_text_graph.npy")

    if not os.path.isfile(vis_graph_path) or not os.path.isfile(text_graph_path):
        print(f"{LOG_TAG} graph 文件缺失, 请先跑 04/05 (需含 N×N graph 输出)")
        sys.exit(1)

    vis_graph = np.load(vis_graph_path).astype(np.float32)
    text_graph = np.load(text_graph_path).astype(np.float32)
    print(f"{LOG_TAG} loaded visual graph: {vis_graph.shape}")
    print(f"{LOG_TAG} loaded text graph:  {text_graph.shape}")

    t0 = time.time()
    scenes, n_edges = connected_component_merge(vis_graph, text_graph, shots)
    dt = time.time() - t0

    n_scenes = len(scenes)
    sizes = [len(s["shot_ids"]) for s in scenes]
    if sizes:
        print(f"{LOG_TAG} {N} shots -> {n_scenes} scenes  ({n_edges} edges, {dt*1000:.0f}ms)")
        print(f"{LOG_TAG} sizes: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.1f}")
    else:
        print(f"{LOG_TAG} {N} shots -> {n_scenes} scenes (empty)")

    out_dir = os.path.join(work, "06_merge")
    os.makedirs(out_dir, exist_ok=True)

    skeleton["scenes"] = scenes
    with open(os.path.join(out_dir, "skeleton.json"), "w") as f:
        json.dump(skeleton, f, ensure_ascii=False, indent=2)

    stats = {
        "method": "cc_triple_check",
        "n_shots": N, "n_scenes": n_scenes, "n_edges": n_edges,
        "params": {
            "MERGE_THR": MERGE_THR, "THR_VIS": THR_VIS,
            "SPLIT_THR": SPLIT_THR,
            "VIS_W": VIS_W, "TEXT_W": TEXT_W,
            "WINDOW": WINDOW, "MAX_GAP": MAX_GAP,
        },
        "scene_sizes": sizes,
        "time_ms": round(dt * 1000),
    }
    with open(os.path.join(out_dir, "graph_stats.json"), "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"{LOG_TAG} -> {out_dir}/")


if __name__ == "__main__":
    main()
