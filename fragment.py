#!/usr/bin/env python3
"""04.5 — Fragment: 短 scene 补偿 (快切合并 + 视觉相似度选侧)。

输入:  06_merge/skeleton.json (shots[] + scenes[])
输出:  fragment/skeleton.json  (+ scenes[] 经碎片补偿)
       fragment/fragment_output.json

v3.0 改动:
  - sem() 从 ASR 关键词匹配改为视觉相似度
  - 删除 stops / re 模块
  - 新增 C1-C3 assert
"""
import json, sys, os, copy
import numpy as np
import torch

assert torch.cuda.is_available(), "CUDA 不可用，检查 torch 安装"

output = sys.argv[1]
in_path = os.path.join(output, "graph_merge", "skeleton.json")
with open(in_path) as f:
    skeleton = json.load(f)

out_dir = os.path.join(output, "fragment")
os.makedirs(out_dir, exist_ok=True)

scenes = skeleton.get("scenes", [])
fps = skeleton["fps"]
MIN_OK = int(3.0 * fps)
print(f"[fragment] {len(scenes)} scenes  min_ok={MIN_OK}fr  frag_thr={int(3.0*fps)}fr")

# ── 加载视觉图 ──
vis_graph_path = os.path.join(output, "dino_cluster", "shot_visual_graph.npy")
vis_graph = np.load(vis_graph_path).astype(np.float32) if os.path.isfile(vis_graph_path) else None
if vis_graph is None:
    print(f"  WARNING: visual graph 缺失，sem() 将返回 0.0")


def dur(s):
    return s["range"]["end"] - s["range"]["start"] + 1


def merge_into(L, R):
    R["range"]["start"] = L["range"]["start"]
    R["shot_ids"] = L.get("shot_ids", []) + R.get("shot_ids", [])


def merge_right(L, R):
    L["range"]["end"] = R["range"]["end"]
    L["shot_ids"].extend(R.get("shot_ids", []))


def sem(scene_a, scene_b):
    """两个 scene 的视觉相似度（shot 间均值）。"""
    if vis_graph is None:
        return 0.0
    sids_a = scene_a.get("shot_ids", [])
    sids_b = scene_b.get("shot_ids", [])
    if not sids_a or not sids_b:
        return 0.0
    sims = []
    for sa in sids_a:
        for sb in sids_b:
            if 0 <= sa < vis_graph.shape[0] and 0 <= sb < vis_graph.shape[1]:
                sims.append(float(vis_graph[sa, sb]))
    return float(np.mean(sims)) if sims else 0.0


def _assert_scenes(scenes):
    """C1-C3 出口断言。"""
    scenes.sort(key=lambda s: s["range"]["start"])
    # C1: 无重叠
    for i in range(len(scenes) - 1):
        assert scenes[i]["range"]["end"] <= scenes[i + 1]["range"]["start"]
    # C2: shot 唯一归属
    all_ids = []
    for s in scenes:
        all_ids.extend(s["shot_ids"])
    assert len(set(all_ids)) == len(all_ids)
    # C3: 无孤儿（所有 shot 恰好属于一个 scene）
    assert sorted(all_ids) == list(range(max(all_ids) + 1))
    print(f"  C1-C3 pass: {len(scenes)} scenes, {len(all_ids)} shots")


# ── 迭代碎片合并 ──
MAX_ITER = 100
merged = list(scenes)
for _iter in range(MAX_ITER):
    n = len(merged)
    if n < 2:
        break
    ds = [dur(s) for s in merged]
    ft = int(3.0 * fps)
    intr = [d < ft for d in ds]
    i = 0
    changed = False
    while i < n:
        if not intr[i]:
            i += 1
            continue
        cs = i
        while i < n and intr[i]:
            i += 1
        ce = i - 1
        ns = ce - cs + 1
        hl = cs > 0
        hr = ce < n - 1

        if ns == 1:
            idx = cs
            if hl and hr:
                ls = sem(merged[idx - 1], merged[idx])
                rs = sem(merged[idx], merged[idx + 1])
                if ls >= rs:
                    merge_into(merged[idx - 1], merged[idx])
                    merged.pop(idx)
                else:
                    merge_right(merged[idx], merged[idx + 1])
                    merged.pop(idx)
            elif hl:
                merge_into(merged[idx - 1], merged[idx])
                merged.pop(idx)
            elif hr:
                merge_right(merged[idx], merged[idx + 1])
                merged.pop(idx)
            else:
                break
            changed = True
            break
        else:
            mx = 0
            if hl:
                mx = max(mx, sem(merged[cs - 1], merged[cs]))
            for k in range(cs, ce):
                mx = max(mx, sem(merged[k], merged[k + 1]))
            if hr:
                mx = max(mx, sem(merged[ce], merged[ce + 1]))

            if mx < 1.0:
                td = sum(dur(merged[k]) for k in range(cs, ce + 1))
                if td >= MIN_OK:
                    cur = copy.deepcopy(merged[cs])
                    for m in range(cs + 1, ce + 1):
                        nt = merged[m]
                        cur["range"]["end"] = nt["range"]["end"]
                        cur["shot_ids"].extend(nt.get("shot_ids", []))
                    for _ in range(cs, ce):
                        merged.pop(cs + 1)
                    merged[cs] = cur
                    ls = sem(merged[cs - 1], merged[cs]) if cs > 0 else 0
                    rs = sem(merged[cs], merged[cs + 1]) if cs < len(merged) - 1 else 0
                    if ls >= rs and cs > 0:
                        merge_into(merged[cs - 1], merged[cs])
                        merged.pop(cs)
                    elif rs > ls and cs < len(merged) - 1:
                        merge_right(merged[cs], merged[cs + 1])
                        merged.pop(cs)
                else:
                    ls = sem(merged[cs - 1], merged[cs]) if hl else 0
                    rs = sem(merged[ce], merged[ce + 1]) if hr else 0
                    if ls >= rs and hl:
                        merge_into(merged[cs - 1], merged[cs])
                        merged.pop(cs)
                    elif rs > ls and hr:
                        merge_right(merged[ce], merged[ce + 1])
                        merged.pop(ce)
                    elif hl:
                        merge_into(merged[cs - 1], merged[cs])
                        merged.pop(cs)
                    elif hr:
                        merge_right(merged[ce], merged[ce + 1])
                        merged.pop(ce)
                changed = True
                break
            else:
                if hl and sem(merged[cs - 1], merged[cs]) >= 1.0:
                    merge_into(merged[cs - 1], merged[cs])
                    merged.pop(cs)
                    changed = True
                    break
                if hr and sem(merged[ce], merged[ce + 1]) >= 1.0:
                    merge_right(merged[ce], merged[ce + 1])
                    merged.pop(ce)
                    changed = True
                    break
                for k in range(cs, ce):
                    if sem(merged[k], merged[k + 1]) >= 1.0:
                        merge_into(merged[k], merged[k + 1])
                        merged.pop(k + 1)
                        changed = True
                        break
                if changed:
                    break
        if not changed:
            i += 1
    if not changed:
        break

# 重新编号
for idx, s in enumerate(merged):
    s["id"] = idx

print(f"  fragment: {len(scenes)} -> {len(merged)} scenes")

# C1-C3 出口断言
_assert_scenes(merged)

skeleton["scenes"] = merged
skel_out = os.path.join(out_dir, "skeleton.json")
with open(skel_out, "w") as f:
    json.dump(skeleton, f, ensure_ascii=False, indent=2)

frag_out = os.path.join(out_dir, "fragment_output.json")
with open(frag_out, "w") as f:
    json.dump({"step": "fragment", "n_before": len(scenes), "n_after": len(merged)},
              f, ensure_ascii=False, indent=2)

print(f"  -> {out_dir}/")
