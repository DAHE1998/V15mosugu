#!/usr/bin/env python3
"""06 — 综合聚类: 文本+视觉合并阈值合并。将相似 shot 合并为 scene 场景层。

输入:  05_text_cluster/skeleton.json (shots[] + asr/visual/text cluster)
输出:  06_merge/skeleton.json  (+ scenes[] 场景层)
       06_merge/merge_output.json
"""
import json, sys, os

output = sys.argv[1]
in_path = os.path.join(output, "05_text_cluster", "skeleton.json")
with open(in_path) as f:
    skeleton = json.load(f)

out_dir = os.path.join(output, "06_merge")
os.makedirs(out_dir, exist_ok=True)

shots = skeleton["shots"]

# 读取文本/视觉聚类边界分数
tb = {}
vb = {}
for s in shots:
    i = s["id"]
    tc = s.get("text_cluster", {})
    vc = s.get("visual_cluster", {})
    tb[i] = tc.get("next_similarity", 0) if tc else 0
    vb[i] = vc.get("next_similarity", 0) if vc else 0

TEXT_THR = float(os.environ.get("TEXT_THR", "0.50"))
VIS_THR = float(os.environ.get("VIS_THR", "0.50"))
MERGE_MODE = os.environ.get("MERGE_MODE", "and")
print(f"[06] {len(shots)} shots  thr={TEXT_THR}/{VIS_THR}  mode={MERGE_MODE}")


def decide(i):
    """合并决策: AND 模式下文本和视觉都达标才合并。"""
    tc = tb.get(i, 0)
    vc = vb.get(i, 0)
    if MERGE_MODE == "or":
        return tc >= TEXT_THR or vc >= VIS_THR
    return tc >= TEXT_THR and vc >= VIS_THR


# 逐 shot 合并为 scene
scenes = []
i = 0
while i < len(shots):
    cur = {
        "id": len(scenes),
        "shot_ids": [shots[i]["id"]],
        "range": {"start": shots[i]["range"]["start"], "end": shots[i]["range"]["end"]},
        "representative_frame": shots[i].get("representative_frame"),
        "asr_text": shots[i].get("asr_text", ""),
    }
    j = i + 1
    while j < len(shots) and decide(j - 1):
        nxt = shots[j]
        cur["shot_ids"].append(nxt["id"])
        cur["range"]["end"] = nxt["range"]["end"]
        # 合并 ASR 文本
        mt = cur.get("asr_text", "") or ""
        nt = nxt.get("asr_text", "") or ""
        if mt and nt:
            cur["asr_text"] = mt.rstrip("。") + "。" + nt
        elif nt:
            cur["asr_text"] = nt
        j += 1
    scenes.append(cur)
    i = j

print(f"  merged: {len(shots)} shots -> {len(scenes)} scenes")

# 骨架添加 scenes 层
skeleton["shots"] = shots
skeleton["scenes"] = scenes

skel_out = os.path.join(out_dir, "skeleton.json")
with open(skel_out, "w") as f:
    json.dump(skeleton, f, ensure_ascii=False, indent=2)

merge_out = os.path.join(out_dir, "merge_output.json")
with open(merge_out, "w") as f:
    json.dump({"step": "06_merge", "n_shots": len(shots), "n_scenes": len(scenes), "scenes": scenes},
              f, ensure_ascii=False, indent=2)

print(f"  -> {out_dir}/")
