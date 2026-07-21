#!/usr/bin/env python3
"""08 — 章节聚类: 基于 VLM profile 的场景相似度聚类 → chapters 填入骨架。

输入:  vlm/skeleton.json (shots[] + scenes[] + scene.profile)
输出:  08_chapter/skeleton.json  (+ chapters[] 章节层)
       08_chapter/chapter_output.json
"""
import json, sys, os
from exp5_event_merge_v2 import build_cluster_profile, compute_merge_score

output = sys.argv[1]
in_path = os.path.join(output, "vlm", "skeleton.json")
with open(in_path) as f:
    skeleton = json.load(f)

out_dir = os.path.join(output, "chapter_old")
os.makedirs(out_dir, exist_ok=True)

scenes = skeleton.get("scenes", [])
print(f"[08] 章节聚类: {len(scenes)} scenes")

# 构建 profile 字典
profiles = {}
for s in scenes:
    sid = s["id"]
    profiles[sid] = {
        "start_frame": s["range"]["start"],
        "end_frame": s["range"]["end"],
        "profile": s.get("profile", {}),
        "asr_text": s.get("asr_text", ""),
        "shot_ids": s.get("shot_ids", []),
    }

keys = sorted(profiles.keys(), key=lambda k: profiles[k]["start_frame"])

scene_p = {sk: build_cluster_profile([sk], profiles) for sk in keys}

THR = 0.17
chapters = []
cur = [keys[0]]

for i in range(1, len(keys)):
    sk = keys[i]
    prev = cur[-1]
    sc, det = compute_merge_score(scene_p[prev], scene_p[sk])
    if sc < THR:
        chapters.append(cur)
        cur = [sk]
    else:
        cur.append(sk)
if cur:
    chapters.append(cur)

# 构建 chapters 层
chapter_list = []
for ci, ch in enumerate(chapters):
    first_s = profiles[ch[0]]
    last_s = profiles[ch[-1]]
    # 统计场景中的主要 environment 和 topic
    locs = set()
    topics = []
    for sk in ch:
        pf = profiles[sk].get("profile", {})
        if pf.get("environment"):
            locs.add(pf["environment"])
        topics.append(pf.get("topic", "") or "")
    main_topic = max(set(topics), key=topics.count)[:30] if topics else ""

    chapter_list.append({
        "id": ci,
        "scene_ids": ch,
        "range": {
            "start": first_s["start_frame"],
            "end": last_s["end_frame"],
        },
        "title": main_topic,
        "location_types": sorted(locs),
    })

print(f"  {len(chapters)} chapters from {len(scenes)} scenes")
for ch in chapter_list:
    ids = ch["scene_ids"]
    print(f"    CH{ch['id']:02d} f{ch['range']['start']}-{ch['range']['end']}  {ch['title']}")

# 填入骨架
skeleton["chapters"] = chapter_list

skel_out = os.path.join(out_dir, "skeleton.json")
with open(skel_out, "w") as f:
    json.dump(skeleton, f, ensure_ascii=False, indent=2)

ch_out = os.path.join(out_dir, "chapter_output.json")
with open(ch_out, "w") as f:
    json.dump({
        "step": "chapter_old",
        "n_scenes": len(scenes),
        "n_chapters": len(chapters),
        "chapters": chapter_list,
    }, f, ensure_ascii=False, indent=2)

print(f"  -> {out_dir}/")
