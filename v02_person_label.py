#!/usr/bin/env python3
"""v02_person_label — 给视觉组标人物。局部频率×全局特异性选主导人物。"""
import json, os, sys
from collections import Counter

def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <output_dir> [input_skel]"); sys.exit(1)
    output = sys.argv[1]
    in_path = sys.argv[2] if len(sys.argv) > 2 else f"{output}/v01_visual_group/skeleton.json"

    with open(in_path) as f: sk = json.load(f)
    with open(f"{output}/face_continuity/person_chains.json") as f: fc = json.load(f)

    shots = sk["shots"]; fps = sk["fps"]
    groups = sk["proto_scenes"]

    shot_persons = {}
    face_total = {}
    for c in fc["chains"]:
        if c["n_shots"] >= 2:
            face_total[c["person_id"]] = c["n_shots"]
            for sid in c["shots"]:
                shot_persons.setdefault(sid, []).append(c["person_id"])

    # 主导 = 局部占比×全局特异性 最高
    labeled = []
    for sc in groups:
        cnt = Counter()
        for sid in sc["shot_ids"]:
            for pid in shot_persons.get(sid, []):
                cnt[pid] += 1
        dom = None
        if len(cnt) >= 2:  # 至少两种人物同框才算人物场景
            best = -1; n = max(len(sc["shot_ids"]), 1)
            for pid, c in cnt.items():
                s = (c / n) * 0.7 + (1.0 / max(face_total.get(pid, 999), 1)) * 0.3
                if s > best: best = s; dom = pid
        labeled.append((sc, dom))

    # 相邻同人物合并
    merged = []
    for sc, dom in labeled:
        if merged and dom is not None and merged[-1][1] == dom:
            m = merged[-1][0]
            m["shot_ids"].extend(sc["shot_ids"]); m["n_shots"] += sc["n_shots"]
        else:
            merged.append((dict(sc), dom))

    # 修剪首尾
    trimmed = []
    for sc, dom in merged:
        if dom is None:
            trimmed.append((sc, None)); continue
        sids = sc["shot_ids"]
        start = 0
        while start < len(sids) and dom not in shot_persons.get(sids[start], []):
            start += 1
        end = len(sids) - 1
        while end >= start and dom not in shot_persons.get(sids[end], []):
            end -= 1
        if start > end or end - start + 1 < 1:
            trimmed.append((sc, None)); continue
        sc["shot_ids"] = sids[start:end+1]; sc["n_shots"] = len(sc["shot_ids"])
        if start > 0:
            trimmed.append(({"shot_ids": sids[:start], "n_shots": start}, None))
        trimmed.append((sc, dom))
        if end < len(sids) - 1:
            trimmed.append(({"shot_ids": sids[end+1:], "n_shots": len(sids)-end-1}, None))
    merged = trimmed

    # 输出
    proto = []
    for ci, (sc, dom) in enumerate(merged):
        sids = sc["shot_ids"]
        sf = shots[sids[0]]["range"]["start"]; ef = shots[sids[-1]]["range"]["end"]
        proto.append({"id": ci, "shot_ids": sids, "n_shots": len(sids),
                      "range": {"start": sf, "end": ef},
                      "duration_s": round((ef-sf+1)/fps, 1),
                      "persons": [dom] if dom is not None else []})
    sk["proto_scenes"] = proto

    out_dir = f"{output}/v02_person_label"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/skeleton.json", "w") as f: json.dump(sk, f, ensure_ascii=False, indent=2)

    print(f"  {len(groups)} groups -> {len(proto)} scenes ({sum(1 for s in proto if s['persons'])} person-labeled)")
    for sc in proto:
        pstr = ",".join(f"P{p}" for p in sc["persons"]) if sc["persons"] else "-"
        print(f"  {pstr:10s} S{sc['id']:3d}: {sc['shot_ids'][0]:3d}-{sc['shot_ids'][-1]:3d} {sc['n_shots']:2d}s {sc['duration_s']:5.1f}s")


if __name__ == "__main__":
    main()
