#!/usr/bin/env python3
"""v01_visual_group — Step 1: DINO 视觉相邻分组。"""
import json, os, sys
import numpy as np

VIS_THR = 0.45

def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <output_dir> [vis_threshold]"); sys.exit(1)
    output = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else VIS_THR

    # 优先读 D50
    sk_path = f"{output}/d50_dedup/skeleton.json"
    if not os.path.isfile(sk_path):
        sk_path = f"{output}/dino_cluster/skeleton.json"
    graph = np.load(f"{output}/dino_cluster/shot_visual_graph.npy")
    with open(sk_path) as f: sk = json.load(f)
    shots = sk["shots"]; n = len(shots); fps = sk["fps"]

    # 读取人物数据用于判断人物变化边界
    shot_persons = {}
    fc_path = f"{output}/face_continuity/person_chains.json"
    if os.path.isfile(fc_path):
        with open(fc_path) as f:
            for c in json.load(f)["chains"]:
                if c["n_shots"] >= 2:
                    for sid in c["shots"]:
                        shot_persons.setdefault(sid, []).append(c["person_id"])

    def same_scene(g, nxt):
        """g尾和nxt头人物构成相同 → 同一场景；人物进出 → 新场景"""
        gp = set(shot_persons.get(g[-1], []))
        np = set(shot_persons.get(nxt[0], []))
        if not gp or not np: return True   # 任一边没脸 → DINO决定
        return gp == np                     # 人物构成完全相同才合

    groups = [[i] for i in range(n)]
    while True:
        changed = False; new = []; i = 0
        while i < len(groups):
            g = groups[i]
            if i >= len(groups)-1: new.append(g); break
            nxt = groups[i+1]
            dino_ok = float(graph[g[-1]][nxt[0]]) >= thr
            person_ok = same_scene(g, nxt)
            if dino_ok and person_ok:
                new.append(g + nxt); i += 2; changed = True
            else:
                new.append(g); i += 1
        groups = new
        if not changed: break

    proto = []
    for ci, g in enumerate(groups):
        sf = shots[g[0]]["range"]["start"]; ef = shots[g[-1]]["range"]["end"]
        proto.append({"id": ci, "shot_ids": g, "n_shots": len(g),
                      "range": {"start": sf, "end": ef},
                      "duration_s": round((ef-sf+1)/fps, 1)})
    sk["proto_scenes"] = proto

    out_dir = f"{output}/v01_visual_group"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/skeleton.json", "w") as f: json.dump(sk, f, ensure_ascii=False, indent=2)

    sizes = [p["n_shots"] for p in proto]
    print(f"  DINO cos≥{thr}: {n} shots -> {len(proto)} groups  max={max(sizes)} avg={np.mean(sizes):.1f}")
    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    main()
