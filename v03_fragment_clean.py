#!/usr/bin/env python3
"""v03_fragment_clean — Step 3: 连续短碎片（<3s）合并为一组，直到遇到 ≥3s 或人物场景。"""
import json, os, sys

SHORT_S = 2.0

def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <output_dir> [input_skel]"); sys.exit(1)
    output = sys.argv[1]
    in_path = sys.argv[2] if len(sys.argv) > 2 else f"{output}/v02_person_label/skeleton.json"

    with open(in_path) as f: sk = json.load(f)
    shots = sk["shots"]; fps = sk["fps"]
    scenes = sk["proto_scenes"]

    # 从左到右扫描，连续短碎片合并
    # 连续 <SHORT_S 的片段组合并，≥SHORT_S 或有人物的保持独立
    result = []; buf = []
    for sc in scenes:
        is_short = sc["duration_s"] < SHORT_S
        if sc["persons"] and not is_short:
            if buf: result.append(_merge_buf(buf, shots, fps)); buf = []
            result.append(sc)
        elif is_short:
            buf.append(sc)
        else:
            if buf: result.append(_merge_buf(buf, shots, fps)); buf = []
            result.append(sc)
    if buf:
        result.append(_merge_buf(buf, shots, fps))

    # 格式化输出
    proto = []
    for ci, sc in enumerate(result):
        sids = sc["shot_ids"]
        sf = shots[sids[0]]["range"]["start"]; ef = shots[sids[-1]]["range"]["end"]
        proto.append({"id": ci, "shot_ids": sids, "n_shots": len(sids),
                      "range": {"start": sf, "end": ef},
                      "duration_s": round((ef-sf+1)/fps, 1),
                      "persons": sc.get("persons", [])})
    sk["proto_scenes"] = proto

    out_dir = f"{output}/v03_fragment_clean"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/skeleton.json", "w") as f: json.dump(sk, f, ensure_ascii=False, indent=2)

    p_count = sum(1 for s in proto if s["persons"])
    f_count = sum(1 for s in proto if not s["persons"])
    print(f"  {len(scenes)} -> {len(proto)} scenes ({p_count} person, {f_count} fragment)")
    print(f"  -> {out_dir}/")


def _merge_buf(buf, shots, fps):
    """合并缓冲区所有短碎片为一个。"""
    all_sids = []
    for sc in buf:
        all_sids.extend(sc["shot_ids"])
    dur = (shots[all_sids[-1]]["range"]["end"] - shots[all_sids[0]]["range"]["start"] + 1) / fps
    return {"persons": [], "shot_ids": all_sids, "n_shots": len(all_sids),
            "duration_s": round(dur, 1)}


if __name__ == "__main__":
    main()
