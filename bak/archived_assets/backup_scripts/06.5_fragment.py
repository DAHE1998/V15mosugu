#!/usr/bin/env python3
"""06.5 — 碎片补偿: 短 scene 合并处理 (快切合并 + 语义选侧)。

输入:  06_merge/skeleton.json (shots[] + scenes[])
输出:  06.5_fragment/skeleton.json  (+ scenes[] 经碎片补偿)
       06.5_fragment/fragment_output.json
"""
import json, sys, os, statistics, re

output = sys.argv[1]
in_path = os.path.join(output, "06_merge", "skeleton.json")
with open(in_path) as f:
    skeleton = json.load(f)

out_dir = os.path.join(output, "06.5_fragment")
os.makedirs(out_dir, exist_ok=True)

scenes = skeleton.get("scenes", [])
fps = skeleton.get("fps", 30)
TEXT_THR = float(os.environ.get("TEXT_THR", "0.40"))
VIS_THR = float(os.environ.get("VIS_THR", "0.50"))
FRAG_RATIO = float(os.environ.get("FRAG_RATIO", "0.35"))
MIN_OK = int(3.0 * fps)
print(f"[06.5] {len(scenes)} scenes  ratio={FRAG_RATIO}  min_ok={MIN_OK}fr")


def dur(s):
    return s["range"]["end"] - s["range"]["start"] + 1


def merge_into(L, R):
    at = (L.get("asr_text", "") or "").rstrip("。")
    bt = R.get("asr_text", "") or ""
    if at and bt:
        L["asr_text"] = at + "。" + bt
    elif bt:
        L["asr_text"] = bt
    L["range"]["end"] = R["range"]["end"]
    L["shot_ids"].extend(R.get("shot_ids", []))


def merge_right(L, R):
    at = L.get("asr_text", "") or ""
    bt = (R.get("asr_text", "") or "").rstrip("。")
    if at and bt:
        R["asr_text"] = at + "。" + bt
    elif at:
        R["asr_text"] = at
    R["range"]["start"] = L["range"]["start"]
    R["shot_ids"] = L.get("shot_ids", []) + R.get("shot_ids", [])


stops = {
    "的", "了", "是", "在", "有", "就", "这", "那", "也", "还", "都",
    "我", "你", "他", "她", "它", "们", "个", "什么", "这个", "那个",
    "一个", "没有", "不是", "因为", "所以", "但是", "和", "与",
    "我们", "他们", "她们", "它们", "那里", "这里", "什么",
}


def sem(ml, li, ri):
    lt = (ml[li].get("asr_text", "") or "").strip()
    rt = (ml[ri].get("asr_text", "") or "").strip()
    if not lt or not rt:
        return 0.0
    lw = set(re.findall(r"[一-鿿\w]{2,}", lt)) - stops
    rw = set(re.findall(r"[一-鿿\w]{2,}", rt)) - stops
    if not lw or not rw:
        return 0.0
    return len(lw & rw) / min(len(lw), len(rw))


# ── 迭代碎片合并 ──
MAX_ITER = 100
merged = list(scenes)
for _iter in range(MAX_ITER):
    n = len(merged)
    if n < 2:
        break
    ds = [dur(s) for s in merged]
    md = statistics.median(ds)
    ft = md * FRAG_RATIO
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
                ls = sem(merged, idx - 1, idx)
                rs = sem(merged, idx, idx + 1)
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
                mx = max(mx, sem(merged, cs - 1, cs))
            for k in range(cs, ce):
                mx = max(mx, sem(merged, k, k + 1))
            if hr:
                mx = max(mx, sem(merged, ce, ce + 1))

            if mx < 1.0:
                td = sum(dur(merged[k]) for k in range(cs, ce + 1))
                if td >= MIN_OK:
                    cur = dict(merged[cs])
                    for m in range(cs + 1, ce + 1):
                        nt = merged[m]
                        mt = cur.get("asr_text", "") or ""
                        ntt = nt.get("asr_text", "") or ""
                        if mt and ntt:
                            cur["asr_text"] = mt.rstrip("。") + "。" + ntt
                        elif ntt:
                            cur["asr_text"] = ntt
                        cur["range"]["end"] = nt["range"]["end"]
                        cur["shot_ids"].extend(nt.get("shot_ids", []))
                    for _ in range(cs, ce):
                        merged.pop(cs + 1)
                    merged[cs] = cur
                    ls = sem(merged, cs - 1, cs) if cs > 0 else 0
                    rs = sem(merged, cs, cs + 1) if cs < len(merged) - 1 else 0
                    if ls >= rs and cs > 0:
                        merge_into(merged[cs - 1], merged[cs])
                        merged.pop(cs)
                    elif rs > ls and cs < len(merged) - 1:
                        merge_right(merged[cs], merged[cs + 1])
                        merged.pop(cs)
                else:
                    ls = sem(merged, cs - 1, cs) if hl else 0
                    rs = sem(merged, ce, ce + 1) if hr else 0
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
                if hl and sem(merged, cs - 1, cs) >= 1.0:
                    merge_into(merged[cs - 1], merged[cs])
                    merged.pop(cs)
                    changed = True
                    break
                if hr and sem(merged, ce, ce + 1) >= 1.0:
                    merge_right(merged[ce], merged[ce + 1])
                    merged.pop(ce)
                    changed = True
                    break
                for k in range(cs, ce):
                    if sem(merged, k, k + 1) >= 1.0:
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

skeleton["scenes"] = merged
skel_out = os.path.join(out_dir, "skeleton.json")
with open(skel_out, "w") as f:
    json.dump(skeleton, f, ensure_ascii=False, indent=2)

frag_out = os.path.join(out_dir, "fragment_output.json")
with open(frag_out, "w") as f:
    json.dump({"step": "06.5_fragment", "n_before": len(scenes), "n_after": len(merged)},
              f, ensure_ascii=False, indent=2)

print(f"  -> {out_dir}/")
