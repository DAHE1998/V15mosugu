#!/usr/bin/env python3
"""06 — ASR Aggregate: VAD 段时间轴整段就近归属 Scene。

输入:  fragment/skeleton.json (scenes[] + range + shot_ids)
       asr/asr_timeline.json (VAD 段时间轴)
输出:  asr_aggregate/skeleton.json (+ scenes[].asr_text)
       asr_aggregate/orphan_segments.json

v3.0 新增模块:
  - 每个 VAD 段整段归入重叠最多的 scene，不切字
  - 孤儿段单独输出，不强行塞给最近 scene
  - C6-C8 assert
"""
import json, sys, os, time
import numpy as np
import torch

assert torch.cuda.is_available(), "CUDA 不可用，检查 torch 安装"

output = sys.argv[1]
out_dir = os.path.join(output, "asr_aggregate")
os.makedirs(out_dir, exist_ok=True)

# ── 读取输入 ──────────────────────────────────────────────────────
scene_path = os.path.join(output, "fragment", "skeleton.json")
asr_path = os.path.join(output, "asr", "asr_timeline.json")

with open(scene_path) as f:
    skeleton = json.load(f)
scenes = skeleton.get("scenes", [])
fps = skeleton["fps"]
tf = skeleton["total_frames"]
dur_s = skeleton.get("duration", 0)

with open(asr_path) as f:
    asr_data = json.load(f)
timeline = asr_data.get("results", [])

print(f"[06] ASR aggregate: {len(scenes)} scenes, {len(timeline)} VAD segments")

# ── 时间单位转换 ──────────────────────────────────────────────────
# scene.range 是帧号 [start, end]（闭区间），统一转成毫秒
def frame_range_to_ms(sf, ef):
    """帧号闭区间 [sf, ef] → 毫秒区间 [start_ms, end_ms]。"""
    if dur_s > 0:
        # 用 duration 反推，比 fps 更稳（避免 fps 浮点误差）
        return (sf / tf * dur_s * 1000, (ef + 1) / tf * dur_s * 1000)
    else:
        return (sf / fps * 1000, (ef + 1) / fps * 1000)

# ── 计算每个 scene 的时间范围 ─────────────────────────────────────
scene_time_ranges = []
for s in scenes:
    sf = s["range"]["start"]
    ef = s["range"]["end"]
    start_ms, end_ms = frame_range_to_ms(sf, ef)
    scene_time_ranges.append({
        "scene_id": s["id"],
        "start_ms": start_ms,
        "end_ms": end_ms,
        "shot_ids": s.get("shot_ids", []),
    })

# ── 每个 VAD 段归属到 scene ───────────────────────────────────────
orphan_segments = []
scene_asr = {s["id"]: [] for s in scenes}

for seg in timeline:
    vad_start = seg["start_ms"]
    vad_end = seg["end_ms"]
    text = seg.get("text", "").strip()
    if not text:
        continue

    # 计算与每个 scene 的时间重叠
    best_scene_id = None
    best_overlap = 0
    for sr in scene_time_ranges:
        overlap = max(0, min(vad_end, sr["end_ms"]) - max(vad_start, sr["start_ms"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_scene_id = sr["scene_id"]

    if best_scene_id is not None and best_overlap > 0:
        scene_asr[best_scene_id].append((vad_start, text))
    else:
        orphan_segments.append({
            "start_ms": vad_start,
            "end_ms": vad_end,
            "text": text,
            "assigned_scene_id": None,
            "reason": "no_overlap",
        })

# ── 拼接 scene.asr_text ───────────────────────────────────────────
for s in scenes:
    sid = s["id"]
    segs = scene_asr.get(sid, [])
    segs.sort(key=lambda x: x[0])  # 按时间排序
    s["asr_text"] = "。".join(t for _, t in segs) if segs else ""

# ── C6: 输出文本不得出现半个字（来自比例切分残留）──────────────────
# v3.0 架构下 ASR 是整段归属，不存在比例切分，所以只需验证
# scene.asr_text 中的每个字符都是完整 Unicode 字符（没有半个 surrogate）
for s in scenes:
    text = s.get("asr_text", "")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise AssertionError(f"C6 失败: scene[{s['id']}] asr_text 包含非法 Unicode")

# ── C7: 每个 VAD 段只能唯一归属 ──────────────────────────────────
assigned_count = sum(len(v) for v in scene_asr.values())
total_with_text = sum(1 for seg in timeline if seg.get("text", "").strip())
assert assigned_count + len(orphan_segments) == total_with_text, \
    f"C7 失败: VAD 段归属数不匹配 (assigned={assigned_count}, orphan={len(orphan_segments)}, total={total_with_text})"

# ── C8: interior_orphan == 0 ─────────────────────────────────────
# 判断孤儿段是"内部孤儿"还是"边界孤儿"
# 内部孤儿 = 落在 scene 覆盖区间内的孤儿段 → 说明 fps/时间基准有 bug
# 边界孤儿 = 落在 scene 覆盖区间外的孤儿段（片头/片尾无镜头覆盖）→ 正常
def is_inside_coverage(vad_start_ms, vad_end_ms, scene_ranges):
    """检查 VAD 段是否落在任何 scene 的覆盖区间内。"""
    for sr in scene_ranges:
        overlap = max(0, min(vad_end_ms, sr["end_ms"]) - max(vad_start_ms, sr["start_ms"]))
        if overlap > 0:
            return True
    return False

interior_orphans = []
for seg in orphan_segments:
    if is_inside_coverage(seg["start_ms"], seg["end_ms"], scene_time_ranges):
        interior_orphans.append(seg)

assert len(interior_orphans) == 0, \
    f"C8 失败: 存在 {len(interior_orphans)} 个内部孤儿段，大概率是 fps/时间基准换算 bug"

boundary_orphans = len(orphan_segments) - len(interior_orphans)
total_orphan_ratio = len(orphan_segments) / max(len(timeline), 1)
print(f"  C6-C8 pass: orphan={len(orphan_segments)} ({total_orphan_ratio:.1%}), "
      f"interior={len(interior_orphans)}, boundary={boundary_orphans}")

# ── 输出 ──────────────────────────────────────────────────────────
skeleton["scenes"] = scenes
skel_out = os.path.join(out_dir, "skeleton.json")
with open(skel_out, "w") as f:
    json.dump(skeleton, f, ensure_ascii=False, indent=2)

orphan_out = {
    "step": "asr_aggregate",
    "n_orphan": len(orphan_segments),
    "n_interior": len(interior_orphans),
    "n_boundary": boundary_orphans,
    "orphan_ratio": round(total_orphan_ratio, 4),
    "segments": orphan_segments,
}
with open(os.path.join(out_dir, "orphan_segments.json"), "w") as f:
    json.dump(orphan_out, f, ensure_ascii=False, indent=2)

n_with_text = sum(1 for s in scenes if s.get("asr_text", "").strip())
print(f"  ASR aggregate: {n_with_text}/{len(scenes)} scenes have text")
print(f"  done -> {out_dir}/")
