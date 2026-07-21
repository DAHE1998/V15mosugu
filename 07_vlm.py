#!/usr/bin/env python3
"""07 — VLM 场景描述: Qwen3-VL 批处理场景帧 → profile 填入骨架。

直接消费 02 的 key_frames，不重新抽帧。

输入:  06.5_fragment/skeleton.json (shots[] + scenes[])
输出:  07_vlm/skeleton.json  (+ scene.profile)
       07_vlm/vlm_output.json
"""
import os, sys, json, time, subprocess as sp, re
import numpy as np
from PIL import Image

output = sys.argv[1]
in_path = os.path.join(output, "06.5_fragment", "skeleton.json")
with open(in_path) as f:
    skeleton = json.load(f)

out_dir = os.path.join(output, "07_vlm")
os.makedirs(out_dir, exist_ok=True)

video = skeleton["video"]
scenes = skeleton.get("scenes", [])
shots = skeleton.get("shots", [])
fps = skeleton["fps"]

PROMPT = """分析这段视频画面，输出以下JSON字段（严格JSON，不要多余文字）：
{
  "environment": "场景环境描述（如：车内、街道、咖啡店、室内外等）",
  "characters": "人物描述（如：戴墨镜的男子、一群路人、无人物等）",
  "actions": "正在发生的动作（如：驾驶、对话、行走、抢劫、逃跑等）",
  "objects": "关键物体（如：汽车、枪、咖啡杯、手机等）",
  "event": "事件描述（如：追车、抢劫、日常对话、用餐等）",
  "story_role": "叙事功能（如：开场、动作场面、对话推进、过渡、高潮、结尾等）",
  "topic": "这段视频的主题（1-5字，如：追车、买咖啡、分赃）",
  "visual_summary": "一句话总结画面内容（10字以内）"
}"""

from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

BATCH = 4
MAX_FRAMES_PER_SCENE = 12


def extract_all_frames(video_path, all_frame_nums, scale=448):
    """一次性提取所有 key_frames，返回 {frame_num: Image} 映射。
    分批提取，每批最多 50 帧（ffmpeg select 表达式长度限制）。"""
    if not all_frame_nums:
        return {}
    all_f = sorted(set(all_frame_nums))
    BATCH = 50
    frame_map = {}
    for batch_start in range(0, len(all_f), BATCH):
        batch_f = all_f[batch_start:batch_start + BATCH]
        sel = "+".join(["eq(n\\,%d)" % int(fn) for fn in batch_f])
        proc = sp.Popen([
            "ffmpeg", "-hwaccel", "cuda", "-loglevel", "error", "-i", video_path,
            "-vf", "select=" + sel + ",scale=%d:%d" % (scale, scale),
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ], stdout=sp.PIPE, stderr=sp.PIPE, bufsize=128 * 1024 * 1024)
        n = len(batch_f)
        raw = proc.stdout.read(n * scale * scale * 3)
        proc.wait()
        if len(raw) < n * scale * scale * 3:
            print(f"    WARNING: batch expected {n * scale * scale * 3} bytes, got {len(raw)}")
            continue
        arr = np.frombuffer(raw[:n * scale * scale * 3], dtype=np.uint8).reshape(n, scale, scale, 3)
        for i, fn in enumerate(batch_f):
            frame_map[fn] = Image.fromarray(arr[i])
    return frame_map


print(f"[07] VLM: {len(scenes)} scenes")
t0 = time.time()

# ── 收集每个 scene 的 key_frames ──
shots_by_id = {s["id"]: s for s in shots}
scene_jobs = []
all_needed_frames = []

for scene in scenes:
    kf_set = []
    for sid in scene.get("shot_ids", []):
        s = shots_by_id.get(sid)
        if s:
            kf_set.extend(s.get("key_frames", []))
    kf_set = sorted(set(int(f) for f in kf_set if f is not None))

    # 限制每 scene 帧数
    if len(kf_set) > MAX_FRAMES_PER_SCENE:
        step = len(kf_set) / MAX_FRAMES_PER_SCENE
        indices = [int(i * step) for i in range(MAX_FRAMES_PER_SCENE)]
        kf_set = [kf_set[i] for i in indices]

    scene_jobs.append((scene, kf_set))
    all_needed_frames.extend(kf_set)

print(f"  {sum(len(s[1]) for s in scene_jobs)} total key_frames across {len(scenes)} scenes")

# ── 一次性提取所有帧 ──
t1 = time.time()
print("  Extracting all key_frames (single ffmpeg call)...")
frame_map = extract_all_frames(video, all_needed_frames)
print(f"    extracted {len(frame_map)} frames ({time.time()-t1:.0f}s)")

# ── 加载 VLM ──
print("  Loading Qwen3-VL-8B (4-bit)...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                          bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/home/dahe/models/hf/hub/Qwen/Qwen3-VL-8B-Instruct",
    quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    "/home/dahe/models/hf/hub/Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)
processor.tokenizer.padding_side = "left"
print(f"    loaded ({time.time()-t0:.0f}s)")

# ── Batch VLM 推理 ──
print(f"  Batched VLM (batch={BATCH})...")
t2 = time.time()

# 先处理所有 scene 的帧，记录上一个 scene 的 profile 供下一个使用
prev_profile = ""

for batch_start in range(0, len(scene_jobs), BATCH):
    batch = scene_jobs[batch_start:batch_start + BATCH]
    batch_texts = []
    all_images_list = []

    for scene, kf_set in batch:
        dur_s = (scene["range"]["end"] - scene["range"]["start"] + 1) / fps

        # ASR 文本（优先用 contextual ASR，它有 ±15秒上下文）
        asr_parts = []
        for sid in scene.get("shot_ids", []):
            s = shots_by_id.get(sid)
            if s:
                t = (s.get("asr_contextual") or s.get("asr_text") or "").strip()
                if t:
                    asr_parts.append(t)
        asr_text = "。".join(asr_parts) if asr_parts else ""

        # 构建带上下文的 prompt
        context = f"场景时长: {dur_s:.0f}秒"
        if asr_text:
            context += f"\n对话: {asr_text}"
        if prev_profile:
            context += f"\n前一个场景: {prev_profile}"

        content = [{"type": "text", "text": context + "\n\n分析这段画面："}]
        for fn in kf_set:
            img = frame_map.get(fn)
            if img:
                content.append({"type": "image", "image": img})
        if not any(c["type"] == "image" for c in content):
            content = [{"type": "text", "text": context + "\n\n无画面内容"}]

        msgs = [
            {"role": "system", "content": [{"type": "text", "text": PROMPT}]},
            {"role": "user", "content": content},
        ]
        imgs, vkw = process_vision_info(
            msgs,
            image_patch_size=processor.image_processor.patch_size,
        )
        if imgs is None:
            imgs = []
        all_images_list.append(imgs)
        batch_texts.append(
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )

    img_counts = [len(imgs) for imgs in all_images_list]

    batch_decoded = []
    if all(c == 0 for c in img_counts):
        for scene, kf_set in batch:
            scene["key_frames"] = [int(f) for f in kf_set]
            scene["profile"] = {"environment": "", "characters": "", "actions": "",
                                "objects": "", "event": "", "story_role": "",
                                "topic": "", "visual_summary": "无画面"}
    else:
        try:
            inputs = processor(text=batch_texts, images=all_images_list,
                               padding=True, return_tensors="pt")
            inputs = inputs.to(model.device)
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            in_len = inputs["input_ids"].shape[1]
            raw_decoded = [out_ids[i][in_len:] for i in range(len(batch))]
            batch_decoded = processor.batch_decode(raw_decoded, skip_special_tokens=True)
        except Exception as e:
            import traceback
            print(f"    Batch error: {type(e).__name__}: {e}")
            traceback.print_exc()
            batch_decoded = []

    for i, (scene, kf_set) in enumerate(batch):
        resp = batch_decoded[i].strip() if i < len(batch_decoded) else ""
        m = re.search(r"{.*}", resp, re.DOTALL)
        profile = json.loads(m.group()) if m else {"_raw": resp[:100]}
        scene["key_frames"] = [int(f) for f in kf_set]
        scene["profile"] = profile

        # 更新 prev_profile 供下一个 scene 作为上下文
        if profile.get("visual_summary"):
            prev_profile = profile["visual_summary"]

    n_done = min(batch_start + BATCH, len(scene_jobs))
    print(f"    [{n_done}/{len(scene_jobs)}] {time.time()-t2:.0f}s")

# ── 输出 ──
skeleton["scenes"] = scenes
skel_out = os.path.join(out_dir, "skeleton.json")
with open(skel_out, "w") as f:
    json.dump(skeleton, f, ensure_ascii=False, indent=2)

vlm_out = os.path.join(out_dir, "vlm_output.json")
vlm_results = [{
    "id": s["id"],
    "range": s["range"],
    "key_frames": s.get("key_frames", []),
    "profile": s.get("profile", {}),
} for s in scenes]
with open(vlm_out, "w") as f:
    json.dump({"step": "07_vlm", "n_scenes": len(scenes), "results": vlm_results},
              f, ensure_ascii=False, indent=2)

n_vlm = sum(1 for s in scenes if "profile" in s)
print(f"\n[07] Done: {n_vlm}/{len(scenes)} scenes profiled ({time.time()-t0:.0f}s)")
print(f"  -> {out_dir}/")
