#!/usr/bin/env python3
"""测试 VLM 加入完整 ASR 文本后的效果"""
import json, sys, os, re, subprocess as sp, time
import numpy as np
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
from qwen_vl_utils import process_vision_info

skeleton_path = sys.argv[1] if len(sys.argv) > 1 else "/home/dahe/VideoCenter/test_videos/street/japanese_street_girls/fragment/skeleton.json"

with open(skeleton_path) as f:
    skeleton = json.load(f)

scenes = skeleton["scenes"]
shots = skeleton["shots"]
video = skeleton["video"]
fps = skeleton["fps"]
shots_by_id = {s["id"]: s for s in shots}

# Pick sc_09 (招牌讲解) and sc_10 (指店讲解)
test_ids = [9, 10]
test_scenes = [s for s in scenes if s["id"] in test_ids]

for scene in test_scenes:
    kf = []
    asr_parts = []
    for sid in scene.get("shot_ids", []):
        s = shots_by_id.get(sid)
        if s:
            kf.extend(s.get("key_frames", []))
            t = (s.get("asr_text") or "").strip()
            if t:
                asr_parts.append(t)
    kf = sorted(set(int(f) for f in kf if f))
    asr_text = "。".join(asr_parts)
    dur_s = (scene["range"]["end"] - scene["range"]["start"] + 1) / fps
    print("sc_%d: f%d-f%d (%ds) ASR=%dchars kf=%s" % (scene["id"], scene["range"]["start"], scene["range"]["end"], dur_s, len(asr_text), kf[:3]))

# Extract frames
all_f = sorted(set(
    int(f) for s in test_scenes
    for sid in s.get("shot_ids", [])
    for f in (shots_by_id.get(sid, {}).get("key_frames", []) if shots_by_id.get(sid) else [])
    if f
))
frame_map = {}
BATCH = 50
for bs in range(0, len(all_f), BATCH):
    bf = all_f[bs:bs + BATCH]
    sel = "+".join(["eq(n\\,%d)" % f for f in bf])
    proc = sp.Popen([
        "ffmpeg", "-hwaccel", "cuda", "-loglevel", "error", "-i", video,
        "-vf", "select=" + sel + ",scale=448:448",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ], stdout=sp.PIPE, stderr=sp.PIPE, bufsize=128*1024*1024)
    n = len(bf)
    raw = proc.stdout.read(n * 448 * 448 * 3)
    proc.wait()
    if len(raw) >= n * 448 * 448 * 3:
        arr = np.frombuffer(raw[:n*448*448*3], dtype=np.uint8).reshape(n, 448, 448, 3)
        for i, fn in enumerate(bf):
            frame_map[fn] = Image.fromarray(arr[i])
print("Extracted %d frames" % len(frame_map))

# Load VLM
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                          bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/home/dahe/models/hf/hub/Qwen/Qwen3-VL-8B-Instruct",
    quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(
    "/home/dahe/models/hf/hub/Qwen/Qwen3-VL-8B-Instruct")
processor.tokenizer.padding_side = "left"

for scene in test_scenes:
    kf = sorted(set(int(f) for f in (scene.get("key_frames") or []) if f))[:6]
    asr_parts = []
    for sid in scene.get("shot_ids", []):
        s = shots_by_id.get(sid)
        if s:
            t = (s.get("asr_text") or "").strip()
            if t:
                asr_parts.append(t)
    asr_text = "。".join(asr_parts)
    dur_s = (scene["range"]["end"] - scene["range"]["start"] + 1) / fps

    context = "场景时长: %d秒\n对话内容: %s\n\n根据以上画面和对话内容，描述这个场景在发生什么" % (dur_s, asr_text)

    content = [{"type": "text", "text": context}]
    for fn in kf:
        img = frame_map.get(fn)
        if img:
            content.append({"type": "image", "image": img})

    msgs = [
        {"role": "system", "content": [{"type": "text", "text": "你是一个视频场景分析师。根据画面和对话内容，用一段话准确描述场景中发生的事件。"}]},
        {"role": "user", "content": content},
    ]

    imgs, _ = process_vision_info(msgs, image_patch_size=processor.image_processor.patch_size)
    if imgs is None: imgs = []

    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[imgs], padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    in_len = inputs["input_ids"].shape[1]
    resp = processor.decode(out_ids[0][in_len:], skip_special_tokens=True).strip()

    print("\n=== sc_%d ===" % scene["id"])
    print("ASR: %s" % asr_text[:200])
    print("VLM: %s" % resp[:300])
