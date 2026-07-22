#!/usr/bin/env python3
"""
asr_scene — 骨架驱动 ASR (SenseVoiceSmall)。

设计原则：
  - 骨架文件（skeleton.json）是唯一时间轴真相源
  - 不返回时间戳，不做 VAD
  - 每个 scene 裁音频 → ASR → 文本写回 skeleton

输入:  <output_dir> [video_path]
       从 output_dir/short_merge/skeleton.json 读取 proto_scenes
输出:  output_dir/asr/skeleton.json
       （原 skeleton 扩展 scenes[].asr = {language, text}）
"""

import json, os, sys, io, time
import subprocess as sp
import numpy as np
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

import torch
assert torch.cuda.is_available(), "CUDA 不可用"

# ── config ──
ASR_MODEL = "/home/dahe/.cache/modelscope/models/iic/SenseVoiceSmall"
MIN_DURATION_S = 0.5


def extract_full_audio(video):
    """提取整轨 WAV 到内存 bytes 返回"""
    print(f"  extracting full audio...")
    t0 = time.time()
    proc = sp.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda", "-i", video,
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav", "-"
    ], capture_output=True, check=True)
    print(f"    done ({time.time()-t0:.0f}s)")
    return proc.stdout


def slice_audio(wav_bytes, start_s, end_s, sr=16000):
    """从 WAV bytes 中切出 [start_s, end_s) 段的 numpy array"""
    data, orig_sr = sf.read(io.BytesIO(wav_bytes))
    assert orig_sr == sr
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    end_sample = min(end_sample, len(data))
    return data[start_sample:end_sample]


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <output_dir> [video_path]")
        sys.exit(1)

    output = sys.argv[1]
    video = sys.argv[2] if len(sys.argv) > 2 else None

    # ── 读取骨架 ──
    sk_path = os.path.join(output, "short_merge", "skeleton.json")
    with open(sk_path) as f:
        skeleton = json.load(f)

    scenes = skeleton["proto_scenes"]
    fps = skeleton["fps"]
    n = len(scenes)
    print(f"[asr_scene] {n} scenes @ {fps}fps")

    if video is None:
        video = skeleton.get("video", "")
        if not video:
            print("ERROR: 未提供 video_path，skeleton 中也无 video 字段")
            sys.exit(1)
    print(f"  video: {video}")

    # ── 整轨提取（一次） ──
    full_wav = extract_full_audio(video)

    # ── 加载模型 ──
    t0 = time.time()
    model = AutoModel(
        model=ASR_MODEL,
        # 不用 VAD — scene 边界就是分段时间
        vad_model=None,
        device="cuda",
    )
    print(f"  model loaded ({time.time()-t0:.0f}s)")

    # ── 逐 scene ASR ──
    t_asr = 0.0
    ok = 0
    skip = 0

    for si, sc in enumerate(scenes):
        sf_frame = sc["range"]["start"]
        ef_frame = sc["range"]["end"]
        dur_s = (ef_frame - sf_frame + 1) / fps

        if dur_s < MIN_DURATION_S:
            sc["asr"] = {"language": "skip", "text": ""}
            skip += 1
            continue

        t_start = sf_frame / fps
        t_end = (ef_frame + 1) / fps

        # 切片
        audio = slice_audio(full_wav, t_start, t_end)

        # ASR
        t1 = time.time()
        res = model.generate(input=audio, language="auto")  # auto 检测语种
        elapsed = time.time() - t1
        t_asr += elapsed

        # 取文本
        text = ""
        language = "unk"
        if res and len(res) > 0:
            raw = res[0].get("text", "")
            # SenseVoice 输出格式: "<|zh|><|NEUTRAL|><|Speech|>..."
            # 去掉标签取纯文本
            clean = rich_transcription_postprocess(raw)
            # 检测语种
            if "<|zh|>" in raw:
                language = "zh"
            elif "<|ja|>" in raw:
                language = "ja"
            elif "<|en|>" in raw:
                language = "en"
            text = clean.strip() if clean else raw.strip()

        sc["asr"] = {"language": language, "text": text}
        ok += 1

        # 预览
        preview = text[:60].replace("\n", " ")
        print(f"  [{si:3d}] {dur_s:6.1f}s {language:4s} {elapsed:4.1f}s | {preview}")

    # ── 输出 ──
    out_dir = os.path.join(output, "asr")
    os.makedirs(out_dir, exist_ok=True)

    skeleton["_asr"] = {
        "model": "SenseVoiceSmall",
        "mode": "scene-aware, no VAD",
        "n_scenes": n,
        "n_asr": ok,
        "n_skip": skip,
        "asr_time_s": round(t_asr, 1),
    }

    out_path = os.path.join(out_dir, "skeleton.json")
    with open(out_path, "w") as f:
        json.dump(skeleton, f, ensure_ascii=False, indent=2)

    # ── 可读文本 ──
    txt_path = os.path.join(out_dir, "asr_text.txt")
    with open(txt_path, "w") as f:
        for sc in scenes:
            sid = sc["id"]
            sf = sc["range"]["start"]
            ef = sc["range"]["end"]
            dur = sc["duration_s"]
            asr = sc.get("asr", {})
            lang = asr.get("language", "?")
            text = asr.get("text", "")
            f.write(f"[{sid:3d}] {sf:6d}-{ef:6d}fr ({dur:6.1f}s) {lang:4s} | {text}\n")

    print(f"\n  done -> {out_dir}/")
    print(f"  {ok}/{n} scenes ASR'd ({skip} skipped), {t_asr:.1f}s")


if __name__ == "__main__":
    main()
