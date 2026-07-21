#!/usr/bin/env python3
"""05 — ASR 转录: VAD 整轨分段 → SenseVoiceSmall → asr_timeline.json。

输入:  视频音轨（独立提取，不依赖 00-04.5 任何输出）
输出:  asr/asr_timeline.json = [{start_ms, end_ms, text, lang}]

v3.0 改动:
  - 删除按时间比例切分给 shot 的逻辑
  - 输出改为 VAD 段时间轴，不映射到 shot/scene
  - 可与 00-04.5 并行执行
"""
import json
import torch, sys, os, time, subprocess as sp
import numpy as np, soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

assert torch.cuda.is_available(), "CUDA 不可用，检查 torch 安装"

output = sys.argv[1]
video_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(output, os.path.basename(output) + ".mp4")
out_dir = os.path.join(output, "asr")
os.makedirs(out_dir, exist_ok=True)

SR = 16000
VAD_MODEL = "fsmn-vad"
ASR_MODEL = "/home/dahe/.cache/modelscope/models/iic/SenseVoiceSmall"
MERGE_GAP_MS = 500        # 合并间隔 <500ms 的相邻 VAD 段
MIN_SEG_DURATION = 0.3    # 最短有效语音段 (秒)

print(f"[05] ASR timeline: {output}")

# ── 工具函数 ──────────────────────────────────────────────────────
def ms_to_sample(ms):
    return int(ms * SR / 1000)

def merge_vad_segments(segments, gap_ms=500):
    """合并间隔 < gap_ms 的相邻 VAD 段。"""
    if not segments:
        return []
    merged = [list(segments[0])]
    for s, e in segments[1:]:
        if s - merged[-1][1] <= gap_ms:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged

def asr_text_clean(text):
    """清理 ASR 输出：截断到最后一个句号/句号。"""
    text = rich_transcription_postprocess(text) if text else ""
    p = max(text.rfind("。"), text.rfind("."))
    if p > 0:
        text = text[:p + 1]
    return text.strip()

# ── 1. 提取全轨音频 ──────────────────────────────────────────────
t0 = time.time()
tmp_audio = os.path.join(out_dir, "tmp.wav")
sp.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1", tmp_audio],
       capture_output=True, check=True)
wav, _ = sf.read(tmp_audio, dtype="float32", always_2d=False)
if wav.ndim > 1:
    wav = wav.mean(axis=-1)
print(f"  audio: {len(wav)/SR:.1f}s, {os.path.getsize(tmp_audio)/1024/1024:.1f}MB")

# ── 2. VAD 检测语音段 ────────────────────────────────────────────
print(f"  Running VAD ({VAD_MODEL})...")
t_vad = time.time()
vad_model = AutoModel(model=VAD_MODEL, device="cuda:0", disable_update=True)
vad_result = vad_model.generate(input=tmp_audio)
vad_segments = vad_result[0]["value"] if vad_result else []
print(f"  VAD: {len(vad_segments)} segments ({time.time()-t_vad:.1f}s)")

merged_segments = merge_vad_segments(vad_segments, MERGE_GAP_MS)
print(f"  After merge (gap<{MERGE_GAP_MS}ms): {len(merged_segments)} segments")

valid_segments = [(s, e) for s, e in merged_segments
                  if (e - s) / 1000 >= MIN_SEG_DURATION]
print(f"  Valid (>={MIN_SEG_DURATION}s): {len(valid_segments)} segments")

# ── 3. ASR: 逐 VAD 段推理 ────────────────────────────────────────
print(f"  Loading ASR model ({ASR_MODEL})...")
t_asr = time.time()
asr_model = AutoModel(model=ASR_MODEL, device="cuda:0", disable_update=True)
print(f"  ASR model loaded ({time.time()-t_asr:.0f}s)")

timeline = []
for seg_idx, (seg_start_ms, seg_end_ms) in enumerate(valid_segments):
    ss = ms_to_sample(seg_start_ms)
    es = min(ms_to_sample(seg_end_ms), len(wav))
    if es <= ss:
        continue

    seg_wav = os.path.join(out_dir, f"vad_seg_{seg_idx:04d}.wav")
    seg_audio = wav[ss:es]
    sf.write(seg_wav, seg_audio.astype(np.float32), SR)

    try:
        res = asr_model.generate(input=seg_wav, language="auto", use_itn=True)
        text = asr_text_clean(res[0]["text"]) if res else ""
    except Exception:
        text = ""
    finally:
        if os.path.exists(seg_wav):
            os.remove(seg_wav)

    timeline.append({
        "start_ms": seg_start_ms,
        "end_ms": seg_end_ms,
        "text": text,
        "lang": "auto",
    })

    if (seg_idx + 1) % 20 == 0:
        print(f"  [{seg_idx+1}/{len(valid_segments)}]")

# ── 4. 输出 asr_timeline.json ────────────────────────────────────
with open(os.path.join(out_dir, "vad_segments.json"), "w") as f:
    json.dump({"method": "fsmn-vad", "merge_gap_ms": MERGE_GAP_MS,
               "n_raw": len(vad_segments), "n_merged": len(merged_segments),
               "n_valid": len(valid_segments), "segments": valid_segments},
              f, ensure_ascii=False, indent=2)

asr_out = {"step": "asr", "n_segments": len(timeline), "results": timeline}
with open(os.path.join(out_dir, "asr_timeline.json"), "w") as f:
    json.dump(asr_out, f, ensure_ascii=False, indent=2)

# ── C5 assert: 输出中无 shot 级字段 ──────────────────────────────
assert "shots" not in asr_out, "C5 失败: 输出包含 shot 级字段"
assert all("shot_id" not in seg for seg in timeline), "C5 失败: timeline 包含 shot_id"
print(f"  C5 pass: 无 shot 级字段")

non_empty = sum(1 for seg in timeline if seg["text"].strip())
print(f"  ASR results: {non_empty}/{len(timeline)} segments have text")
print(f"  done ({time.time()-t0:.0f}s) -> {out_dir}/")
