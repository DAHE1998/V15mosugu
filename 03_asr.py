#!/usr/bin/env python3
"""03 — ASR 转录: VAD 整轨分段 → SenseVoiceSmall → 时间映射回 shot。

输入:  02_select_frames/skeleton.json (shots[] + representative_frame)
输出:  03_asr/skeleton.json  (+ shot.asr_text)
       03_asr/asr_output.json
       03_asr/vad_segments.json  ← VAD 检测的语音段
"""
import json, sys, os, time, subprocess as sp, math
import numpy as np, soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

output = sys.argv[1]
in_path = os.path.join(output, "02_select_frames", "skeleton.json")
with open(in_path) as f:
    skeleton = json.load(f)

out_dir = os.path.join(output, "03_asr")
os.makedirs(out_dir, exist_ok=True)

shots = skeleton["shots"]
video = skeleton["video"]
tf = skeleton["total_frames"]
dur = skeleton.get("duration", 0)
fps = skeleton["fps"]
SR = 16000
VAD_MODEL = "fsmn-vad"
ASR_MODEL = "/home/dahe/.cache/modelscope/models/iic/SenseVoiceSmall"
MERGE_GAP_MS = 500        # 合并间隔 <500ms 的相邻 VAD 段
MIN_SEG_DURATION = 0.3    # 最短有效语音段 (秒)

print(f"[03] VAD ASR: {len(shots)} shots, {dur:.1f}s video")

# ── 工具函数 ──────────────────────────────────────────────────────
def frame_to_sample(fn):
    return int(fn * dur * SR / tf) if tf and dur else int(fn / fps * SR)

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
sp.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
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

# 合并相邻段
merged_segments = merge_vad_segments(vad_segments, MERGE_GAP_MS)
print(f"  After merge (gap<{MERGE_GAP_MS}ms): {len(merged_segments)} segments")

# 过滤太短的段
valid_segments = [(s, e) for s, e in merged_segments
                  if (e - s) / 1000 >= MIN_SEG_DURATION]
print(f"  Valid (≥{MIN_SEG_DURATION}s): {len(valid_segments)} segments")

# 保存 VAD 结果
with open(os.path.join(out_dir, "vad_segments.json"), "w") as f:
    json.dump({"method": "fsmn-vad", "merge_gap_ms": MERGE_GAP_MS,
               "n_raw": len(vad_segments), "n_merged": len(merged_segments),
               "n_valid": len(valid_segments), "segments": valid_segments},
              f, ensure_ascii=False, indent=2)

# ── 3. ASR: 逐 VAD 段推理 + 映射到 shots ────────────────────────
print(f"  Loading ASR model ({ASR_MODEL})...")
t_asr = time.time()
asr_model = AutoModel(model=ASR_MODEL, device="cuda:0", disable_update=True)
print(f"  ASR model loaded ({time.time()-t_asr:.0f}s)")

# 先对所有 VAD 段做 ASR，收集 (seg_text, seg_start_s, seg_end_s)
seg_results = []

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

    seg_results.append((seg_start_ms / 1000, seg_end_ms / 1000, text))

    if (seg_idx + 1) % 20 == 0:
        print(f"  [{seg_idx+1}/{len(valid_segments)}]")

# 将 ASR 结果映射到所有重叠的 shots（一段旁白可能跨多个 shot）
shot_asr = {s["id"]: [] for s in shots}

for seg_start_s, seg_end_s, text in seg_results:
    if not text:
        continue

    seg_dur = seg_end_s - seg_start_s
    # 将文本按时间比例切分给重叠的 shot
    for shot in shots:
        shot_start_s = shot["range"]["start"] / fps
        shot_end_s = (shot["range"]["end"] + 1) / fps
        overlap = max(0, min(seg_end_s, shot_end_s) - max(seg_start_s, shot_start_s))
        if overlap < 0.3:
            continue
        # 文本按重叠位置切分: 该 shot 在段内的比例
        ratio = overlap / seg_dur
        char_start = int((max(shot_start_s, seg_start_s) - seg_start_s) / seg_dur * len(text))
        char_end = int((min(shot_end_s, seg_end_s) - seg_start_s) / seg_dur * len(text))
        shot_text = text[char_start:char_end].strip()
        if shot_text:
            shot_asr[shot["id"]].append(shot_text)

# ── 4. 将 ASR 结果填入骨架 ──────────────────────────────────────
for shot in shots:
    sid = shot["id"]
    texts = shot_asr[sid]
    if texts:
        # 合并所有段文本，strip 尾部句号防止叠句号
        cleaned = [t.rstrip("。") for t in texts if t.rstrip("。")]
        shot["asr_text"] = "。".join(cleaned)
    else:
        shot["asr_text"] = ""

# 统计
non_empty = sum(1 for s in shots if s.get("asr_text", "").strip())
short_asr = sum(1 for s in shots
                if s.get("asr_text", "").strip()
                and len(__import__('re').sub(r'[^一-鿿\w]', '', s["asr_text"])) <= 4)
print(f"  ASR results: {non_empty}/{len(shots)} shots have text")
print(f"  Short ASR (≤4 chars): {short_asr}/{len(shots)}")
print(f"  done ({time.time()-t0:.0f}s) -> {out_dir}/")

# ── 输出 ─────────────────────────────────────────────────────────
skeleton["shots"] = shots
skel_path = os.path.join(out_dir, "skeleton.json")
with open(skel_path, "w") as f:
    json.dump(skeleton, f, ensure_ascii=False, indent=2)

asr_out = [{"id": s["id"], "range": s["range"],
            "asr_text": s.get("asr_text", "")} for s in shots]
ref_path = os.path.join(out_dir, "asr_output.json")
with open(ref_path, "w") as f:
    json.dump({"step": "03_asr", "n_shots": len(shots),
               "n_non_empty": non_empty, "n_short": short_asr,
               "results": asr_out}, f, ensure_ascii=False, indent=2)
