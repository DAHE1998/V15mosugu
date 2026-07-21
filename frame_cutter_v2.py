#!/usr/bin/env python3
"""frame_cutter_v2 — 单次 GPU decode + 多路并行 GPU encode。

流程:
  Phase 1: ffmpeg GPU decode 整视频 -> rawvideo 文件 (一次 decode)
  Phase 2: Python 按帧号路由 rawvideo 到 per-segment 文件
  Phase 2b: 音频按时间路由到 per-segment wav
  Phase 3: 多路并行 GPU encode (max_workers=5)

禁止: -ss, -to, -t 等时间 seek
"""
import json, sys, os, subprocess as sp, time
from concurrent.futures import ThreadPoolExecutor, as_completed


def probe_video(video_path):
    r = sp.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "default=noprint_wrappers=1", video_path],
        capture_output=True, text=True, check=True
    )
    info = {}
    for line in r.stdout.strip().split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v
    return info


def phase1_decode(video_path, raw_path):
    print(f"[phase 1] GPU decode -> {raw_path}")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-hwaccel", "cuda", "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "rgb24", raw_path,
    ]
    r = sp.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"decode failed: {r.stderr[-300:]}"
    print(f"  {os.path.getsize(raw_path) / 1024 / 1024:.1f} MB")


def phase1b_audio(video_path, audio_path):
    print(f"[phase 1b] extract audio -> {audio_path}")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path, "-vn",
        "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path,
    ]
    r = sp.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"audio failed: {r.stderr[-300:]}"


def phase2_route(raw_path, width, height, segments, seg_dir):
    print(f"[phase 2] route {width}x{height} frames")
    frame_size = width * height * 3
    seg_files = {}
    for seg in segments:
        sid = seg["id"]
        seg_files[sid] = open(os.path.join(seg_dir, f"seg_{sid:04d}.raw"), "wb")

    with open(raw_path, "rb") as f:
        fi = 0
        while True:
            data = f.read(frame_size)
            if len(data) < frame_size:
                break
            for seg in segments:
                if seg["start_frame"] <= fi <= seg["end_frame"]:
                    seg_files[seg["id"]].write(data)
            fi += 1

    for fh in seg_files.values():
        fh.close()

    for seg in segments:
        sid = seg["id"]
        path = os.path.join(seg_dir, f"seg_{sid:04d}.raw")
        expected = seg["end_frame"] - seg["start_frame"] + 1
        actual = os.path.getsize(path) // frame_size
        assert actual == expected, f"seg {sid}: expected {expected}, got {actual}"
    print(f"  {fi} frames routed, verified")


def phase2b_route_audio(audio_path, segments, seg_dir, fps):
    import wave
    print(f"[phase 2b] route audio")
    with wave.open(audio_path, "rb") as wav_in:
        nframes = wav_in.getnframes()
        audio_data = wav_in.readframes(nframes)
        sr = wav_in.getframerate()

    for seg in segments:
        sid = seg["id"]
        start_s = int(seg["start_frame"] / fps * sr)
        end_s = int((seg["end_frame"] + 1) / fps * sr)
        seg_audio = audio_data[start_s * 2:end_s * 2]
        out = os.path.join(seg_dir, f"seg_{sid:04d}.wav")
        with wave.open(out, "wb") as wout:
            wout.setnchannels(1)
            wout.setsampwidth(2)
            wout.setframerate(sr)
            wout.writeframes(seg_audio)
    print(f"  {len(segments)} audio segments routed")


def encode_one(sid, raw_path, wav_path, out_path, width, height, fps):
    n_frames = os.path.getsize(raw_path) // (width * height * 3)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", raw_path,
        "-i", wav_path,
        "-fps_mode", "passthrough",
        "-c:v", "h264_nvenc", "-preset", "p3", "-cq", "26",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest", out_path,
    ]
    r = sp.run(cmd, capture_output=True, text=True, timeout=120)
    ok = r.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    return sid, ok, r.stderr.strip()[-200:] if not ok else ""


def phase3_encode(segments, width, height, fps, seg_dir, out_dir):
    print(f"[phase 3] parallel encode (max_workers=5)")
    os.makedirs(out_dir, exist_ok=True)

    jobs = []
    for seg in segments:
        sid = seg["id"]
        raw = os.path.join(seg_dir, f"seg_{sid:04d}.raw")
        wav = os.path.join(seg_dir, f"seg_{sid:04d}.wav")
        out = os.path.join(out_dir, f"segment_{sid:04d}.mp4")
        jobs.append((sid, raw, wav, out, width, height, fps))

    ok = 0
    fail = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(encode_one, sid, raw, wav, out, w, h, f): sid
            for sid, raw, wav, out, w, h, f in jobs
        }
        for future in as_completed(futures):
            sid, ok_flag, err = future.result()
            if ok_flag:
                ok += 1
            else:
                fail += 1
                if fail <= 3:
                    print(f"  ERR seg {sid}: {err}")
            if (ok + fail) % 20 == 0:
                print(f"  [{ok+fail}/{len(jobs)}] {ok} ok, {fail} fail")

    dt = time.time() - t0
    print(f"\n  {ok}/{len(jobs)} OK ({dt:.1f}s)")
    if fail:
        print(f"  FAILED: {fail}")
        sys.exit(1)


def main():
    if len(sys.argv) < 4:
        print("用法: python frame_cutter_v2.py <input.json> <seg_dir> <output_dir>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    video_path = data["video"]
    segments = data["segments"]
    seg_dir = sys.argv[2]
    out_dir = sys.argv[3]
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    info = probe_video(video_path)
    width = int(info.get("width", 1920))
    height = int(info.get("height", 1080))
    fps = eval(info.get("r_frame_rate", "30/1"))

    print(f"[frame_cutter_v2] {len(segments)} segments")
    print(f"  {width}x{height} @ {fps:.3f}fps")

    raw_path = os.path.join(seg_dir, "full.rgb")
    if not os.path.isfile(raw_path):
        phase1_decode(video_path, raw_path)
    else:
        print(f"[phase 1] cached")

    audio_path = os.path.join(seg_dir, "full.wav")
    if not os.path.isfile(audio_path):
        phase1b_audio(video_path, audio_path)
    else:
        print(f"[phase 1b] cached")

    phase2_route(raw_path, width, height, segments, seg_dir)
    phase2b_route_audio(audio_path, segments, seg_dir, fps)
    phase3_encode(segments, width, height, fps, seg_dir, out_dir)

    print(f"\n[done] -> {out_dir}")


if __name__ == "__main__":
    main()
