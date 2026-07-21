#!/usr/bin/env python3
"""video_cut.py — GPU frame-accurate video cutter, FFmpeg 8.1, one-pass"""

import json, subprocess, sys, os

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <skeleton.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        skel = json.load(f)

    video = skel["video"]
    shots = skel["shots"]
    outdir = os.path.dirname(sys.argv[1]) or "."

    # FPS
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", video],
        capture_output=True, text=True, check=True)
    num, den = map(int, r.stdout.strip().split("/"))
    fps = num / den

    # Source profile/level
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=profile,level",
        "-of", "default=noprint_wrappers=1:nokey=1", video],
        capture_output=True, text=True)
    parts = r.stdout.strip().split()
    profile = parts[0].lower() if len(parts) > 0 else "high"
    raw_level = parts[1] if len(parts) > 1 else "50"
    if raw_level.isdigit():
        level = f"{int(raw_level)//10}.{int(raw_level)%10}"
    else:
        level = raw_level

    # force_key_frames: frame# to timestamp
    kf_ts = []
    for s in shots:
        if s["range"]["start"] > 0:
            kf_ts.append(f'{s["range"]["start"] / fps:.6f}')

    # segment_frames: end+1 for each shot
    sf = ",".join(str(s["range"]["end"] + 1) for s in shots)

    out_pat = os.path.join(outdir, "segment_%04d.mp4")

    print(f"Video: {video}")
    print(f"Shots: {len(shots)}")
    print(f"FPS: {num}/{den} = {fps:.6f}")
    print(f"Source: profile={profile}, level={level}")
    print(f"Output: {out_pat}")

    cmd = [
        "ffmpeg", "-y", "-v", "error", "-stats",
        "-c:v", "h264_cuvid",
        "-i", video,
        "-c:v", "h264_nvenc",
        "-cq", "26",
        "-preset", "p4",
        "-profile:v", profile,
        "-level:v", level,
        "-an",
        "-forced-idr", "1",
        "-movflags", "+faststart",
        "-force_key_frames", ",".join(kf_ts),
        "-f", "segment",
        "-segment_frames", sf,
        "-reset_timestamps", "1",
        out_pat
    ]

    subprocess.run(cmd, check=True)

    # Verify
    ok = fail = 0
    for s in shots:
        f = out_pat % s["id"]
        exp = s["range"]["end"] - s["range"]["start"] + 1
        if not os.path.exists(f):
            print(f"  [{s['id']:3d}] MISSING")
            fail += 1
            continue
        r = subprocess.run(["ffprobe", "-v", "quiet", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0", f],
            capture_output=True, text=True)
        got = r.stdout.strip()
        if got == str(exp):
            ok += 1
        else:
            print(f"  [{s['id']:3d}] FAIL got={got} exp={exp}")
            fail += 1

    print(f"\nOK={ok}  FAIL={fail}  TOTAL={len(shots)}")
    if fail > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
