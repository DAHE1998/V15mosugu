#!/usr/bin/env python3
"""frame_cutter — 帧级精准切割，纯 JSON 驱动。

输入 JSON 格式:
{
  "video": "input.mp4",
  "segments": [
    {"id": 0, "start_frame": 0,   "end_frame": 50},
    {"id": 1, "start_frame": 51,  "end_frame": 316}
  ]
}

输出: output_dir/segment_000.mp4, segment_001.mp4 ...

禁止: -ss, -to, -t 等时间 seek
方法: select/aselect filter + between(n,start,end) 帧号路由
编码: GPU h264_nvenc, 音频 aac re-encode
PTS: 每个 segment 从 0 重新开始
"""
import json, sys, os, subprocess as sp


def cut_segment(video_path, seg, out_dir):
    sid = seg["id"]
    sf = seg["start_frame"]
    ef = seg["end_frame"]

    out_path = os.path.join(out_dir, f"segment_{sid:04d}.mp4")

    vf = f"select='between(n\\,{sf}\\,{ef})',setpts=PTS-STARTPTS"
    af = f"aselect='between(n\\,{sf}\\,{ef})',asetpts=PTS-STARTPTS"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-hwaccel", "cuda", "-i", video_path,
        "-vf", vf,
        "-fps_mode", "passthrough",
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "26",
        "-af", af,
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]

    r = sp.run(cmd, capture_output=True, text=True, timeout=300)
    ok = r.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    status = "OK" if ok else f"ERR: {r.stderr.strip()[-200:]}"
    print(f"  [{sid:4d}] f{sf:05d}-{ef:05d} ({ef - sf + 1}fr) -> {os.path.basename(out_path)} {status}")
    return ok, sid


def main():
    if len(sys.argv) < 3:
        print("用法: python frame_cutter.py <input.json> <output_dir>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    video_path = data["video"]
    segments = data["segments"]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    print(f"[frame_cutter] {len(segments)} segments -> {out_dir}")

    results = []
    for seg in segments:
        ok, sid = cut_segment(video_path, seg, out_dir)
        results.append((sid, ok))

    ok_count = sum(1 for _, ok in results if ok)
    fail_count = len(results) - ok_count
    print(f"[done] {ok_count}/{len(segments)} OK")
    if fail_count:
        print(f"  FAILED: {fail_count}")
        for sid, ok in results:
            if not ok:
                print(f"    segment_{sid:04d}")
        sys.exit(1)


if __name__ == "__main__":
    main()
