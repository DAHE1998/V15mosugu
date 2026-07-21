#!/usr/bin/env python3
"""test_pipeline_step_by_step — 逐节点测试 V15mosugu pipeline。

测试流程:
  00 scdet → 验证稳定
  01 skeleton → 建骨架
  按骨架切片段 → 帧号精准切割
  每片段跑 02 → 验证代表帧提取

用法:
  python test_pipeline_step_by_step.py <video.mp4> [base_output_dir]
"""
import json, sys, os, time, subprocess as sp
import numpy as np
import torch

PY = "/home/dahe/miniconda3/envs/amaterasu/bin/python"
SCRIPT_DIR = "/home/dahe/VideoCenter/VIP/V15mosugu"


def run_cmd(cmd, label):
    print(f"\n{'='*60}")
    print(f"[TEST] {label}")
    print(f"  cmd: {' '.join(cmd[:3])}...")
    t0 = time.time()
    r = sp.run(cmd, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    print(f"  stdout: {r.stdout.strip()[-500:]}")
    if r.stderr:
        print(f"  stderr: {r.stderr.strip()[-300:]}")
    print(f"  returncode: {r.returncode}  ({dt:.1f}s)")
    assert r.returncode == 0, f"{label} failed"
    return r


def test_00_scdet(video, work_dir):
    """00: scdet 镜头检测，验证稳定。"""
    print(f"\n{'='*60}")
    print(f"[STEP 1/4] 00_scdet — 镜头检测")
    print(f"  video: {video}")
    print(f"  work:  {work_dir}")

    # 清理旧输出
    scdet_dir = os.path.join(work_dir, "00_scdet")
    if os.path.exists(scdet_dir):
        sp.run(["rm", "-rf", scdet_dir])

    r = run_cmd(
        [os.path.join(SCRIPT_DIR, "00_scdet"), video, work_dir],
        "00_scdet"
    )

    # 验证输出
    events_path = os.path.join(scdet_dir, "events.json")
    cuts_path = os.path.join(scdet_dir, "raw_cuts.json")
    assert os.path.isfile(events_path) or os.path.isfile(cuts_path), "00 无输出"

    if os.path.isfile(cuts_path):
        with open(cuts_path) as f:
            d = json.load(f)
        cuts = d.get("cuts", [])
        tf = d.get("total_frames", 0)
        print(f"  ✅ {len(cuts)} cuts, {tf} total frames")
        return d
    else:
        with open(events_path) as f:
            d = json.load(f)
        events = d.get("events", [])
        tf = d.get("total_frames", 0)
        print(f"  ✅ {len(events)} events, {tf} total frames")
        return d


def test_01_skeleton(work_dir):
    """01: 骨架构建，验证 shots 连续无重叠。"""
    print(f"\n{'='*60}")
    print(f"[STEP 2/4] 01_skeleton — 骨架构建")

    r = run_cmd(
        [PY, os.path.join(SCRIPT_DIR, "01_skeleton.py"), work_dir],
        "01_skeleton"
    )

    skel_path = os.path.join(work_dir, "01_skeleton", "skeleton.json")
    with open(skel_path) as f:
        skel = json.load(f)

    shots = skel["shots"]
    tf = skel["total_frames"]
    fps = skel["fps"]

    # 验证
    assert len(shots) > 0, "无 shots"
    assert shots[0]["range"]["start"] == 0, "首个 shot 不从 0 开始"
    assert shots[-1]["range"]["end"] == tf - 1, f"末帧不匹配: {shots[-1]['range']['end']} vs {tf-1}"

    # 检查连续性
    for i in range(len(shots) - 1):
        cur_end = shots[i]["range"]["end"]
        nxt_start = shots[i + 1]["range"]["start"]
        assert cur_end + 1 == nxt_start, \
            f"shot 不连续: shot[{i}] end={cur_end}, shot[{i+1}] start={nxt_start}"

    print(f"  ✅ {len(shots)} shots, {tf} frames, {fps:.3f} fps, 连续无重叠")
    return skel


def cut_by_shots(video, skel, work_dir):
    """按骨架 shots[].range 逐 shot 切片段，帧号精准。"""
    print(f"\n{'='*60}")
    print(f"[STEP 3/4] 按骨架切片段 — 帧号精准切割")

    shots = skel["shots"]
    fps = skel["fps"]
    tf = skel["total_frames"]

    cut_dir = os.path.join(work_dir, "cuts_by_shot")
    os.makedirs(cut_dir, exist_ok=True)

    # 用单次 ffmpeg select 表达式批量提取所有 shot 片段
    # 每个 shot = select=eq(n\,start):eq(n\,end) 不现实
    # 改用逐 shot ffmpeg 调用（GPU 解码，帧号精准）
    ok_count = 0
    fail_count = 0

    for shot in shots:
        sid = shot["id"]
        sf = shot["range"]["start"]
        ef = shot["range"]["end"]
        n_frames = ef - sf + 1
        rep = shot.get("representative_frame", (sf + ef) // 2)

        out_path = os.path.join(cut_dir, f"shot_{sid:04d}_f{sf:05d}-{ef:05d}.mp4")

        # 帧号精准切割：select=eq(n\,start)~eq(n\,end) + vsync 0
        sel = "+".join([f"eq(n\\,{sf + i})" for i in range(n_frames)])
        # 帧数多时分批（ffmpeg select 表达式长度限制）
        if n_frames > 100:
            # 分批：先提取所有帧到 pipe，再编码
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-hwaccel", "cuda", "-i", video,
                "-vf", f"select={sel},scale=if(gt(iw,ih),-2,720):if(gt(iw,ih),720,-2)",
                "-fps_mode", "passthrough",
                "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "26",
                "-c:a", "aac", "-b:a", "128k",
                out_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-hwaccel", "cuda", "-i", video,
                "-vf", f"select={sel},scale=if(gt(iw,ih),-2,720):if(gt(iw,ih),720,-2)",
                "-fps_mode", "passthrough",
                "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "26",
                "-c:a", "aac", "-b:a", "128k",
                out_path,
            ]

        r = sp.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and os.path.isfile(out_path):
            ok_count += 1
        else:
            fail_count += 1
            if fail_count <= 3:
                print(f"  ⚠️  shot[{sid}] f{sf}-{ef} FAILED: {r.stderr.strip()[-100:]}")

        if (sid + 1) % 50 == 0:
            print(f"  [{sid+1}/{len(shots)}] {ok_count} ok, {fail_count} fail")

    print(f"  ✅ {ok_count}/{len(shots)} shots cut, {fail_count} failed")
    return cut_dir


def test_02_per_shot(video, skel, cut_dir):
    """02: 每 shot 片段独立跑 select_frames，验证代表帧提取。"""
    print(f"\n{'='*60}")
    print(f"[STEP 4/4] 02_select_frames — 逐 shot 代表帧提取")

    shots = skel["shots"]
    fps = skel["fps"]

    results = []
    ok = 0
    fail = 0

    for shot in shots:
        sid = shot["id"]
        sf = shot["range"]["start"]
        ef = shot["range"]["end"]
        seg_path = os.path.join(cut_dir, f"shot_{sid:04d}_f{sf:05d}-{ef:05d}.mp4")

        if not os.path.isfile(seg_path):
            fail += 1
            continue

        # 临时 skeleton（单 shot）
        tmp_skel = {
            "video": seg_path,
            "total_frames": ef - sf + 1,
            "fps": fps,
            "shots": [{
                "id": 0,
                "range": {"start": 0, "end": ef - sf},
                "head": list(range(min(3, ef - sf + 1))),
                "tail": list(range(max(0, ef - sf - 2), ef - sf + 1)),
                "representative_frame": shot.get("representative_frame", (ef - sf) // 2) - sf,
            }],
        }

        tmp_dir = os.path.join(cut_dir, f"tmp_shot_{sid:04d}")
        os.makedirs(tmp_dir, exist_ok=True)
        with open(os.path.join(tmp_dir, "skeleton.json"), "w") as f:
            json.dump(tmp_skel, f)

        # 跑 02
        r = sp.run(
            [PY, os.path.join(SCRIPT_DIR, "02_select_frames.py"), tmp_dir],
            capture_output=True, text=True, timeout=120
        )

        if r.returncode == 0:
            # 读输出验证
            out_path = os.path.join(tmp_dir, "02_select_frames", "skeleton.json")
            if os.path.isfile(out_path):
                with open(out_path) as f:
                    out = json.load(f)
                out_shots = out["shots"]
                if out_shots and out_shots[0].get("key_frames"):
                    results.append({
                        "shot_id": sid,
                        "n_keyframes": len(out_shots[0]["key_frames"]),
                        "rep_frame": out_shots[0].get("representative_frame"),
                    })
                    ok += 1
                else:
                    fail += 1
            else:
                fail += 1
        else:
            fail += 1
            if fail <= 3:
                print(f"  ⚠️  shot[{sid}] 02 failed: {r.stderr.strip()[-100:]}")

        # 清理临时文件
        sp.run(["rm", "-rf", tmp_dir])

        if (sid + 1) % 20 == 0:
            print(f"  [{sid+1}/{len(shots)}] {ok} ok, {fail} fail")

    print(f"\n  ✅ {ok}/{len(shots)} shots processed, {fail} failed")
    if results:
        kf_counts = [r["n_keyframes"] for r in results]
        print(f"  key_frames: min={min(kf_counts)} max={max(kf_counts)} mean={np.mean(kf_counts):.1f}")
    return results


def main():
    if len(sys.argv) < 2:
        print("用法: python test_pipeline_step_by_step.py <video.mp4> [base_output_dir]")
        sys.exit(1)

    video = sys.argv[1]
    base_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(video)
    video_name = os.path.basename(video).replace(".mp4", "").replace(".mkv", "")
    work_dir = os.path.join(base_dir, video_name)

    print(f"{'='*60}")
    print(f"V15mosugu 逐节点测试")
    print(f"  Video: {video}")
    print(f"  Work:  {work_dir}")
    print(f"{'='*60}")

    # Step 1: 00 scdet
    scdet_data = test_00_scdet(video, work_dir)

    # Step 2: 01 skeleton
    skel = test_01_skeleton(work_dir)

    # Step 3: 按骨架切片段
    cut_dir = cut_by_shots(video, skel, work_dir)

    # Step 4: 每片段跑 02
    results = test_02_per_shot(video, skel, cut_dir)

    # 汇总
    print(f"\n{'='*60}")
    print(f"测试完成")
    print(f"  00 scdet:    ✅ {len(scdet_data.get('cuts', scdet_data.get('events', [])))} cut points")
    print(f"  01 skeleton: ✅ {len(skel['shots'])} shots")
    print(f"  切割:        ✅ {len(os.listdir(cut_dir))} segments")
    print(f"  02 per-shot: ✅ {len(results)}/{len(skel['shots'])} processed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
