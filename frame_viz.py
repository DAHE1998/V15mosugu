#!/usr/bin/env python3
"""
frame_viz — 多端口帧可视化独立模块。

端口规则（按帧数降序自动分配）:
  Port1: 帧数最多 → 无框 (02原始帧)
  Port2: 第二多   → 黄框 (DINO)
  Port3: 第三多   → 绿框 (D50 kept)
  Port4+:         → 蓝/红/紫...

用法:
  python frame_viz.py <output_dir> <video> <json1> [json2 json3 ...]
"""

import json, os, sys, time, subprocess as sp
import numpy as np
import cv2

PORT_BORDERS = [None, None, "#f1c40f", "#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#e67e22"]
W, H = 640, 360


def extract_frames(data):
    """从 JSON 提取帧号集合。优先 shots[].key_frames。"""
    frames = set()
    if isinstance(data, dict) and "shots" in data:
        for s in data["shots"]:
            for fn in s.get("key_frames", []):
                if isinstance(fn, int):
                    frames.add(fn)
        return frames
    # 兜底
    for k in ("frames", "kept", "frame_ids", "key_frames"):
        if k in data and isinstance(data[k], list):
            for fn in data[k]:
                if isinstance(fn, int):
                    frames.add(fn)
    if not frames and isinstance(data, list):
        for x in data:
            if isinstance(x, int):
                frames.add(x)
    return frames


def gen_html(ports, all_frames, out_path):
    """生成 HTML。边框按最高端口号着色。"""
    shots = []
    try:
        with open(ports[0]["path"]) as f:
            sk = json.load(f)
            shots = sk.get("shots", [])
    except:
        pass

    # border CSS classes
    border_css = "\n".join(
        f".cell.p{i}{{border-color:{PORT_BORDERS[i]}}}"
        for i in range(2, len(PORT_BORDERS)) if PORT_BORDERS[i]
    )

    css = f"""*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,'PingFang SC',sans-serif;background:#111;color:#ddd;padding:16px}}
h1{{font-size:18px;margin-bottom:4px}}
.sub{{font-size:12px;color:#888;margin-bottom:12px}}
.leg{{display:flex;gap:16px;margin-bottom:14px;font-size:12px;flex-wrap:wrap;align-items:center}}
.ldot{{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;vertical-align:middle}}
.shot{{border:1px solid #333;border-radius:6px;padding:10px;margin-bottom:10px;background:#1a1a2e}}
.shot h3{{font-size:12px;color:#8af;margin-bottom:6px}}
.shot span{{color:#555;font-weight:400}}
.row{{display:flex;gap:3px;overflow-x:auto;align-items:start}}
.cell{{position:relative;border-radius:3px;overflow:hidden;flex-shrink:0;border:3px solid #444}}
{border_css}
.cell img{{display:block;width:160px;height:90px;object-fit:cover}}
.cell .lbl{{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.75);font-size:10px;padding:1px 4px;text-align:center}}"""

    h = [f"<!DOCTYPE html><html lang=zh><head><meta charset=UTF-8><title>Frame Viz</title><style>{css}</style></head><body>"]
    h.append("<h1>Frame Viz — 多端口帧对比</h1>")

    # 图例
    h.append('<p class=sub>')
    for p in ports:
        c = p["color"] or "#444"
        h.append(f'<span style="margin-right:12px"><span class=ldot style=background:{c};border:1px solid {c}"></span>{p["label"]} ({len(p["frames"])}fr)</span>')
    h.append('</p>')

    # 按 shot 展示
    if shots:
        for s in shots:
            kfs = s.get("key_frames", [])
            if not kfs:
                continue
            r = s["range"]
            h.append(f'<div class=shot><h3>Shot {s["id"]} <span>f{r["start"]}-f{r["end"]} ({len(kfs)}fr)</span></h3><div class=row>')
            for fn in kfs:
                # 找该帧的最高端口
                max_port = 0
                for pi, p in enumerate(ports):
                    if fn in p["frames"]:
                        max_port = pi + 1
                cls = f" p{max_port}" if max_port >= 2 else ""
                h.append(f'<div class="cell{cls}"><img src="frames/f{fn}.png"><div class=lbl>f{fn}</div></div>')
            h.append("</div></div>")
    else:
        h.append('<div class=row>')
        for fn in sorted(all_frames):
            max_port = 0
            for pi, p in enumerate(ports):
                if fn in p["frames"]:
                    max_port = pi + 1
            cls = f" p{max_port}" if max_port >= 2 else ""
            h.append(f'<div class="cell{cls}"><img src="frames/f{fn}.png"><div class=lbl>f{fn}</div></div>')
        h.append('</div>')

    h.append("</body></html>")
    with open(out_path, "w") as f:
        f.write("\n".join(h))


def main():
    if len(sys.argv) < 4:
        print(f"用法: {sys.argv[0]} <output_dir> <video> <json1> [json2 json3 ...]")
        sys.exit(1)

    output_dir, video = sys.argv[1], sys.argv[2]
    json_paths = sys.argv[3:]

    # ── 读取 JSON，提取帧号 ──
    port_data = []
    for jp in json_paths:
        with open(jp) as f:
            data = json.load(f)
        frames = extract_frames(data)
        port_data.append({"path": jp, "data": data, "frames": frames, "n": len(frames)})

    # 按帧数降序 → 端口号
    port_data.sort(key=lambda x: x["n"], reverse=True)
    for i, p in enumerate(port_data):
        p["port"] = i + 1
        p["color"] = PORT_BORDERS[i + 1] if i + 1 < len(PORT_BORDERS) else "#fff"
        # 标签名 = 父目录名（如 02_select_frames, d50_dedup）
        dirname = os.path.basename(os.path.dirname(p["path"]))
        p["label"] = f"P{i+1}:{dirname}"

    print(f"[frame_viz] {len(port_data)} ports:")
    for p in port_data:
        print(f"  Port{p['port']}: {p['n']}fr [{p['color']}] ← {os.path.basename(p['path'])}")

    all_frames = sorted(set.union(*[p["frames"] for p in port_data]))
    print(f"  total unique: {len(all_frames)}")

    # ── GPU 提取帧 ──
    TMP = os.path.join(output_dir, "frame_viz", "_frames")
    os.makedirs(TMP, exist_ok=True)

    total = int(sp.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=noprint_wrappers=1:nokey=1", video
    ]).strip() or 0)
    if not total:
        dur = float(sp.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video
        ]).strip())
        fps = eval(sp.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", video
        ]).strip())
        total = int(dur * fps)

    frame_set = set(all_frames)
    print(f"  GPU pipe decode {total} frames...")
    t0 = time.time()

    proc = sp.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", video,
        "-vf", f"scale_cuda={W}:{H},hwdownload,format=nv12",
        "-f", "rawvideo", "-pix_fmt", "nv12", "-"
    ], stdout=sp.PIPE, stderr=sp.DEVNULL)

    fb = W * H * 3 // 2
    saved = 0
    for fn in range(total):
        raw = proc.stdout.read(fb)
        if len(raw) < fb:
            break
        if fn in frame_set:
            yuv = np.frombuffer(raw, np.uint8).reshape(H * 3 // 2, W)
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
            cv2.imwrite(f"{TMP}/f{fn}.png", bgr)
            saved += 1
            if saved % 50 == 0:
                print(f"    {saved}/{len(all_frames)} ({time.time()-t0:.0f}s)")
    proc.stdout.close()
    proc.wait()
    print(f"  {saved} frames ({time.time()-t0:.0f}s)")

    # ── HTML ──
    out_dir = os.path.join(output_dir, "frame_viz")
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, "report.html")
    gen_html(port_data, all_frames, html_path)

    # symlink frames
    fdst = os.path.join(out_dir, "frames")
    os.makedirs(fdst, exist_ok=True)
    for fn in os.listdir(TMP):
        if fn.startswith("f"):
            ln = f"{fdst}/{fn}"
            if not os.path.exists(ln):
                os.symlink(os.path.join(TMP, fn), ln)

    print(f"  -> {html_path}")


if __name__ == "__main__":
    main()
