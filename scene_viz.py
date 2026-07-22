#!/usr/bin/env python3
"""scene_viz — 场景聚类可视化。用法: python scene_viz.py <output_dir>"""
import json, os, sys

output = sys.argv[1]
skel_path = sys.argv[2] if len(sys.argv) > 2 else f"{output}/visual_face_merge/skeleton.json"

with open(skel_path) as f:
    sk = json.load(f)

scenes = sk["proto_scenes"]
shots = sk["shots"]

# 读取人脸数据 — 帧级别
frame_face_count = {}  # frame_num → n_faces
face_is_solo = {}
try:
    with open(f"{output}/face_continuity/face_data.json") as f:
        fd = json.load(f)
    frame_face_count = {int(k): v for k, v in fd.get("per_frame_faces", {}).items()}
    with open(f"{output}/face_continuity/person_chains.json") as f:
        fc = json.load(f)
    for c in fc["chains"]:
        face_is_solo[c["person_id"]] = c["n_shots"] == 1
except:
    pass

colors = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22","#f1c40f",
          "#e91e63","#00bcd4","#ff5722","#8bc34a","#3f51b5","#ff9800"]

CSS = """*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,sans-serif;background:#111;color:#ddd;padding:16px}
h1{font-size:18px;margin-bottom:8px}
.sub{font-size:12px;color:#888;margin-bottom:16px}
.grid{display:flex;gap:6px;overflow-x:auto;padding-bottom:12px}
.col{min-width:100px;max-width:220px;flex-shrink:0;border:1px solid #333;border-radius:6px;overflow:hidden;background:#1a1a2e}
.col .head{padding:8px 10px;font-size:12px;font-weight:600;background:#16213e;border-bottom:1px solid #333}
.col .head .n{color:#8af}.col .head .t{font-size:10px;color:#666}
.shot{padding:6px 8px;border-bottom:1px solid #222;display:flex;flex-direction:column;gap:3px}
.shot .sid{font-size:10px;color:#8af}
.shot .fr{font-size:9px;color:#555}
.shot img{width:100%;height:auto;max-height:100px;object-fit:cover;border-radius:3px;margin-top:2px}
.shot .faces{display:flex;gap:2px;flex-wrap:wrap}
.shot .face{font-size:8px;padding:1px 4px;border-radius:2px;color:#fff}"""

h = [f"<!DOCTYPE html><html lang=zh><head><meta charset=UTF-8><title>Scene Viz</title><style>{CSS}</style></head><body>"]
h.append(f"<h1>DINO + Face 纯视觉场景聚类 — {len(scenes)} scenes</h1>")
h.append(f"<p class=sub>{len(frame_face_count)} frames with faces</p>")
h.append('<div class=grid>')

for sc in scenes:
    label = f"S{sc['id']}"
    h.append(f'<div class=col><div class=head><span class=n>{label}</span> '
             f'<span class=t>{sc["n_shots"]} shots {sc["duration_s"]}s</span></div>')
    for sid in sc["shot_ids"]:
        s = shots[sid]
        r = s["range"]
        kfs = s.get("key_frames", [s.get("representative_frame", 0)])
        # shot 头部 + 人物标签
        shot_tags = ""
        for c in (json.load(open(f"{output}/face_continuity/person_chains.json")) if os.path.exists(f"{output}/face_continuity/person_chains.json") else {}).get("chains", []):
            if sid in c["shots"]:
                pid = c["person_id"]
                clr = colors[pid % len(colors)]
                if c["n_shots"] == 1:
                    shot_tags += f'<span class=face style="background:{clr};opacity:0.4;font-size:7px">s{pid}</span>'
                else:
                    shot_tags += f'<span class=face style=background:{clr}>P{pid}</span>'
        import os as _os
        h.append(f'<div class=shot><div><span class=sid>Shot {sid}</span>'
                 f'<span class=fr>f{r["start"]}-f{r["end"]} ({len(kfs)}kf)</span>'
                 f'<div class=faces>{shot_tags}</div></div>'
                 f'<div style=display:flex;gap:3px;flex-wrap:wrap;margin-top:4px>')
        for kf in kfs:
            nf = frame_face_count.get(kf, 0)
            badge = f'<span style=position:absolute;top:2px;right:2px;background:rgba(0,0,0,.7);color:#0f0;font-size:9px;padding:0 3px;border-radius:2px>{nf}f</span>' if nf > 0 else ''
            h.append(f'<div style=position:relative>'
                     f'<img src=frame_viz/frames/f{kf}.png onerror="this.style.display=\'none\'" '
                     f'style=width:120px;height:68px;object-fit:cover;border-radius:3px title=f{kf}>'
                     f'{badge}'
                     f'</div>')
        h.append('</div></div>')
    h.append('</div>')

h.append('</div></body></html>')

out_dir = os.path.dirname(skel_path)
out = os.path.join(out_dir, "report.html")
with open(out, "w") as f:
    f.write("\n".join(h))
print(f"-> {out}")
