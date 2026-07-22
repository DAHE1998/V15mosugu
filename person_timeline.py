#!/usr/bin/env python3
"""person_timeline — 人物镜头分布时间线。用法: python person_timeline.py <output_dir>"""
import json, os, sys

output = sys.argv[1]

with open(f"{output}/face_continuity/person_chains.json") as f:
    fc = json.load(f)
with open(f"{output}/dino_cluster/skeleton.json") as f:
    sk = json.load(f)

shots = sk["shots"]
n = len(shots)

# 每人出现的 shot 列表
person_shots = {}
solo_pids = set()
for c in fc["chains"]:
    person_shots[c["person_id"]] = set(c["shots"])
    if c["n_shots"] == 1:
        solo_pids.add(c["person_id"])

cluster_pids = sorted(set(person_shots.keys()) - solo_pids)
print(f"{len(cluster_pids)} cluster + {len(solo_pids)} solo persons, {n} shots")

colors = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22","#f1c40f",
          "#e91e63","#00bcd4","#ff5722","#8bc34a","#3f51b5","#ff9800","#795548","#607d8b",
          "#c0392b","#2980b9","#27ae60","#d35400","#8e44ad","#16a085","#c0392b","#2c3e50"]

CSS = """*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,sans-serif;background:#111;color:#ddd;padding:16px}
h1{font-size:18px;margin-bottom:8px}
.sub{font-size:12px;color:#888;margin-bottom:16px}
.timeline{position:relative;margin-top:12px}
.row{display:flex;align-items:center;height:24px;margin-bottom:2px}
.row .label{width:50px;font-size:10px;color:#fff;text-align:right;padding-right:8px;flex-shrink:0}
.row .bar-container{flex:1;height:18px;position:relative;background:#1a1a2e;border-radius:2px;overflow:hidden}
.seg{position:absolute;top:0;height:100%;border-radius:2px;opacity:.85}
.axis{display:flex;justify-content:space-between;font-size:9px;color:#555;padding-left:50px;margin-top:4px}
.tip{font-size:10px;color:#aaa;margin-top:8px}
"""

h = [f"<!DOCTYPE html><html lang=zh><head><meta charset=UTF-8><title>Person Timeline</title><style>{CSS}</style></head><body>"]
h.append(f"<h1>人物镜头分布 — {len(cluster_pids)} 聚类 + {len(solo_pids)} 独苗</h1>")
h.append(f"<p class=sub>{n} shots | 鼠标悬停看 shot 号 | 亮=聚类 暗=独苗</p>")

for pi, pid in enumerate(cluster_pids + sorted(solo_pids)):
    sids = sorted(person_shots[pid])
    is_solo = pid in solo_pids
    clr = colors[pi % len(colors)]
    opacity = "0.08" if is_solo else "0.85"
    prefix = "s" if is_solo else "P"
    segs = []
    seg_start = sids[0]
    seg_end = sids[0]
    for s in sids[1:]:
        if s == seg_end + 1:
            seg_end = s
        else:
            segs.append((seg_start, seg_end))
            seg_start = s
            seg_end = s
    segs.append((seg_start, seg_end))

    h.append(f'<div class=row>')
    h.append(f'<div class=label style=color:{clr};opacity:{opacity}>{prefix}{pid}</div>')
    h.append(f'<div class=bar-container>')
    for ss, se in segs:
        left_pct = ss / n * 100
        width_pct = (se - ss + 1) / n * 100
        h.append(f'<div class=seg style="left:{left_pct:.2f}%;width:{width_pct:.2f}%;background:{clr};opacity:{opacity}" '
                 f'title="{prefix}{pid}: shot {ss}-{se} ({se-ss+1})"></div>')
    h.append(f'</div></div>')

# 坐标轴
ticks = []
for i in range(0, n, 25):
    ticks.append(f'<span>shot{i}</span>')
h.append(f'<div class=axis>{"".join(ticks)}</div>')

h.append(f'<p class=tip>悬停色块查看 shot 范围</p>')
h.append('</body></html>')

out = f"{output}/face_continuity/person_timeline.html"
with open(out, "w") as f:
    f.write("\n".join(h))
print(f"-> {out}")
