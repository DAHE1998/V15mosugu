#!/usr/bin/env python3
"""D50 阈值可视化 — 每阈值一个 HTML，横向排列。"""
import json, os, sys, shutil
import numpy as np

THRESHOLDS = [0.95, 0.90, 0.85, 0.80, 0.75]


def dedup(kfs, embs, f2i, thr):
    """返回 (kept, killed) — D50 保留/去重。"""
    if len(kfs) <= 1:
        return set(kfs), set()
    idxs = np.array([f2i[f] for f in kfs if f in f2i])
    valid = [f for f in kfs if f in f2i]
    if len(valid) <= 1:
        return set(kfs), set()
    sub = embs[idxs]
    sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-10)
    kept, killed = set(), set()
    assigned = set()
    for i in range(len(valid)):
        if i in assigned:
            continue
        g = [i]
        for j in range(i + 1, len(valid)):
            if j not in assigned and float(sub[i].dot(sub[j])) > thr:
                g.append(j)
                assigned.add(j)
        assigned.add(i)
        c = sub[g].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-10)
        best = int(np.argmax(sub[g].dot(c)))
        for gi_idx, gi in enumerate(g):
            (kept if gi_idx == best else killed).add(valid[gi])
    return kept, killed


CSS = '''
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,'PingFang SC',sans-serif;background:#111;color:#ddd;padding:16px}
h1{font-size:18px;margin-bottom:4px}
.sub{font-size:12px;color:#888;margin-bottom:12px}
.leg{display:flex;gap:20px;margin-bottom:14px;font-size:12px;align-items:center}
.lb{display:inline-block;width:14px;height:14px;border-radius:3px;margin-right:4px;vertical-align:middle}
.lb.y{background:#f1c40f}
.lb.g{background:#2ecc71}
.lb.r{background:#e74c3c}
.shot{border:1px solid #333;border-radius:6px;padding:10px;margin-bottom:10px;background:#1a1a2e}
.shot h3{font-size:12px;color:#8af;margin-bottom:6px}
.shot h3 span{color:#555;font-weight:400}
.row{display:flex;gap:4px;overflow-x:auto}
.cell{position:relative;border-radius:3px;overflow:hidden;flex-shrink:0;border:3px solid transparent}
.cell.y{border-color:#f1c40f}
.cell.g{border-color:#2ecc71}
.cell.r{border-color:#e74c3c}
.cell img{display:block;width:160px;height:90px;object-fit:cover}
.cell .lbl{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.75);font-size:10px;padding:2px 4px;text-align:center}
.corner{position:absolute;top:0;right:0;width:0;height:0;border-left:20px solid transparent;border-top:20px solid #f1c40f}
'''


def gen_html(thr, multi, results, name, src_dir, out_dir):
    nk = sum(len(results[s['id']][0]) for s in multi)
    nd = sum(len(results[s['id']][1]) for s in multi)

    h = [f'''<!DOCTYPE html>
<html lang="zh">
<head><meta charset="UTF-8"><title>D50 cos>{thr:.2f}</title><style>{CSS}</style></head>
<body>
<h1>D50 段内去重 — cos>{thr:.2f}</h1>
<p class="sub">{len(multi)} multi-frame shots | 🟡 DINO留存 | 🟢{nk} D50保留 | 🔴{nd} D50去重</p>
<div class="leg">
<span><span class="lb y"></span> DINO 关键帧（黄色角标）</span>
<span><span class="lb g"></span> D50 保留</span>
<span><span class="lb r"></span> D50 去重</span>
</div>
''']

    for s in multi:
        sid = s['id']
        kfs = s['key_frames']
        rng = s['range']
        kept, killed = results[sid]
        h.append(f'<div class="shot"><h3>Shot {sid} <span>f{rng["start"]}-f{rng["end"]} ({len(kfs)}fr)</span></h3><div class="row">')
        for fn in kfs:
            cls = 'g' if fn in kept else 'r'
            h.append(f'<div class="cell {cls}"><div class="corner"></div><img src="frames/f{fn}.png"><div class="lbl">f{fn}</div></div>')
        h.append('</div></div>')

    h.append('</body></html>')

    out_path = f'{out_dir}/d50_{int(thr*100):02d}.html'
    with open(out_path, 'w') as f:
        f.write('\n'.join(h))

    frames_dst = f'{out_dir}/frames'
    if not os.path.isdir(frames_dst):
        os.symlink(os.path.relpath(src_dir, out_dir), frames_dst)

    print(f'  cos>{thr:.2f}: {nk}k/{nd}d -> {out_path}')


def main():
    output = sys.argv[1]
    video = sys.argv[2]

    with open(f'{output}/dino_cluster/skeleton.json') as f:
        sk = json.load(f)
    d = np.load(f'{output}/dino_cluster/key_frame_embeddings.npz')
    embs = d['embeddings'].astype(np.float32)
    f2i = {int(fn): i for i, fn in enumerate(d['frame_ids'])}

    multi = [s for s in sk['shots'] if len(s.get('key_frames', [])) > 1]
    name = os.path.basename(video)
    frame_src = f'{output}/d50_visualize/_frames'

    # 每个阈值跑 D50 + 生成独立 HTML
    for thr in THRESHOLDS:
        res = {}
        for s in multi:
            kept, killed = dedup(s['key_frames'], embs, f2i, thr)
            res[s['id']] = (kept, killed)
        gen_html(thr, multi, res, name, frame_src, f'{output}/d50_visualize')

    # 删除旧的 report.html
    old = f'{output}/d50_visualize/report.html'
    if os.path.isfile(old):
        os.remove(old)

    print(f'\n  -> {output}/d50_visualize/d50_*.html')


if __name__ == '__main__':
    main()
