#!/usr/bin/env python3
"""
d50_visualize — D50 阈值对比可视化 HTML。

只展示有 >1 key_frame 的 shot（D50 真正起作用的镜头）。
每个 shot 的所有 key_frame 按阈值展示 kept(绿)/killed(红)。
"""

import json, os, sys, time
import subprocess as sp
import numpy as np
import base64

THRESHOLDS = [0.95, 0.90, 0.85, 0.80, 0.75]
BATCH_SIZE = 30  # 每批帧数


def extract_frames_batch(video, frame_list, tmp_dir):
    """批量提取多帧：一条 ffmpeg select 表达式提取全部。"""
    os.makedirs(tmp_dir, exist_ok=True)
    for start in range(0, len(frame_list), BATCH_SIZE):
        batch = frame_list[start:start + BATCH_SIZE]
        select_parts = [f"eq(n\\,{fn})" for fn in batch]
        select_expr = "+".join(select_parts)
        sp.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "cuda", "-i", video,
            "-vf", f"select={select_expr}",
            os.path.join(tmp_dir, "_%d.png")
        ], check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        # 重命名: _1.png → f{frame_list[start]}.png
        for i, fn in enumerate(batch):
            src = os.path.join(tmp_dir, f"_{i+1}.png")
            dst = os.path.join(tmp_dir, f"f{fn}.png")
            if os.path.exists(src):
                os.rename(src, dst)
            else:
                # 若某帧不存在（可能是黑场/损坏），补空白占位
                print(f"    WARN: frame {fn} not extracted")
                # 创建小空白占位
                sp.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                        "color=c=black:s=160x90:d=0.1",
                        "-frames:v", "1", dst],
                       stdout=sp.DEVNULL, stderr=sp.DEVNULL)


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_d50(shots, embeddings, frame_to_idx, threshold):
    result = {}
    for s in shots:
        sid = s["id"]
        kf = s.get("key_frames", [])
        valid_kf = [f for f in kf if f in frame_to_idx]
        if len(valid_kf) <= 1:
            result[sid] = {"kept": set(kf), "killed": set()}
            continue

        indices = np.array([frame_to_idx[f] for f in valid_kf])
        sub = embeddings[indices]
        nrm = np.linalg.norm(sub, axis=1, keepdims=True)
        nrm[nrm == 0] = 1
        sub = sub / nrm

        clusters = []
        assigned = set()
        for i in range(len(valid_kf)):
            if i in assigned:
                continue
            group = [i]
            for j in range(i + 1, len(valid_kf)):
                if j in assigned:
                    continue
                if float(sub[i].dot(sub[j])) > threshold:
                    group.append(j)
                    assigned.add(j)
            assigned.add(i)
            clusters.append(group)

        kept, killed = set(), set()
        for grp in clusters:
            grp_emb = sub[grp]
            centroid = grp_emb.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            best_local = int(np.argmax(grp_emb.dot(centroid)))
            for gi_idx, gi in enumerate(grp):
                if gi_idx == best_local:
                    kept.add(valid_kf[gi])
                else:
                    killed.add(valid_kf[gi])
        result[sid] = {"kept": kept, "killed": killed}
    return result


def main():
    if len(sys.argv) < 3:
        print(f"用法: {sys.argv[0]} <output_dir> <video>")
        sys.exit(1)

    output = sys.argv[1]
    video = sys.argv[2]
    out_dir = os.path.join(output, "d50_visualize")
    os.makedirs(out_dir, exist_ok=True)

    # 读取骨架
    src_path = os.path.join(output, "dino_cluster", "skeleton.json")
    with open(src_path) as f:
        skeleton = json.load(f)
    shots = skeleton["shots"]

    # DINO 嵌入
    data = np.load(os.path.join(output, "dino_cluster", "key_frame_embeddings.npz"))
    embeddings = data["embeddings"].astype(np.float32)
    frame_ids = data["frame_ids"]
    frame_to_idx = {int(fn): i for i, fn in enumerate(frame_ids)}

    # 筛选有 >1 key_frame 的 shot
    multi_shots = [s for s in shots if len(s.get("key_frames", [])) > 1]
    all_kf = set()
    for s in multi_shots:
        for fn in s["key_frames"]:
            all_kf.add(fn)
    print(f"  {len(multi_shots)} multi-frame shots, {len(all_kf)} frames to extract")

    # 提取帧（批量）
    tmp_dir = os.path.join(out_dir, "_frames")
    t0 = time.time()
    extract_frames_batch(video, sorted(all_kf), tmp_dir)
    print(f"    extracted {len(all_kf)} frames ({time.time()-t0:.0f}s)")

    # 各阈值
    thresh_results = {}
    for thr in THRESHOLDS:
        res = run_d50(multi_shots, embeddings, frame_to_idx, thr)
        n_kept = sum(len(v["kept"]) for v in res.values())
        n_killed = sum(len(v["killed"]) for v in res.values())
        thresh_results[thr] = res
        print(f"  cos>{thr:.2f}: {n_kept} kept / {n_killed} killed")

    # --- 生成 HTML ---
    print("  generating HTML...")
    html = []
    html.append("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>D50 去重可视化</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,'PingFang SC',sans-serif;background:#1a1a2e;color:#eee;padding:20px;}
h1{font-size:20px;margin-bottom:4px}
.summary{font-size:13px;color:#888;margin-bottom:16px}
.tabs{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.tab{padding:6px 14px;border-radius:4px;cursor:pointer;font-size:12px;border:1px solid #444;background:#2a2a3e;color:#aaa}
.tab.active{background:#4a6cf7;color:#fff;border-color:#4a6cf7}
.tab .n{font-size:10px;opacity:.7}
.legend{display:flex;gap:16px;margin-bottom:16px;font-size:12px}
.legend-item{display:flex;align-items:center;gap:6px}
.legend-box{width:14px;height:14px;border-radius:3px}
.legend-box.kept{background:#2ecc71}
.legend-box.killed{background:#e74c3c}
.shot{border:1px solid #333;border-radius:6px;padding:10px;margin-bottom:12px;background:#16213e}
.shot h3{font-size:13px;color:#8af;margin-bottom:6px}
.shot h3 span{color:#666;font-weight:400}
.row{display:flex;gap:5px;flex-wrap:wrap;margin-top:4px}
.cell{position:relative;border-radius:3px;overflow:hidden;border:2px solid transparent}
.cell.kept{border-color:#2ecc71}
.cell.killed{border-color:#e74c3c}
.cell img{display:block;width:140px;height:79px;object-fit:cover}
.cell .label{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.7);font-size:10px;padding:1px 4px;text-align:center}
.d{display:none}
.d.active{display:flex;flex-wrap:wrap;gap:5px}
</style>
</head>
<body>
<h1>D50 段内去重 — 阈值对比</h1>
<p class="summary">%d shots (>1 key_frame), %d frames, video: %s</p>
<div class="legend">
  <div class="legend-item"><div class="legend-box kept"></div> 保留</div>
  <div class="legend-item"><div class="legend-box killed"></div> 被去重</div>
</div>
<div class="tabs">
""" % (len(multi_shots), len(all_kf), os.path.basename(video)))

    for thr in THRESHOLDS:
        nk = sum(len(thresh_results[thr][s["id"]]["kept"]) for s in multi_shots)
        nl = sum(len(thresh_results[thr][s["id"]]["killed"]) for s in multi_shots)
        a = "active" if thr == THRESHOLDS[0] else ""
        html.append(f'<div class="tab {a}" onclick="t({thr*100:.0f})">cos&gt;{thr:.2f} <span class="n">{nk}k/{nl}d</span></div>')
    html.append("</div>")

    for s in multi_shots:
        sid = s["id"]
        kfs = s["key_frames"]
        r = s["range"]
        html.append(f'<div class="shot"><h3>Shot {sid} <span>f{r["start"]}-f{r["end"]} ({len(kfs)}fr)</span></h3>')

        # 每个阈值列
        for thr in THRESHOLDS:
            res = thresh_results[thr][sid]
            a = "active" if thr == THRESHOLDS[0] else ""
            html.append(f'<div class="d {a}" data-t="{thr*100:.0f}">')
            for fn in kfs:
                path = os.path.join(tmp_dir, f"f{fn}.png")
                if not os.path.isfile(path):
                    continue
                b64 = img_to_base64(path)
                status = "kept" if fn in res["kept"] else "killed"
                html.append(f'<div class="cell {status}"><img src="data:image/jpeg;base64,{b64}"><div class="label">f{fn}</div></div>')
            html.append("</div>")
        html.append("</div>")

    html.append("""<script>
function t(v){document.querySelectorAll('.d').forEach(e=>e.classList.toggle('active',+e.dataset.t===v))
document.querySelectorAll('.tab').forEach(e=>e.classList.toggle('active',Math.round(+e.textContent.split('>')[1].split('<')[0]*100)===v))}
</script></body></html>""")

    report_path = os.path.join(out_dir, "report.html")
    with open(report_path, "w") as f:
        f.write("\n".join(html))
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
