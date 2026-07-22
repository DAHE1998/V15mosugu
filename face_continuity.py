#!/usr/bin/env python3
"""
face_continuity — 人脸连续矩阵 (Claude 方案 v2)。

修复 max-pool + Union-Find 的 chaining 误合并:
  1. 主脸按 bbox 面积选（面积最大 = 镜头主体，不按置信度）
  2. 在线质心聚类：按时间顺序，主脸比对已确认人物质心
  3. 边界匹配日志：记录每次匹配的 top-2 相似度差

输入:  d50_dedup/skeleton.json (shots[].representative_frame)
输出:  face_continuity/
         face_data.json       — per-shot 人脸 (area + embedding)
         person_chains.json   — 人物链
         match_log.json       — 边界匹配日志
"""

import json, os, sys, time, subprocess as sp
import numpy as np
import cv2

DET_SIZE = (640, 640)
MIN_DET_SCORE = 0.3
MIN_FACE_AREA = 5000   # 最小人脸面积（像素），过滤广告牌等小脸
DBSCAN_EPS = 0.55           # 1-cos 距离阈值 (cos≥0.45 归为同人)
DBSCAN_MIN_SAMPLES = 2
MAX_CENTROID_WINDOW = 10


def extract_rep_frames(video, frame_set, tmp_dir):
    """GPU pipe 全片解码 960×540 → 只存需要的帧。"""
    os.makedirs(tmp_dir, exist_ok=True)
    W, H = 960, 540
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
            cv2.imwrite(f"{tmp_dir}/f{fn}.png", bgr)
            saved += 1
    proc.stdout.close()
    proc.wait()
    return saved


class PersonTrack:
    """在线质心人物追踪。"""
    def __init__(self, pid, embedding, shot_id):
        self.pid = pid
        self.shots = [shot_id]
        self.centroid = np.array(embedding, dtype=np.float32)
        self.recent = [np.array(embedding, dtype=np.float32)]

    def add(self, embedding, shot_id):
        self.shots.append(shot_id)
        emb = np.array(embedding, dtype=np.float32)
        self.recent.append(emb)
        if len(self.recent) > MAX_CENTROID_WINDOW:
            self.recent.pop(0)
        # 质心 = 近期嵌入的平均
        self.centroid = np.mean(self.recent, axis=0)
        self.centroid /= np.linalg.norm(self.centroid) + 1e-10

    def sim(self, embedding):
        emb = np.array(embedding, dtype=np.float32)
        return float(np.dot(emb, self.centroid))


def main():
    if len(sys.argv) < 3:
        print(f"用法: {sys.argv[0]} <output_dir> <video>")
        sys.exit(1)

    output, video = sys.argv[1], sys.argv[2]

    # ── 读取骨架 ──
    sk_path = os.path.join(output, "d50_dedup", "skeleton.json")
    if not os.path.isfile(sk_path):
        sk_path = os.path.join(output, "02_select_frames", "skeleton.json")
    with open(sk_path) as f:
        skeleton = json.load(f)

    shots = skeleton["shots"]
    n = len(shots)
    print(f"[face_continuity v2] {n} shots")

    out_dir = os.path.join(output, "face_continuity")
    os.makedirs(out_dir, exist_ok=True)

    # ── 收集所有 key_frames（不用 representative_frame）──
    all_key_frames = set()
    shot_kf_map = {}  # shot_id → [frame_nums]
    for si, s in enumerate(shots):
        kfs = s.get("key_frames", [])
        if not kfs:
            rf = s.get("representative_frame")
            kfs = [rf] if rf else []
        shot_kf_map[si] = kfs
        for fn in kfs:
            all_key_frames.add(fn)
    print(f"  {len(all_key_frames)} unique key_frames across {n} shots")

    # ── 1. 提取帧 ──
    tmp = os.path.join(out_dir, "_tmp")
    t0 = time.time()
    extract_rep_frames(video, all_key_frames, tmp)
    print(f"  frames: {len(all_key_frames)} ({time.time()-t0:.0f}s)")

    # ── 2. 人脸检测 ──
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    # frame → faces
    frame_faces = {}
    for fn in all_key_frames:
        img_path = os.path.join(tmp, f"f{fn}.png")
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = app.get(img)
        faces = [f for f in faces if f.det_score >= MIN_DET_SCORE and
                 (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) >= MIN_FACE_AREA]
        if not faces:
            continue
        for f in faces:
            f._area = (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        faces.sort(key=lambda f: f._area, reverse=True)
        frame_faces[fn] = [
            {"area": float(f._area), "det_score": float(f.det_score),
             "embedding": f.normed_embedding.tolist()}
            for f in faces[:5]
        ]

    # 聚合到 shot 级别：取该 shot 所有 key_frames 中的人脸
    shot_all_faces = [[] for _ in range(n)]
    for si in range(n):
        all_f = []
        for fn in shot_kf_map.get(si, []):
            all_f.extend(frame_faces.get(fn, []))
        # 按面积排，去重（相同 embedding dot>0.99 视为同一张脸）
        all_f.sort(key=lambda x: x["area"], reverse=True)
        deduped = []
        for f in all_f:
            dup = False
            for d in deduped:
                if np.dot(f["embedding"], d["embedding"]) > 0.99:
                    dup = True; break
            if not dup:
                deduped.append(f)
        shot_all_faces[si] = deduped[:5]
            # all faces already stored in shot_all_faces[si] above

    # 清理临时帧
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)

    # ── 3. DBSCAN 图聚类 ──
    #    构建 M×M 余弦矩阵，DBSCAN 通过密度连通连接侧脸和正脸
    from sklearn.cluster import DBSCAN

    all_faces = []  # [(shot_id, face_idx, embedding)]
    for si in range(n):
        for fi, f in enumerate(shot_all_faces[si]):
            all_faces.append((si, fi, np.array(f["embedding"], dtype=np.float32)))

    M = len(all_faces)
    print(f"  {M} face detections, building {M}×{M} matrix...")
    t0 = time.time()
    embs = np.stack([e for _, _, e in all_faces])        # M×512
    sim_matrix = embs @ embs.T                            # M×M cosine
    dist_matrix = 1.0 - np.clip(sim_matrix, 0, 1)        # cosine distance
    print(f"    done ({time.time()-t0:.2f}s)")

    # DBSCAN: eps=1-threshold (cos 0.45 → dist 0.55), min_samples=2
    EPS = 1.0 - 0.45
    clustering = DBSCAN(eps=EPS, min_samples=2, metric="precomputed").fit(dist_matrix)
    labels = clustering.labels_

    # 构建 tracks
    tracks = []
    shot_assignments = [[] for _ in range(n)]
    match_log = []

    label_to_pid = {}
    for fi, label in enumerate(labels):
        si, face_idx, emb = all_faces[fi]
        if label == -1:
            # solo track — 独苗也保留
            pid = len(tracks)
            tracks.append(PersonTrack(pid, emb.tolist(), si))
            shot_assignments[si].append(pid)
            match_log.append({"shot": si, "face_idx": face_idx, "match": f"SOLO_P{pid}"})
        else:
            if label not in label_to_pid:
                label_to_pid[label] = len(tracks)
                tracks.append(PersonTrack(len(tracks), emb.tolist(), si))
            pid = label_to_pid[label]
            tracks[pid].add(emb.tolist(), si)
            shot_assignments[si].append(pid)
            match_log.append({"shot": si, "face_idx": face_idx, "match": f"P{pid}"})

    n_solo = sum(1 for l in labels if l == -1)
    n_clust = len(tracks) - n_solo
    print(f"  DBSCAN: {n_clust} clusters + {n_solo} solo tracks")

    # ── 4. 输出 ──
    # face_data.json + 每帧人脸数
    face_data = {
        "step": "face_continuity_v2",
        "method": "bbox_area_primary + DBSCAN",
        "n_shots": n,
        "n_with_faces": sum(1 for faces in shot_all_faces if faces),
        "centroid_threshold": DBSCAN_EPS,
        "per_frame_faces": {str(fn): len(ff) for fn, ff in frame_faces.items()},
        "results": [
            {
                "shot_id": si,
                "frames_checked": shot_kf_map.get(si, []),
                "n_faces": len(shot_all_faces[si]),
                "all_faces": shot_all_faces[si],
            }
            for si in range(n)
        ],
    }
    with open(os.path.join(out_dir, "face_data.json"), "w") as f:
        json.dump(face_data, f, ensure_ascii=False, indent=2)

    # person_chains.json (去重：同一 shot 可能被同 track 多次添加)
    chains = []
    for t in tracks:
        unique_shots = sorted(set(t.shots))
        if len(unique_shots) >= 1:
            chains.append({
                "person_id": t.pid,
                "shots": unique_shots,
                "n_shots": len(unique_shots),
            })
    with open(os.path.join(out_dir, "person_chains.json"), "w") as f:
        json.dump({"threshold": DBSCAN_EPS, "method": "DBSCAN",
                   "n_tracks": len(tracks), "chains": chains}, f, ensure_ascii=False, indent=2)

    # match_log.json
    with open(os.path.join(out_dir, "match_log.json"), "w") as f:
        json.dump(match_log, f, ensure_ascii=False, indent=2)

    # 报告
    n_face = sum(1 for faces in shot_all_faces if faces)
    print(f"\n  {n_face}/{n} shots have faces")
    print(f"  {len(tracks)} person tracks ({len(chains)} with ≥2 shots)")
    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    main()
