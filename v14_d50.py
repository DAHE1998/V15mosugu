"""V14 D50 — 密度峰值选帧，取代V13的余弦谷值选帧。

V13 D50: 余弦谷值 -> 找帧间变化最大 -> 边界帧
V14 D50: 密度峰值 -> 找子群落密度中心 -> 核心帧
"""
import numpy as np

MAX_KF_PER_SCENE = 6  # 最大关键帧数，避免VLM溢出


def select_frames_by_density(emb, ts, st, et, min_cluster_size=2):
    mask = (ts >= st) & (ts <= et)
    idxs = np.where(mask)[0]
    n_frames = len(idxs)

    if n_frames <= 2:
        return ([ts[idxs[0]], ts[idxs[-1]]] if n_frames >= 2 else [st]), 1
    if n_frames <= 6:
        mid = n_frames // 2
        return [ts[idxs[mid]]], 1

    se = emb[idxs]
    sts = ts[idxs]

    nrm = np.linalg.norm(se, axis=1, keepdims=True)
    nrm[nrm == 0] = 1
    n = se / nrm

    from hdbscan import HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric='euclidean',
        cluster_selection_epsilon=0.25,  # 合并相邻群落，减少碎片
    )
    labels = clusterer.fit_predict(n)
    unique_labels = set(labels) - {-1}

    # 全噪声 -> 密度峰值选帧
    if not unique_labels:
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(5, n_frames), metric='cosine')
            nn.fit(n)
            distances, _ = nn.kneighbors(n)
            best = int(np.argmax(-distances.mean(axis=1)))
            return [sts[best]], 1
        except Exception:
            return [sts[len(sts) // 2]], 1

    # 每个子群落取离密度中心最近的一帧，按群落大小排序取前MAX_KF
    clusters_info = []
    for cid in unique_labels:
        mask_c = labels == cid
        size = int(mask_c.sum())
        sub_emb = n[mask_c]
        centroid = sub_emb.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        sims = sub_emb @ centroid
        best_idx = int(np.argmax(sims))
        sub_indices = np.where(mask_c)[0]
        clusters_info.append((size, sts[sub_indices[best_idx]]))

    # 按群落大小降序取帧（大群落代表主要语义，小群落代表细节）
    clusters_info.sort(key=lambda x: -x[0])
    keyframes = [t for _, t in clusters_info[:MAX_KF_PER_SCENE]]

    return sorted(keyframes), len(unique_labels)
