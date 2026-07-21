#!/usr/bin/env python3
"""
Phase 4.5: DINO 嵌入 + 代表帧筛选（独立模块，无 VLM）
=====================================================

输入: Phase 4 抽取的 event_frames/ 目录
输出: events_dino.json（DINO 嵌入 + 每事件最佳代表帧）

用法:
  python3 phase45_dino_embed.py <event_frames_dir> [--output <path>]

模块设计:
  - 独立模块，不依赖 pipeline 骨架
  - 输出 JSON 供后续 02_heartbeat 等步骤消费
"""

import json
import os
import sys
import argparse
import time
import warnings
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image


# ── DINO 初始化（全局单例）──

_MODEL = None
_DEVICE = None
_TRANSFORM = None


def get_dino_model():
    global _MODEL, _DEVICE, _TRANSFORM
    if _MODEL is not None:
        return _MODEL, _DEVICE, _TRANSFORM

    hub_dir = os.path.expanduser("~/.cache/torch/hub/")
    repo_dir = os.path.join(hub_dir, "facebookresearch_dinov2_main")

    if not os.path.isdir(repo_dir):
        # fallback: try download
        print("[DINO] Local repo not found, downloading...")
        _MODEL = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    else:
        sys.path.insert(0, repo_dir)
        from hubconf import dinov2_vitl14
        _MODEL = dinov2_vitl14(pretrained=True)

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _MODEL = _MODEL.to(_DEVICE)
    _MODEL.eval()

    _TRANSFORM = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"[DINO] ViT-L/14 loaded on {_DEVICE} ({sum(p.numel() for p in _MODEL.parameters())/1e6:.0f}M)")
    return _MODEL, _DEVICE, _TRANSFORM


def embed_image(model, device, transform, image_path: str) -> list:
    """返回 1024 维 L2-normalized DINO 嵌入"""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x)  # (1, 1024)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().tolist()


def embed_frames(frame_dir: str, output_path: str = None):
    """主流程：嵌入所有帧 → 按事件聚类 → 选最佳代表帧"""

    # 读取 Phase 4 summary
    summary_path = os.path.join(frame_dir, "extracted_frames.json")
    if not os.path.exists(summary_path):
        # fallback: list all PNGs directly
        pngs = sorted(Path(frame_dir).glob("frame_*.png"))
        frames = {}
        for p in pngs:
            fn = int(p.stem.split("_")[1])
            frames[str(fn)] = {"file": p.name, "exists": True, "events": []}
        summary = {"n_frames_extracted": len(pngs), "frames": frames}
        print(f"[DINO] No extracted_frames.json, found {len(pngs)} PNGs directly")
    else:
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"[DINO] Loaded Phase 4 summary: {summary['n_frames_extracted']} frames")

    model, device, transform = get_dino_model()
    t_start = time.time()

    results = []
    for fn_str, meta in summary["frames"].items():
        frame_num = int(fn_str)
        fpath = os.path.join(frame_dir, meta["file"])
        if not os.path.exists(fpath):
            print(f"[DINO]  SKIP: {meta['file']} not found")
            continue

        emb = embed_image(model, device, transform, fpath)
        results.append({
            "frame": frame_num,
            "file": meta["file"],
            "events": meta.get("events", []),
            "embedding": emb,        # 1024-dim L2-normalized
        })

        if len(results) % 10 == 0:
            elapsed = time.time() - t_start
            fps = len(results) / elapsed if elapsed > 0 else 0
            print(f"[DINO]  {len(results)}/{len(summary['frames'])} embedded ({fps:.1f} img/s)")

    elapsed = time.time() - t_start
    print(f"[DINO]  {len(results)} frames embedded in {elapsed:.1f}s ({len(results)/elapsed:.1f} img/s)")

    # 按事件分组，找每事件最佳帧
    event_best_frames = {}
    for r in results:
        for eid in r["events"]:
            if eid not in event_best_frames:
                event_best_frames[eid] = []
            event_best_frames[eid].append(r["frame"])

    # 构建事件帧嵌入映射
    frame_emb = {r["frame"]: r["embedding"] for r in results}

    # 为每事件选最佳帧：medoid（与窗口内所有帧的余弦距离和最小）
    import numpy as np
    event_selections = {}
    for eid, candidates in event_best_frames.items():
        valid = [fn for fn in candidates if fn in frame_emb]
        if len(valid) <= 1:
            event_selections[eid] = {
                "best_frame": valid[0] if valid else None,
                "n_candidates": len(valid),
            }
            continue

        embs = np.array([frame_emb[fn] for fn in valid])
        # Compute pairwise cosine distances
        sim = embs @ embs.T  # (N, N)
        dist = 1.0 - sim
        # Medoid: frame with minimum sum of distances to all others
        sum_dist = dist.sum(axis=1)
        best_idx = int(np.argmin(sum_dist))
        event_selections[eid] = {
            "best_frame": valid[best_idx],
            "n_candidates": len(valid),
            "medoid_score": float(1.0 - sum_dist[best_idx] / len(valid)),
        }

    # 构建输出
    output = {
        "n_frames": len(results),
        "n_events": len(event_selections),
        "frames": {r["frame"]: r for r in results},
        "event_best_frames": event_selections,
        "embedding_dim": 1024,
    }

    # 去除 embedding（已按事件选完，保留会太大）
    for r in results:
        del r["embedding"]

    if output_path is None:
        output_path = os.path.join(frame_dir, "events_dino.json")

    with open(output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[DINO] Saved: {output_path}")
    print(f"[DINO] Events with best frames: {len(event_selections)}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4.5: DINO embed event frames")
    parser.add_argument("frame_dir", help="Phase 4 event_frames/ directory")
    parser.add_argument("--output", "-o", help="Output JSON path")
    args = parser.parse_args()

    embed_frames(args.frame_dir, args.output)
