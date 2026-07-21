#!/usr/bin/env python3
"""
selector/dino_selector.py — DINO on candidates only (Section 9-10)

不重新 decode。从 decoder 缓存 frame tensor。
GPU batch inference → DINO ViT-L/14 → semantic curve → peak detection。

输出: semantic_curve.json, scenes.json
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# ── 参数 ────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32
DINO_COS_THR = 0.70  # centroid drift 阈值


def load_dinov2():
    """加载 DINOv2 ViT-L/14 (FP16)。"""
    print(f"[dino] loading DINOv2 on {DEVICE} ...")
    t0 = time.time()
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model = model.to(DEVICE).eval().half()
    torch.backends.cudnn.benchmark = True
    print(f"  loaded ({time.time()-t0:.1f}s)")
    return model


TR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


@torch.no_grad()
def compute_embeddings(model, frame_tensors):
    """
    批量计算 DINO embeddings。

    Args:
        model: DINOv2 model
        frame_tensors: list of (3, H, W) float32 tensors on GPU

    Returns:
        np.ndarray (N, 1024) float32
    """
    n = len(frame_tensors)
    embs = []
    for i in range(0, n, BATCH):
        batch = frame_tensors[i:i+BATCH]
        # TR expects PIL/numpy, convert from GPU tensor
        batch_np = [t.cpu().permute(1, 2, 0).numpy() for t in batch]
        tensors = torch.stack([TR(img) for img in batch_np]).to(DEVICE).half()
        with torch.autocast(DEVICE, torch.float16):
            e = model(tensors).float()
        embs.append(e.cpu().numpy())
    return np.concatenate(embs).astype(np.float32) if embs else np.array([])


def load_frame_tensors(video_path, frame_indices, device="cuda"):
    """
    从视频加载指定帧的 tensor (GPU)。
    不加载所有帧，只加载候选帧。
    """
    sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    from decoder.gpu_reader import create_decoder

    reader = create_decoder(video_path, device)
    reader.open()

    tensors = {}
    # 按 batch 读取
    batch_size = 64
    for i in range(0, len(frame_indices), batch_size):
        batch_idx = frame_indices[i:i+batch_size]

        if hasattr(reader, 'get_frame_batch'):
            frames = reader.get_frame_batch(batch_idx)
            for fr in frames:
                tensors[fr.index] = fr.tensor
        else:
            # FFmpeg fallback: 逐帧
            for idx in batch_idx:
                for frame in reader:
                    if frame.index == idx:
                        tensors[idx] = frame.tensor
                        break

    reader.close()
    return tensors


def semantic_boundary_detection(embeddings, frame_indices, cos_thr=0.70):
    """
    Section 10: Semantic Boundary Detection。

    维护 scene centroid，逐帧判断是否切换场景。

    Args:
        embeddings: (N, 1024) np.ndarray, L2 normalized
        frame_indices: list of int (对应 embeddings 的帧号)
        cos_thr: cosine similarity 阈值

    Returns:
        boundaries: list of frame indices (scene start frames)
        scenes: list of (start_frame, end_frame, centroid)
    """
    boundaries = [frame_indices[0]]  # 第一个场景
    centroid = embeddings[0].copy()
    scene_start = 0

    for i in range(1, len(embeddings)):
        emb = embeddings[i]
        cos_sim = float(np.dot(centroid, emb))

        if cos_sim < cos_thr:
            # 场景切换
            boundaries.append(frame_indices[i])
            scene_start = i

        # 更新 centroid (EMA)
        centroid = 0.9 * centroid + 0.1 * emb
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

    return boundaries


def run_dino_selector(video_path, candidates_json, out_dir, device="cuda"):
    """
    主函数: 对候选帧做 DINO 语义边界检测。

    Args:
        video_path: 视频路径
        candidates_json: candidates.json 路径
        out_dir: 输出目录
        device: cuda / cpu
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. 加载候选
    with open(candidates_json) as f:
        data = json.load(f)

    candidates = data["candidates"]
    total_frames = data["total_frames"]
    fps = data.get("fps", 25.0)

    # 提取候选帧号 (包括 frame 0)
    frame_indices = sorted(set([0] + [c["frame"] for c in candidates]))
    print(f"[dino] {len(candidates)} candidates, {len(frame_indices)} unique frames")

    # 2. 加载帧 tensor (GPU)
    print(f"[dino] loading frame tensors ...")
    t0 = time.time()
    frame_tensors = load_frame_tensors(video_path, frame_indices, device)
    print(f"  loaded {len(frame_tensors)} frames ({time.time()-t0:.1f}s)")

    # 3. 按帧号排序的 tensor list
    sorted_tensors = [frame_tensors[i] for i in frame_indices if i in frame_tensors]

    # 4. DINO inference
    model = load_dinov2()
    print(f"[dino] computing embeddings ...")
    t0 = time.time()
    embeddings = compute_embeddings(model, sorted_tensors)
    print(f"  embeddings: {embeddings.shape} ({time.time()-t0:.1f}s)")

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    # 5. Semantic boundary detection
    print(f"[dino] detecting boundaries (cos_thr={DINO_COS_THR}) ...")
    boundaries = semantic_boundary_detection(embeddings, frame_indices, DINO_COS_THR)
    nscenes = len(boundaries) - 1
    print(f"  {len(boundaries)} boundaries → {nscenes} scenes")

    # 6. 输出
    result = {
        "video": os.path.basename(video_path),
        "method": "dino_semantic_boundary",
        "model": "dinov2_vitl14",
        "cos_threshold": DINO_COS_THR,
        "total_frames": total_frames,
        "fps": fps,
        "n_candidates": len(candidates),
        "n_boundaries": len(boundaries) - 1,
        "boundaries": boundaries[1:],  # 排除 frame 0
        "scenes": [
            {
                "scene_id": i,
                "start": boundaries[i],
                "end": boundaries[i+1] if i+1 < len(boundaries) else total_frames,
            }
            for i in range(len(boundaries) - 1)
        ],
    }

    out_path = os.path.join(out_dir, "scenes.json")
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  -> {out_path}")

    # 7. 打印场景
    for s in result["scenes"]:
        dur = (s["end"] - s["start"]) / fps
        print(f"  scene[{s['scene_id']:3d}]: fr{s['start']:5d}-{s['end']:5d}  {dur:.1f}s")

    # 8. 清理 GPU
    del model
    torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (os.path.splitext(args.video)[0] + "_v15mosugu")
    run_dino_selector(args.video, args.candidates, out_dir)
