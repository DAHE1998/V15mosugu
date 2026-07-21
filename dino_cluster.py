#!/usr/bin/env python3
"""04 — DINO 视觉 shot 图: key_frames → DINOv2 → per-shot embedding + N×N graph。

输入:  02_select_frames/skeleton.json (shots[] 含 key_frames + representative_frame)
输出:  dino_cluster/skeleton.json
       dino_cluster/shot_embedding_mean.npy   — (N_shots, 1024) shot 级 mean-pooled embedding
       dino_cluster/shot_visual_graph.npy      — (N_shots, N_shots) 余弦相似度矩阵
       dino_cluster/key_frame_embeddings.npz   — key_frame 级 embedding + 映射关系
"""
import json, sys, os, subprocess as sp
import numpy as np, torch, torch.nn.functional as F

output = sys.argv[1]
in_path = os.path.join(output, "02_select_frames", "skeleton.json")
with open(in_path) as f:
    skeleton = json.load(f)

out_dir = os.path.join(output, "dino_cluster")
os.makedirs(out_dir, exist_ok=True)

shots = skeleton["shots"]
video = skeleton["video"]
print(f"[04] DINO 视觉 shot 图: {len(shots)} shots")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ── 收集所有 shot 的 key_frames ──
kf_per_shot = []
kf_all = []
for s in shots:
    kf = s.get("key_frames")
    if not kf:
        # fallback: 用 representative_frame
        rf = s.get("representative_frame")
        if rf is None:
            rf = (s["range"]["start"] + s["range"]["end"]) // 2
        kf = [int(rf)] if isinstance(rf, (int, float, np.integer)) else [int(rf[0])]
    kf = [int(f) for f in kf]
    kf_per_shot.append(kf)
    kf_all.extend(kf)

all_f = sorted(set(kf_all))
f2idx = {fn: i for i, fn in enumerate(all_f)}
n_all = len(all_f)
n_multi = sum(1 for kf in kf_per_shot if len(kf) > 1)
print(f"  {n_all} key_frames ({len(shots)} shots, {n_multi} multi-frame)")

# ── FFmpeg GPU 分批抽帧 ──
S = 224
BATCH_SIZE = 50
batches = [all_f[i:i + BATCH_SIZE] for i in range(0, n_all, BATCH_SIZE)]
raw_parts = []
for batch in batches:
    sel = "+".join(["eq(n\\,%d)" % f for f in batch])
    proc = sp.Popen([
        "ffmpeg", "-hwaccel", "cuda", "-loglevel", "error", "-i", video,
        "-vf", "select=" + sel + ",scale=%d:%d" % (S, S),
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ], stdout=sp.PIPE, stderr=sp.PIPE, bufsize=512 * 1024 * 1024)
    part = proc.stdout.read()
    proc.wait()
    raw_parts.append(part)
raw = b"".join(raw_parts)

exp = n_all * S * S * 3
if len(raw) < exp:
    print(f"  ERROR: expected {exp} bytes, got {len(raw)}")
    sys.exit(1)
arr = np.frombuffer(raw[:exp], dtype=np.uint8).reshape(n_all, S, S, 3)
del raw, raw_parts

# ── GPU 预处理 ──
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
BS = 512
ts = []
for b in range(0, n_all, BS):
    ch = arr[b:b + BS]
    t = torch.from_numpy(ch).to("cuda", dtype=torch.float16, non_blocking=True)
    t = t.permute(0, 3, 1, 2).contiguous()
    t = F.interpolate(t, size=256, mode="bicubic", align_corners=False).clamp(0, 255)
    hs = (256 - 224) // 2
    t = t[:, :, hs:hs + 224, hs:hs + 224]
    t = t / 255.0
    t = (t - MEAN) / STD
    ts.append(t)
all_t = torch.cat(ts)
del arr

# ── DINOv2 ──
print(f"  loading DINOv2...")
m0 = torch.cuda.memory_allocated() / 1024 ** 3
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
dinov2 = dinov2.to("cuda").eval().half()
print(f"  GPU: {m0:.1f} -> {torch.cuda.memory_allocated() / 1024 ** 3:.1f}GB")

BS2 = 256
embs = []
for b in range(0, n_all, BS2):
    with torch.inference_mode():
        embs.append(dinov2(all_t[b:b + BS2]))
all_e = torch.cat(embs)
del all_t
torch.cuda.empty_cache()

# ── 保存 key_frame 级 embedding（供 4.5 D50 去重）──
kf_emb = all_e.cpu().numpy().astype(np.float16)
np.savez(os.path.join(out_dir, "key_frame_embeddings.npz"),
         embeddings=kf_emb,
         frame_ids=np.array(all_f, dtype=np.int32))
print(f"  saved key_frame embeddings: {kf_emb.shape}")

# ── Per-shot mean pooling ──
shot_emb = np.zeros((len(shots), all_e.shape[1]), dtype=np.float16)
for i, kf_list in enumerate(kf_per_shot):
    indices = [f2idx[f] for f in kf_list]
    if len(indices) == 1:
        shot_emb[i] = all_e[indices[0]].cpu().numpy().astype(np.float16)
    else:
        shot_emb[i] = all_e[indices].mean(dim=0).cpu().numpy().astype(np.float16)

np.save(os.path.join(out_dir, "shot_embedding_mean.npy"), shot_emb)
print(f"  saved shot embedding (mean): {shot_emb.shape}")

# ── N×N visual graph ──
normed = F.normalize(torch.from_numpy(shot_emb.astype(np.float32)), dim=1)
graph = (normed @ normed.T).cpu().numpy()
np.save(os.path.join(out_dir, "shot_visual_graph.npy"), graph)
print(f"  saved visual graph: {graph.shape}")

# ── 相邻 shot 余弦相似度（兼容下游 visual_cluster 字段） ──
shot_vecs = F.normalize(torch.from_numpy(shot_emb.astype(np.float32)).to("cuda"), dim=-1)
boundaries = []
for i in range(len(shots) - 1):
    c = float((shot_vecs[i] * shot_vecs[i + 1]).sum().item())
    boundaries.append({"i": i, "score": round(max(0.0, c), 4)})

ss = [b["score"] for b in boundaries]
print(f"  {len(boundaries)} boundaries  {min(ss):.4f} ~ {max(ss):.4f}")

# ── 填入骨架 ──
for b in boundaries:
    idx = b["i"]
    shots[idx]["visual_cluster"] = {"next_similarity": b["score"]}
if len(shots) > 0:
    shots[-1]["visual_cluster"] = {"next_similarity": None}

skeleton["shots"] = shots
skel_path = os.path.join(out_dir, "skeleton.json")
with open(skel_path, "w") as f:
    json.dump(skeleton, f, ensure_ascii=False, indent=2)

ref_path = os.path.join(out_dir, "dino_cluster.json")
with open(ref_path, "w") as f:
    json.dump({"method": "dino_keyframes_mean_pool", "boundaries": boundaries},
              f, ensure_ascii=False, indent=2)

print(f"  -> {out_dir}/")
