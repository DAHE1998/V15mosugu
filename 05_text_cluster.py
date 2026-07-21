#!/usr/bin/env python3
"""05 — 文本聚类: Qwen3-Embedding shot 级上下文文本 → text_cluster + N×N graph。

每 shot 的文本 = 自身 ASR + 前后 15 秒上下文（避免时间切分导致的碎片）。

输入:  04_dino_cluster/skeleton.json (shots[] + asr_text + visual_cluster)
输出:  05_text_cluster/skeleton.json  (+ shot.text_cluster)
       05_text_cluster/text_cluster.json
"""
import json, sys, os, numpy as np, torch, torch.nn.functional as F
from sentence_transformers import SentenceTransformer

CONTEXT_SECONDS = 15  # 上下文窗口: 前后各 15 秒

output = sys.argv[1]
in_path = os.path.join(output, "04_dino_cluster", "skeleton.json")
with open(in_path) as f:
    skeleton = json.load(f)

out_dir = os.path.join(output, "05_text_cluster")
os.makedirs(out_dir, exist_ok=True)

shots = skeleton["shots"]
fps = skeleton["fps"]
print(f"[05] 文本聚类: {len(shots)} shots")

# ── 为每 shot 构建上下文文本 ──
contextual_texts = []
for i, s in enumerate(shots):
    shot_start_s = s["range"]["start"] / fps
    shot_end_s = (s["range"]["end"] + 1) / fps
    ctx_start = shot_start_s - CONTEXT_SECONDS
    ctx_end = shot_end_s + CONTEXT_SECONDS

    # 收集该时间窗口内的所有 shot 文本
    parts = []
    for j, sj in enumerate(shots):
        js = sj["range"]["start"] / fps
        je = (sj["range"]["end"] + 1) / fps
        # 如果 shot j 与上下文窗口重叠
        if js < ctx_end and je > ctx_start:
            t = (sj.get("asr_text") or "").strip()
            if t:
                parts.append(t)

    ctx_text = "。".join(parts) if parts else ""
    contextual_texts.append(ctx_text)

# 标记上下文文本是否有效
asr_valid = [len(t.strip()) > 1 for t in contextual_texts]
n_valid = sum(asr_valid)
print(f"  valid ASR (contextual): {n_valid}/{len(shots)}")

model = SentenceTransformer("/home/dahe/models/hf/hub/Qwen/Qwen3-Embedding-0___6B/")
vecs = model.encode(contextual_texts, convert_to_numpy=True, normalize_embeddings=True)

# 保存 contextual ASR text（供 07 VLM 使用）
for i in range(len(shots)):
    if not asr_valid[i]:
        vecs[i] = 0.0
    shots[i]["text_valid"] = bool(asr_valid[i])
    shots[i]["asr_contextual"] = contextual_texts[i]

boundaries = []
for i in range(len(shots) - 1):
    sc = float(vecs[i] @ vecs[i + 1]) if asr_valid[i] and asr_valid[i + 1] else 0.0
    boundaries.append({"i": i, "score": round(sc, 4)})

ss = [b["score"] for b in boundaries]
print(f"  {len(boundaries)} boundaries  {min(ss):.4f} ~ {max(ss):.4f}")


# 保存 per-shot text embedding
np.save(os.path.join(out_dir, "shot_text_embeddings.npy"), vecs.astype(np.float16))
print(f"  saved per-shot text embeddings: {vecs.shape}")

# ── N×N text graph ──
graph = vecs @ vecs.T
for i in range(len(shots)):
    if not asr_valid[i]:
        graph[i, :] = 0.0
        graph[:, i] = 0.0
np.save(os.path.join(out_dir, "shot_text_graph.npy"), graph)
print(f"  saved text graph: {graph.shape}")

# 填入骨架
for b in boundaries:
    idx = b["i"]
    shots[idx]["text_cluster"] = {"next_similarity": b["score"]}
if len(shots) > 0:
    shots[-1]["text_cluster"] = {"next_similarity": None}

skeleton["shots"] = shots
skel_path = os.path.join(out_dir, "skeleton.json")
with open(skel_path, "w") as f:
    json.dump(skeleton, f, ensure_ascii=False, indent=2)

ref_path = os.path.join(out_dir, "text_cluster.json")
with open(ref_path, "w") as f:
    json.dump({"method": "asr_text_context_window", "boundaries": boundaries},
              f, ensure_ascii=False, indent=2)

print(f"  -> {out_dir}/")
