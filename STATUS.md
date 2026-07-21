# V15mosugu 现状 (2026-07-16)

## 已有成果

### 骨架架构 ✅
- 统一数据结构：`skeleton.json`
- `{video, total_frames, duration, fps, width, height, scenes: [{scene, start_frame, end_frame, ...}]}`
- 各阶段向骨架写入新字段，不破坏已有字段
- 已改写文件：`pipeline.py`, `asr.py`, `vlm_profile.py`, `run.sh`

### ASR：SenseVoiceSmall ✅
- 模型：`/home/dahe/.cache/modelscope/models/iic/SenseVoiceSmall` (~234M, ~2GB VRAM)
- 逐场景转录，帧精确映射
- RTF ~0.003，111 场景 ≈ 30 秒
- 中日混合 auto-detect 正常
- 句号裁切：保留到最后一个 `。`，丢弃尾部碎片
- 实测：japanese_street_girls (13min) → 110 场景 → 每场景有 text

### 文本聚类：Qwen3-Embedding-0.6B ✅
- 模型：`/home/dahe/models/hf/hub/Qwen/Qwen3-Embedding-0.6B`
- 相邻场景余弦相似度：0.33 ~ 0.62
- 阈值 0.45 左右可区分话题切换
- 实测：sc_002→003 (0.33, 乱码) vs sc_007→008 (0.62, 同话题老人)

## 当前问题

### Pipeline.py DINO 合并逻辑
- 原始 206 行 pipeline.py 已被覆盖，无备份
- 已根据 README 算法重写（165 行），含完整 scdet→heartbeat→DINO embedding→cosine merge→cleanup→re-insert
- **未实际跑过验证**，可能精度不如原版

### 关键帧来源：D50 待定 ⚠️
- v14_d50.py 依赖外部 `.embeddings.npz`（全帧 2fps DINO，预生成）
- 现状：无独立生成 embeddings.npz 的步骤
- 需要设计无外部依赖的帧来源方案

#### 已讨论的方案：

| 方案 | 思路 | 问题 |
|------|------|------|
| 均匀采样 | 每场景固定 K 帧，按帧间隔 | 采样帧不一定有代表性 |
| scdet 切点 | 用场景内的 scdet 切点做 keyframe | 切点是边界，不一定是核心帧 |
| 场景内 DINO 抽帧 | 每场景内抽 K 帧→DINO→HDBSCAN 聚类→取质心帧 | 需内置 DINO 抽帧逻辑，不依赖外部 npz |

#### D50 核心逻辑（v14_d50.py）：
```python
# 输入：emb (N,1024), frame_nums (N,), st_frame, et_frame
# 1. 按帧范围截取 embedding
# 2. HDBSCAN 聚类（min_cluster_size=2, metric=euclidean）
# 3. 每簇取最接近质心的帧
# 4. 按簇大小排序，最多取 6 帧
```
当前 DINO 模型（ViT-L/14）已在 pipeline 时加载，可用于场景内帧嵌入。

### VLM Profile
- 已改写为批量推理（BATCH=4），读骨架→抽帧→Qwen3-VL→写入骨架
- **依赖 D50 关键帧，尚未实际测试**

## 服务器环境

| 项 | 值 |
|----|-----|
| 主机 | amaterasu (192.168.0.44) |
| GPU | RTX 4070 12GB |
| Python | 3.10, conda env=amaterasu |
| transformers | 5.12.1 |
| funasr | 1.3.14 |
| faster-whisper | 1.2.1 |

## 模型位置

| 模型 | 路径 |
|------|------|
| SenseVoiceSmall | /home/dahe/.cache/modelscope/models/iic/SenseVoiceSmall |
| Qwen3-Embedding-0.6B | /home/dahe/models/hf/hub/Qwen/Qwen3-Embedding-0.6B |
| Qwen3-VL-8B | /home/dahe/models/hf/hub/Qwen/Qwen3-VL-8B-Instruct |
| Qwen3-ASR-1.7B | /home/dahe/.cache/modelscope/models/Qwen--Qwen3-ASR-1.7B/snapshots/master |
| Qwen3-ASR-0.6B | /home/dahe/models/hf/hub/Qwen/Qwen3-ASR-0.6B |
| DINOv2 ViT-L/14 | torch.hub 自动下载 (~/.cache/torch/hub) |

## 待讨论

1. D50 帧来源：pipeline 内 DINO 抽帧 vs 均匀采样 vs scdet 切点
2. Pipeline.py 重写版是否需要跑验证
3. chapters.py 聚类策略
