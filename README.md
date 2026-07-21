# V15mosugu — 视频场景分割管线

自包含的视频场景分割系统。纯帧号，纯GPU。

## 架构

```
H.264视频
    │
    ├── 1. scdet (GPU) ───→ 硬切候选边界 (帧号)
    │
    ├── 2. Heartbeat ───→ 长段渐变检测 (20/40/60/80%采样 + DINO余弦)
    │
    ├── 3. GPU帧抽取 ───→ rawvideo pipe + Resize256+Crop224
    │
    ├── 4. DINO ViT-L/14 ───→ 段均值embedding (batch=32, FP16)
    │
    ├── 5. DINO余弦合并 ───→ 相邻段视觉连续性 (thr=0.5)
    │
    ├── 6. 短段清理 ───→ <200帧的段合并到邻居
    │
    ├── 7. scdet长段重插入 ───→ >730帧的段恢复scdet切点
    │
    └── 8. 输出 ───→ boundaries.txt (帧号)
```

## 文件结构

```
V15mosugu/
├── pipeline.py              # 视觉检测管线 (scdet→DINO→场景边界)
├── vlm_profile.py           # VLM结构化描述 (D50选帧 + Qwen3-VL)
├── chapters.py              # 章节聚类 (exp5权重融合 + compute_merge_score)
├── exp5_event_merge_v2.py   # V12-A 聚类引擎 (话题/实体/字段提取)
├── v14_d50.py               # 密度峰值选帧 (HDBSCAN子群落中心)
├── run.sh                   # 一键运行
└── README.md                # 本文档
```

## 环境依赖

### 硬件
- NVIDIA GPU (CUDA), 显存建议 ≥8GB
- NVIDIA Video Codec SDK (NVDEC/NVENC)

### 软件 (amaterasu环境)
- Python 3.10+
- PyTorch 2.x + CUDA
- FFmpeg 6.x (编译带cuda支持)
- torchvision (DINO预处理)
- transformers, bitsandbytes, qwen-vl-utils (VLM)
- sentence-transformers (Qwen3-Embedding)
- hdbscan (D50帧选择)
- sklearn (cosine similarity)
- jieba (exp5中文分词)
- PIL/Pillow
- numpy

### 模型文件
- DINOv2: `facebookresearch/dinov2` (torch.hub自动下载到~/.cache)
- Qwen3-VL-8B-Instruct: `/home/dahe/models/hf/hub/Qwen/Qwen3-VL-8B-Instruct/`
- Qwen3-Embedding-0.6B: `/home/dahe/models/hf/hub/Qwen/Qwen3-Embedding-0___6B/`

### 数据文件 (视频依赖)
- `<视频>.mp4` — H.264/H.265 视频文件
- `<视频>.embeddings.npz` — 2fps DINO embeddings (用于D50选帧)
- `<视频>.asr.json` — 语音识别结果 (用于章节聚类)

## 使用方法

### 一键运行
```bash
cd V15mosugu
bash run.sh /path/to/video.mp4
```

### 分步运行
```bash
# Step 1: 视觉场景检测
python3 pipeline.py /path/to/video.mp4

# Step 2: VLM场景描述 (需要 .embeddings.npz)
python3 vlm_profile.py <output_dir>/boundaries.txt video.mp4 <output_dir>/profiles.json

# Step 3: 章节聚类
python3 chapters.py <output_dir>/boundaries.txt <output_dir>/profiles.json <output_dir>/chapters.json
```

## 各模块详解

### pipeline.py — 视觉检测

**功能**: 从视频中检测镜头/场景切换边界

**流程**:
1. `ffprobe` 获取总帧数、分辨率
2. `ffmpeg -hwaccel cuda scdet=threshold=10` 检测硬切 (793fps)
3. 长段心跳: 无scdet边界超200帧的段 → 20/40/60/80%抽4帧 → DINO首尾余弦<0.70切分
4. 每段25/50/75%抽3帧 → rawvideo pipe → GPU tensor → DINO ViT-L/14 FP16 batch=32
5. 相邻段余弦 <0.5 → 切分; ≥0.5 → 合并
6. <200帧的段合并到邻居
7. >730帧的段恢复内部scdet切点

**参数**:
- `scdet threshold=10`: 场景切换灵敏度 (越大越保守)
- `BATCH=32`: DINO推理批次
- `HB_GAP=200`: 心跳触发阈值 (帧数)
- `MIN_SEG=200`: 短段合并阈值 (帧数)
- `LONG_SEG=730`: 长段scdet重插入阈值 (帧数)
- `HEARTBEAT_RATIOS=[0.20,0.40,0.60,0.80]`: 心跳采样位置

**输出**:
- `boundaries.txt`: 每行一个帧号，表示场景起始帧

### vlm_profile.py — VLM结构化描述

**功能**: 对每个场景生成7字段结构化描述

**流程**:
1. 加载2fps DINO embeddings
2. v14_d50密度峰值选帧 (HDBSCAN子群落质心)
3. GPU提取选中的帧 (rawvideo pipe, resize=448)
4. Qwen3-VL-8B 4-bit推理 → 7字段JSON

**Prompt字段**:
| 字段 | 选项 | 说明 |
|------|------|------|
| person_type | 少年/青年/中年/老年/群体 | 画面人物类型 |
| location_type | 街头/商业区/住宅区/室内/交通设施/其他 | 场景位置 |
| action_type | 采访/行走/聚集/休息/观察/展示/解说/其他 | 人物动作 |
| event_type | 人物采访/街景漫步/商业消费/交通中转/餐饮休息/室内活动/其他 | 场景事件 |
| social_object | 字符串 | 社会实体名 (如"忠犬八公雕像"), 无则留空 |
| narrative_role | 主持人解说/环境展示/字幕说明/动作记录/观点采访/案例人物/背景介绍/其他 | 叙事角色 |
| topic_hint | 字符串 (≤15字) | 主题提示 |

**输出**:
- `profiles.json`: `{scene_id: {start, end, n_frames, profile:{...7字段...}, asr}}`

### chapters.py — 章节聚类

**功能**: 用exp5权重公式对相邻场景做语义合并

**流程**:
1. 加载pipeline边界文件 (帧号)
2. 对每个场景调用 `build_cluster_profile` 提取: entities, keywords, topics, social_objects
3. 顺序遍历: 相邻场景用 `compute_merge_score` 算加权相似度
4. 总分 <0.15 → 保留章节边界; 总分 ≥0.15 → 合并
5. 去掉时间分量 (0.10权重)

**exp5 compute_merge_score 权重**:
| 信号 | 权重 | 说明 |
|------|------|------|
| social_object | 0.25 | 社会实体重叠 |
| asr_entity | 0.15 | ASR实体匹配 |
| asr | 0.15 | char n-gram TF-IDF文本相似 |
| topic | 0.15 | topic_hint相似度 |
| event_type | 0.15 | 事件类型重叠 |
| narrative_role | 0.05 | 叙事角色重叠 |
| time | 0.10 | **已被移除** |

**输出**:
- `chapters.json`: `{n_scenes, n_chapters, chapters: [{id, scenes}]}`
- 终端: 帧号 + 场景key + 7字段 + topic_hint

## 输出格式

### boundaries.txt
```
169
284
451
...
21016
```
每行一个帧号，按升序排列。视频总帧段 = [0, boundaries[0], ..., total_frames-1]

### profiles.json
```json
{
  "sc_000": {
    "start": 0.0,
    "end": 3.5,
    "n_frames": 5,
    "profile": {
      "person_type": "群体",
      "location_type": "商业区",
      "action_type": "行走",
      "event_type": "街景漫步",
      "social_object": "",
      "narrative_role": "环境展示",
      "topic_hint": "夜晚繁华商业街实景"
    },
    "asr": "白天的色谷大家应该都见过..."
  }
}
```

### chapters.json
```json
{
  "n_scenes": 70,
  "n_chapters": 29,
  "chapters": [
    {
      "id": "ch_000",
      "scenes": ["sc_000", "sc_001", "sc_002"]
    }
  ]
}
```

## 设计原则

1. **纯帧号**: 所有输入输出使用帧号，不使用秒数/时间戳
2. **纯GPU**: ffmpeg GPU解码 → GPU tensor → DINO GPU推理
3. **自包含**: 所有Python依赖在项目目录内，不引用外部项目路径
4. **无PNG**: rawvideo pipe直传，零磁盘IO
5. **帧号映射**: scene[i]直接对应pipeline边界[i]，无需时间换算

## 已知限制

1. D50帧选择需要视频的2fps DINO embeddings (.embeddings.npz)
2. Qwen3-VL路径硬编码，需根据环境修改
3. babydriver类快切视频建议调高MIN_SEG (当前200帧≈8s)
