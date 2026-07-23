# V15mosugu 模块文档

## 流水线总览

```
face_continuity ──┐
                  ├──> v01_visual_group ──> v02_person_label ──> v03_fragment_clean
DINO cluster ─────┘
```

---

## face_continuity.py — 人脸检测与人物追踪

**定位**：独立人脸模块，构建人物链（person chains），不参与任何镜头合并逻辑。

### 输入
| 文件 | 来源 |
|------|------|
| `d50_dedup/skeleton.json` | D50 去重后的骨架（含 `shots[].key_frames`） |
| 视频文件 | `sys.argv[2]` |

### 输出 (`face_continuity/`)
| 文件 | 内容 |
|------|------|
| `face_data.json` | 逐帧人脸数据：embedding、面积、置信度、每帧人脸数 |
| `person_chains.json` | 人物链：`person_id` → `[shot_id, ...]`，含 `n_shots` |
| `match_log.json` | 每张脸的匹配结果 |

### 核心参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `MIN_DET_SCORE` | 0.3 | 人脸检测最低置信度 |
| `MIN_FACE_AREA` | 5000 | 最小人脸面积（过滤广告牌小脸） |
| `DBSCAN_EPS` | 0.55 | 1-cos 距离阈值（cos≥0.45 归为同人） |
| `DBSCAN_MIN_SAMPLES` | 2 | 至少2张脸才能成簇 |

### 算法流程
1. **GPU帧提取** — ffmpeg CUDA pipe 全片解码 960×540，只存 `key_frames` 帧
2. **InsightFace检测** — SCRFD-10G + ArcFace w600k_r50，取面积 top-5
3. **Shot内去重** — 同 shot 内 embedding dot>0.99 视为重复脸，保留面积最大的
4. **DBSCAN图聚类** — M×M 余弦距离矩阵 → DBSCAN precomputed 模式，solo 脸（label=-1）也保留为独立 track
5. **输出person_chains** — 去重后的 `person_id → [shot_ids]`

### 设计决策
- **不用 representative_frame**：遍历 shot 的**所有** `key_frames`，确保不会漏掉只在某一帧出现的人物
- **按面积选主脸**：bbox 面积最大 = 镜头主体，不用置信度（广告牌人脸置信度也很高）
- **DBSCAN非在线追踪**：侧脸和正脸通过密度连通（embedding 图上的 transitive closure），不像在线 centroid 方法那样被时间顺序割裂

---

## v01_visual_group.py — 相邻视觉分组（含人物边界）

**定位**：Step 1，把 DINO 视觉相似 + 人物构成不变的相邻 shot 合并成 proto-scene。

### 输入
| 文件 | 用途 |
|------|------|
| `d50_dedup/skeleton.json` | 优先；不存在则回退 `dino_cluster/skeleton.json` |
| `dino_cluster/shot_visual_graph.npy` | N×N 余弦矩阵 |
| `face_continuity/person_chains.json` | 人物链（可选，无此文件时纯 DINO） |

### 输出 (`v01_visual_group/`)
- `skeleton.json` — 添加了 `proto_scenes` 字段的骨架

### 核心参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VIS_THR` | 0.45 | DINO 余弦阈值（Japan: 0.35, babydriver: 0.20） |

### 算法流程
1. 每个 shot 初始为独立组 `[[0], [1], ..., [N-1]]`
2. **反复合并**直到稳定：
   - 取相邻组 `g` 和 `nxt`
   - `dino_ok`: `graph[g[-1]][nxt[0]] >= thr`（组尾 vs 组首的余弦相似度）
   - `person_ok`: `same_scene(g, nxt)` 检查人物构成
   - 两者都通过才合并
3. **`same_scene()` 人物边界检测**：
   ```python
   gp = shot_persons.get(g[-1])  # 前组最后一个shot的人物集合
   np = shot_persons.get(nxt[0])  # 后组第一个shot的人物集合
   if not gp or not np: return True  # 任一边没脸 → DINO说了算
   return gp == np  # 人物集合完全相同才合并
   ```
4. 输出 proto_scenes（`id`, `shot_ids[]`, `range`, `duration_s`）

### 设计决策
- **人物进出 = 硬边界**：P8退场P37登场 → 即使 DINO 相似度很高也不合并
- **人物集合必须完全相同**（`gp == np`）：部分重叠也不行
- **无脸场景不拦**：任一边没人脸时退回到纯 DINO，避免误杀

---

## v02_person_label.py — 主导人物标注

**定位**：Step 2，给 v01 的视觉组标注主导人物（dominant person），基于局部频率×全局特异性。

### 输入
| 文件 | 来源 |
|------|------|
| `v01_visual_group/skeleton.json` | v01 输出（默认）或任意骨架 |
| `face_continuity/person_chains.json` | 人物链 |

### 输出 (`v02_person_label/`)
- `skeleton.json` — `proto_scenes` 每项增加 `"persons": [Pid]` 字段

### 核心公式
```
主导得分 = (c/n) × 0.7 + (1/max(total_shots_of_person, 1)) × 0.3
```
- `c/n`：该人物在本组内出现次数 / 组总 shot 数（局部频率）
- `1/total_shots`：人物越稀有→权重越高（全局特异性）

### 算法流程
1. 统计每组所有 shot 的人物出现次数
2. **`len(cnt) >= 2`** 才标注人物场景（筛选条件：组内至少要出现≥2种不同人物）
3. 按公式选最高分人物为 `dominant`
4. **相邻同人物合并**：`dom == prev_dom` → 合并为一个场景
5. **首尾修剪**：切掉主导人物不在的前导/后随 shot，独立成无标签片段

### 设计决策
- **特异性权重替代 ubiquitous 过滤**：旁白 P25 出现频繁 → `total_shots` 大 → 得分低；稀有主角 → `total_shots` 小 → 得分高。一个公式同时适配纪录片（Japan）和叙事片（babydriver）
- **`len(cnt) >= 2` 门控**：单人场景不算人物场景（旁白单人不标注，避免每段都标成旁白）

---

## v03_fragment_clean.py — 碎片合并清理

**定位**：Step 3，把 <2s 的连续短碎片合并，长片段和人物场景保持独立。

### 输入
| 文件 | 来源 |
|------|------|
| `v02_person_label/skeleton.json` | v02 输出（默认） |

### 输出 (`v03_fragment_clean/`)
- `skeleton.json` — 碎片合并后的最终 `proto_scenes`

### 核心参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `SHORT_S` | 2.0s | 短片段阈值 |

### 算法流程
1. 从左到右单次扫描
2. `duration < 2s` → 放入 `buf` 缓冲区
3. 遇到 `≥2s` 或有 `persons` 标签的场景 → **flush 缓冲区**（`_merge_buf`），再输出当前场景
4. 尾部残留 buffer → flush
5. `_merge_buf()` 把所有缓冲的短碎片合并成一个无标签片段

### 设计决策
- **有人物的短场景不保留**：即使 <2s，如果有人物标签也直接输出（不缓冲）。这是为了防止把有意义的人物镜头合并进碎片堆
- **实际上 v02 已合并且修剪过**，v03 主要处理 v02 修剪首尾产生的无标签碎片

---

## 典型视频适配

| 视频 | v01 VIS_THR | 特点 |
|------|-------------|------|
| Japan | 0.35 | 采访纪录片，画面变化慢 |
| babydriver | 0.20 | 动作片，画面切换快 |
| 默认 | 0.45 | 通用 |
