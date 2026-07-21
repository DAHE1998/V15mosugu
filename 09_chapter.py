#!/usr/bin/env python3
"""09 — Chapter: Event → Chapter, linear pass with multi-dimensional similarity。

输入:  08_event_merge/skeleton.json (events[] + scenes[] with profile + asr_text)
输出:  09_chapter/skeleton.json (+ chapters[])
       09_chapter/chapter_output.json

v3.0 改动:
  - 从原 08_chapter.py 第二级逻辑拆出
  - 综合 VLM profile + ASR + 视觉信息多维度判断
  - 线性 pass，不建 N×N 矩阵
  - 具体阈值和权重待实测调优
"""
import json, sys, os
from collections import Counter
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

assert torch.cuda.is_available(), "CUDA 不可用，检查 torch 安装"

LOG_TAG = "[09ch]"


def cos_sim(a, b):
    return float(np.dot(a, b))


_emb_model = None


def get_emb_model():
    global _emb_model
    if _emb_model is None:
        _emb_model = SentenceTransformer(
            "/home/dahe/models/hf/hub/Qwen/Qwen3-Embedding-0___6B/"
        )
    return _emb_model


def embed(texts):
    model = get_emb_model()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)


def jaccard(a, b):
    """Jaccard 相似度（两个集合）。"""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.5
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def build_event_profile(scene_ids, scenes_by_id):
    """从 scene 列表构建 event 级聚合 profile。"""
    environments, characters = Counter(), Counter()
    story_roles, events = Counter(), Counter()
    topics, asr_texts, visual_summaries = [], [], []

    for sid in scene_ids:
        s = scenes_by_id.get(sid, {})
        p = s.get("profile", {})
        if p.get("environment"): environments[p["environment"]] += 1
        if p.get("characters"): characters[p["characters"]] += 1
        if p.get("story_role"): story_roles[p["story_role"]] += 1
        if p.get("event"): events[p["event"]] += 1
        if p.get("topic"): topics.append(p["topic"])
        if s.get("asr_text"):
            asr_texts.append(s["asr_text"])
        if p.get("visual_summary"):
            visual_summaries.append(p["visual_summary"])

    asr_combined = " ".join(asr_texts)
    vs_combined = " ".join(visual_summaries)

    return {
        "scene_ids": scene_ids,
        "environments": [k for k, _ in environments.most_common(3)],
        "characters": [k for k, _ in characters.most_common(3)],
        "story_roles": [k for k, _ in story_roles.most_common(3)],
        "events": [k for k, _ in events.most_common(3)],
        "topics": list(set(topics)),
        "asr_combined": asr_combined[:500],
        "visual_summaries": visual_summaries,
    }


def compute_chapter_similarity(ev_a, ev_b):
    """两个 event 的综合相似度（六维度加权）。"""
    scores = {}

    # 1. 话题一致性 (0.25): topic 列表 Jaccard
    scores["topic"] = jaccard(ev_a.get("topics", []), ev_b.get("topics", []))

    # 2. 场景连续性 (0.15): environment Jaccard
    scores["environment"] = jaccard(ev_a.get("environments", []), ev_b.get("environments", []))

    # 3. 人物连贯性 (0.15): characters Jaccard
    scores["characters"] = jaccard(ev_a.get("characters", []), ev_b.get("characters", []))

    # 4. 叙事逻辑 (0.15): story_role + event Jaccard 均值
    sr_sim = jaccard(ev_a.get("story_roles", []), ev_b.get("story_roles", []))
    ev_sim = jaccard(ev_a.get("events", []), ev_b.get("events", []))
    scores["narrative"] = (sr_sim + ev_sim) / 2

    # 5. 时间连续性 (0.10): 相邻事件默认为 1.0
    scores["time"] = 1.0

    # 6. 视觉连贯性 (0.20): visual_summary embedding cosine
    vs_a = ev_a.get("visual_summaries", [])
    vs_b = ev_b.get("visual_summaries", [])
    if vs_a and vs_b:
        try:
            v = embed([" ".join(vs_a), " ".join(vs_b)])
            scores["visual"] = cos_sim(v[0], v[1])
        except Exception:
            scores["visual"] = 0.0
    else:
        scores["visual"] = 0.0

    # ASR 话题补充 (0.00): 不计入总分，仅做辅助判断
    # ASR 文本通过 topic 和 visual_summary 已经间接参与

    total = (0.25 * scores["topic"] + 0.15 * scores["environment"]
             + 0.15 * scores["characters"] + 0.15 * scores["narrative"]
             + 0.10 * scores["time"] + 0.20 * scores["visual"])
    return total, scores


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <work_dir>")
        sys.exit(1)

    work = sys.argv[1]
    in_path = os.path.join(work, "08_event_merge", "skeleton.json")
    with open(in_path) as f:
        skeleton = json.load(f)

    events = skeleton.get("events", [])
    scenes = skeleton.get("scenes", [])
    scenes_by_id = {s["id"]: s for s in scenes}
    events_by_id = {ev["id"]: ev for ev in events}

    print(f"{LOG_TAG} {len(events)} events")

    # 构建 event 级 profile
    event_profiles = {}
    for ev in events:
        eid = ev["id"]
        event_profiles[eid] = build_event_profile(ev["scene_ids"], scenes_by_id)

    # 线性 pass: 相邻 event 两两比较
    THR = 0.40  # 待实测调优
    chapters = []
    cur = [events[0]["id"]]

    for i in range(1, len(events)):
        eid = events[i]["id"]
        prev_id = cur[-1]
        sc, det = compute_chapter_similarity(event_profiles[prev_id], event_profiles[eid])
        if sc < THR:
            chapters.append(cur)
            cur = [eid]
        else:
            cur.append(eid)
    if cur:
        chapters.append(cur)

    # 构建 chapters 层
    chapter_list = []
    for ci, ch_ids in enumerate(chapters):
        first_ev = events_by_id[ch_ids[0]]
        last_ev = events_by_id[ch_ids[-1]]
        all_scene_ids = []
        for eid in ch_ids:
            all_scene_ids.extend(event_profiles[eid]["scene_ids"])

        # 统计 dominant topic
        all_topics = []
        for eid in ch_ids:
            all_topics.extend(event_profiles[eid].get("topics", []))
        main_topic = max(set(all_topics), key=all_topics.count)[:30] if all_topics else ""

        chapter_list.append({
            "id": ci,
            "event_ids": ch_ids,
            "scene_ids": all_scene_ids,
            "range": {
                "start": first_ev["range"]["start"],
                "end": last_ev["range"]["end"],
            },
            "title": main_topic,
        })

    print(f"  {len(chapters)} chapters from {len(events)} events")
    for ch in chapter_list:
        ids = ch["event_ids"]
        print(f"    CH{ch['id']:02d} E{ids} f{ch['range']['start']}-{ch['range']['end']}  {ch['title']}")

    # 填入骨架
    skeleton["chapters"] = chapter_list

    out_dir = os.path.join(work, "09_chapter")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "skeleton.json"), "w") as f:
        json.dump(skeleton, f, ensure_ascii=False, indent=2)

    ch_out = os.path.join(out_dir, "chapter_output.json")
    with open(ch_out, "w") as f:
        json.dump({
            "step": "09_chapter",
            "n_events": len(events),
            "n_chapters": len(chapters),
            "thr": THR,
            "chapters": chapter_list,
        }, f, ensure_ascii=False, indent=2)

    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    main()
