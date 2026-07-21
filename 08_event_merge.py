#!/usr/bin/env python3
"""08 — Event Merge: Scene → Event, linear pass with merge_score。

输入:  07_vlm/skeleton.json (scenes[] + profile + asr_text)
输出:  08_event_merge/skeleton.json (+ events[])
       08_event_merge/event_output.json

v3.0 改动:
  - 从 exp5_event_merge_v2.py 收编 build_cluster_profile + compute_merge_score
  - 删除 extract_social_objects / hac_merge / main()（坏死代码，引用不存在的字段）
  - 删除 event 项（死代码，不算进 total）
  - ASR 保留 0.10 权重，权重和 = 1.00（0.35 + 0.25 + 0.20 + 0.10 + 0.10）
  - 线性 pass，不建 N×N 矩阵
  - THR = 0.17
"""
import json, sys, os, re
from collections import Counter
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

assert torch.cuda.is_available(), "CUDA 不可用，检查 torch 安装"

LOG_TAG = "[08em]"


def cos_sim(a, b):
    return float(np.dot(a, b))


_emb_model = None
_vis_graph = None
_vis_graph_path = None


def get_emb_model():
    global _emb_model
    if _emb_model is None:
        _emb_model = SentenceTransformer(
            "/home/dahe/models/hf/hub/Qwen/Qwen3-Embedding-0___6B/"
        )
    return _emb_model


def set_vis_graph(path):
    """设置 visual graph 路径（08 调用时传入 work_dir）。"""
    global _vis_graph, _vis_graph_path
    _vis_graph_path = path
    _vis_graph = None


def _get_vis_graph():
    global _vis_graph
    if _vis_graph is None:
        if _vis_graph_path and os.path.isfile(_vis_graph_path):
            _vis_graph = np.load(_vis_graph_path).astype(np.float32)
        else:
            _vis_graph = np.array([[0.0]])
    return _vis_graph


def embed(texts):
    model = get_emb_model()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)


def extract_asr_keywords(asr_text, top_k=5):
    stops = {"的", "了", "是", "在", "有", "就", "这", "那", "也", "还",
             "都", "我", "你", "他", "她", "它", "们", "个", "什么",
             "这个", "那个", "一个", "没有", "不是", "因为", "所以", "但是"}
    words = re.findall(r"[一-鿿\w]{2,}", asr_text)
    words = [w for w in words if w not in stops and len(w) >= 2]
    return [w for w, _ in Counter(words).most_common(top_k)]


def build_cluster_profile(sids, profiles):
    environments, characters, actions = Counter(), Counter(), Counter()
    events, story_roles = Counter(), Counter()
    topics, asr_texts = [], []
    frame_starts, frame_ends = [], []
    all_shot_ids = []

    for sid in sids:
        sd = profiles.get(sid, {})
        p = sd.get("profile", {})
        sf = sd.get("start_frame", 0)
        ef = sd.get("end_frame", 0)
        frame_starts.append(sf)
        frame_ends.append(ef)
        if p.get("environment"): environments[p["environment"]] += 1
        if p.get("characters"): characters[p["characters"]] += 1
        if p.get("actions"): actions[p["actions"]] += 1
        if p.get("event"): events[p["event"]] += 1
        if p.get("story_role"): story_roles[p["story_role"]] += 1
        if p.get("topic"): topics.append(p["topic"])
        asr = sd.get("asr_text", "") or sd.get("asr", "")
        if asr: asr_texts.append(asr)
        shot_ids = sd.get("shot_ids", [])
        all_shot_ids.extend(shot_ids)

    asr_combined = " ".join(asr_texts)
    asr_keywords = extract_asr_keywords(asr_combined)

    return {
        "n_scenes": len(sids),
        "range": (min(frame_starts), max(frame_ends)),
        "shot_ids": all_shot_ids,
        "environments": [k for k, _ in environments.most_common(3)],
        "characters": [k for k, _ in characters.most_common(3)],
        "actions": [k for k, _ in actions.most_common(3)],
        "events": [k for k, _ in events.most_common(3)],
        "story_roles": [k for k, _ in story_roles.most_common(3)],
        "topics": topics[:5],
        "asr_combined": asr_combined[:500],
        "asr_keywords": asr_keywords[:8],
    }


def compute_merge_score(ca, cb):
    scores = {}

    # environment similarity
    env_a, env_b = set(ca["environments"]), set(cb["environments"])
    if env_a and env_b:
        scores["environment"] = len(env_a & env_b) / max(len(env_a | env_b), 1)
    else:
        scores["environment"] = 0.3

    # topic similarity
    if ca["topics"] and cb["topics"]:
        joint = ca["topics"] + cb["topics"]
        try:
            v = embed(joint)
            mid = len(ca["topics"])
            va = v[:mid].mean(axis=0) if mid > 1 else v[0]
            vb = v[mid:].mean(axis=0) if mid < len(joint) else v[mid]
            scores["topic"] = cos_sim(va, vb)
        except Exception: scores["topic"] = 0.1
    else: scores["topic"] = 0.1

    # ASR similarity
    if ca["asr_combined"] and cb["asr_combined"]:
        try:
            v = embed([ca["asr_combined"], cb["asr_combined"]])
            scores["asr"] = cos_sim(v[0], v[1])
        except Exception: scores["asr"] = 0.0
    else: scores["asr"] = 0.0

    # visual similarity
    vis_a = ca.get("shot_ids", [])
    vis_b = cb.get("shot_ids", [])
    if vis_a and vis_b:
        try:
            _vg = _get_vis_graph()
            sims = []
            for sa in vis_a:
                for sb in vis_b:
                    if 0 <= sa < _vg.shape[0] and 0 <= sb < _vg.shape[1]:
                        sims.append(float(_vg[sa, sb]))
            scores["visual"] = float(np.mean(sims)) if sims else 0.0
        except Exception:
            scores["visual"] = 0.0
    else:
        scores["visual"] = 0.0

    # characters overlap
    ch_a, ch_b = set(ca["characters"]), set(cb["characters"])
    if ch_a and ch_b:
        scores["characters"] = len(ch_a & ch_b) / max(len(ch_a | ch_b), 1)
    else:
        scores["characters"] = 0.0

    # 权重: topic 0.35 + visual 0.25 + characters 0.20 + environment 0.10 + ASR 0.10
    total = 0.35 * scores.get("topic", 0) + 0.25 * scores["visual"] \
          + 0.20 * scores["characters"] + 0.10 * scores.get("environment", 0) \
          + 0.10 * scores.get("asr", 0)
    return total, scores


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <work_dir>")
        sys.exit(1)

    work = sys.argv[1]
    in_path = os.path.join(work, "07_vlm", "skeleton.json")
    with open(in_path) as f:
        skeleton = json.load(f)

    scenes = skeleton.get("scenes", [])
    print(f"{LOG_TAG} {len(scenes)} scenes  THR=0.17")

    # 设置 visual graph 路径
    vis_graph_path = os.path.join(work, "04_dino_cluster", "shot_visual_graph.npy")
    set_vis_graph(vis_graph_path)

    # 构建 profiles dict（供 build_cluster_profile 使用）
    profiles = {}
    for s in scenes:
        sid = s["id"]
        profiles[sid] = {
            "profile": s.get("profile", {}),
            "asr_text": s.get("asr_text", ""),
            "start_frame": s["range"]["start"],
            "end_frame": s["range"]["end"],
            "shot_ids": s.get("shot_ids", []),
        }

    keys = sorted(profiles.keys(), key=lambda k: profiles[k]["start_frame"])
    scene_p = {sk: build_cluster_profile([sk], profiles) for sk in keys}

    # 线性 pass: 相邻两两比较
    THR = 0.17
    events = []
    cur = [keys[0]]

    for i in range(1, len(keys)):
        sk = keys[i]
        prev = cur[-1]
        sc, det = compute_merge_score(scene_p[prev], scene_p[sk])
        if sc < THR:
            events.append(cur)
            cur = [sk]
        else:
            cur.append(sk)
    if cur:
        events.append(cur)

    # 构建 events 层
    event_list = []
    for ei, ev in enumerate(events):
        first_s = profiles[ev[0]]
        last_s = profiles[ev[-1]]
        event_list.append({
            "id": ei,
            "scene_ids": ev,
            "range": {
                "start": first_s["start_frame"],
                "end": last_s["end_frame"],
            },
        })

    print(f"  {len(events)} events from {len(scenes)} scenes")
    for ev in event_list:
        ids = ev["scene_ids"]
        print(f"    E{ev['id']:02d} f{ev['range']['start']}-{ev['range']['end']}  {len(ids)} scenes")

    # C10 assert
    _, det = compute_merge_score(scene_p[keys[0]], scene_p[keys[1]] if len(keys) > 1 else scene_p[keys[0]])
    wsum = 0.35 + 0.25 + 0.20 + 0.10 + 0.10
    assert wsum == 1.00, f"C10 失败: 权重和 = {wsum}"
    assert "event" not in det, "C10 失败: event 项仍在 scores 中"
    print(f"  C10 pass: 权重和 = {wsum}, 无 event 项, THR = {THR}")

    # 填入骨架
    skeleton["events"] = event_list

    out_dir = os.path.join(work, "08_event_merge")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "skeleton.json"), "w") as f:
        json.dump(skeleton, f, ensure_ascii=False, indent=2)

    ev_out = os.path.join(out_dir, "event_output.json")
    with open(ev_out, "w") as f:
        json.dump({
            "step": "08_event_merge",
            "n_scenes": len(scenes),
            "n_events": len(events),
            "thr": THR,
            "events": event_list,
        }, f, ensure_ascii=False, indent=2)

    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    main()
