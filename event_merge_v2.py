#!/usr/bin/env python3
"""V12-A Exp5: Event Merge v2 - self-contained chapter clustering.
Pure frame numbers. No hardcoded entities. No time dependency."""
import sys, os, json, math, re
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = "."



def cos_sim(a, b):
    # Vectors L2-normalized by Qwen3-Embedding
    return float(np.dot(a, b))


# Qwen3-Embedding-0.6B (lazy-loaded)
_emb_model = None
_vis_graph = None

def get_emb_model():
    global _emb_model
    if _emb_model is None:
        _emb_model = SentenceTransformer(
            "/home/dahe/models/hf/hub/Qwen/Qwen3-Embedding-0___6B/"
        )
    return _emb_model

def _get_vis_graph():
    global _vis_graph
    if _vis_graph is None:
        path = os.path.join(DATA_DIR, "dino_cluster", "shot_visual_graph.npy")
        if os.path.isfile(path):
            _vis_graph = np.load(path).astype(np.float32)
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


def extract_social_objects(asr_text, topic_hints=None):
    keywords = extract_asr_keywords(asr_text, top_k=8)
    if not keywords:
        return []
    try:
        text_emb = embed([asr_text])[0]
        kw_embs = embed(keywords)
        sims = kw_embs @ text_emb
        top_idx = np.argsort(sims)[::-1][:8]
        emb_top = {keywords[i] for i in top_idx if sims[i] > 0.1}
    except Exception:
        emb_top = set(keywords)
    candidates = list(dict.fromkeys([k for k in keywords if k in emb_top]))
    if topic_hints:
        ht = " ".join(topic_hints)
        candidates = [c for c in candidates if c in ht] or candidates
    return candidates[:5]


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
        "frame_range": (min(frame_starts), max(frame_ends)),
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

    # environment similarity (取代旧 location_type)
    env_a, env_b = set(ca["environments"]), set(cb["environments"])
    if env_a and env_b:
        scores["environment"] = len(env_a & env_b) / max(len(env_a | env_b), 1)
    else:
        scores["environment"] = 0.3

    # event similarity (取代旧 event_type)
    ev_a, ev_b = set(ca["events"]), set(cb["events"])
    if not ev_a and not ev_b: scores["event"] = 0.5
    elif not ev_a or not ev_b: scores["event"] = 0.3
    else: scores["event"] = len(ev_a & ev_b) / max(len(ev_a | ev_b), 1)

    # topic similarity (从 topic_hints 改为 topics)
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

    # ASR 权重降低（ASR 弱信号）
    if ca["asr_combined"] and cb["asr_combined"]:
        try:
            v = embed([ca["asr_combined"], cb["asr_combined"]])
            scores["asr"] = cos_sim(v[0], v[1])
        except Exception: scores["asr"] = 0.0
    else: scores["asr"] = 0.0

    # visual similarity（scene 间 shot 视觉相似度均值）
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

    # GPT 权重: topic 0.35 + visual 0.25 + characters 0.20 + environment 0.10 + ASR 0.10
    total = 0.35 * scores.get("topic", 0) + 0.25 * scores["visual"] \
          + 0.20 * scores["characters"] + 0.10 * scores.get("environment", 0) \
          + 0.10 * scores.get("asr", 0)
    return total, scores


def hac_merge(cids, cluster_profiles, threshold=0.25):
    n = len(cids)
    dist = np.ones((n, n))
    for i, ci in enumerate(cids):
        for j, cj in enumerate(cids):
            if i >= j: continue
            score, _ = compute_merge_score(cluster_profiles[ci], cluster_profiles[cj])
            dist[i, j] = dist[j, i] = 1.0 - score
    np.fill_diagonal(dist, 0)
    groups = [{cids[i]} for i in range(n)]
    while len(groups) > 1:
        best_i = best_j = -1; best_d = float("inf")
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                ds = [dist[cids.index(ci), cids.index(cj)] for ci in groups[i] for cj in groups[j]]
                avg_d = float(np.mean(ds))
                if avg_d < best_d: best_d, best_i, best_j = avg_d, i, j
        if best_d > 1.0 - threshold: break
        groups[best_i] |= groups[best_j]; groups.pop(best_j)
    return groups


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id", nargs="?", default="japanese_street_girls")
    parser.add_argument("--type", default="street")
    parser.add_argument("--merge-threshold", type=float, default=0.25)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    vid, vtype = args.video_id, args.type
    clusters_path = "%s/%s/%s.event_clusters.json" % (DATA_DIR, vtype, vid)
    profiles_path = "%s/%s/%s.event_profiles.json" % (DATA_DIR, vtype, vid)
    print("Loading %s..." % vid)
    labels = json.load(open(clusters_path)).get("labels", {})
    profiles = json.load(open(profiles_path))
    cluster_scenes = {}
    for sid, lbl in labels.items():
        cluster_scenes.setdefault(str(lbl), []).append(sid)
    cluster_profiles = {}
    for lbl in sorted([k for k in cluster_scenes if k != "-1"], key=lambda x: int(x)):
        sids = cluster_scenes[lbl]
        cp = build_cluster_profile(sids, profiles)
        cluster_profiles[lbl] = cp
        so = " ".join(cp["social_objects"][:3]) if cp["social_objects"] else "(ASR:%s)" % " ".join(cp["asr_entities"][:2])
        print("  %s (%2dsc) 人物=%-8s 对象=%s" % (lbl, len(sids), (cp["person_types"] or ["?"])[0], so))
    cids = sorted(cluster_profiles.keys())
    groups = hac_merge(cids, cluster_profiles, threshold=args.merge_threshold)
    print("\n\n== Event Candidates (HAC merge, th=%.2f) ==" % args.merge_threshold)
    events = []
    for gi, group in enumerate(groups):
        all_sids = sorted([sid for ci in group for sid in cluster_scenes.get(ci, [])],
            key=lambda s: profiles.get(s, {}).get("start_frame", 0))
        if not all_sids: continue
        st, et = profiles[all_sids[0]]["start_frame"], profiles[all_sids[-1]]["end_frame"]
        cp = build_cluster_profile(all_sids, profiles)
        entities = cp["social_objects"][:3] or cp["asr_entities"][:2]
        persons = cp["person_types"][:3]
        actions = cp["action_types"][:3]
        event_name = entities[0] if entities else cp["topic_hints"][0] if cp["topic_hints"] else "unnamed"
        narrative = "%s在%s,%s" % (
            "、".join(persons) if persons else "crowd",
            "、".join(cp["location_types"][:2]) if cp["location_types"] else "outdoor",
            "、".join(actions) if actions else "activity")
        if entities: narrative += "，围绕" + "、".join(entities)
        if cp["topic_hints"]: narrative += "，" + cp["topic_hints"][0]
        events.append({"event_id": gi + 1, "name": event_name,
            "frame_range": (int(st), int(et)), "n_scenes": len(all_sids),
            "from_clusters": sorted([int(c) for c in group]), "scenes": all_sids,
            "profile": {"entities": entities, "persons": persons, "actions": actions,
                "locations": cp["location_types"][:2], "topics": cp["topic_hints"][:3],
                "asr_keywords": cp["asr_keywords"][:6]},
            "narrative": narrative.strip()})
        print("  E%02d %-16s [f%5d-f%5d] %2d scenes %s" % (
            gi + 1, event_name[:14], st, et, len(all_sids), "clusters=%s" % [int(c) for c in group]))
        print("       %s" % narrative[:85])
    print("\n  Input: %d clusters -> Output: %d events" % (len(cids), len(events)))
    out_path = args.output or "%s/%s/%s.event_candidates_v2.json" % (DATA_DIR, vtype, vid)
    with open(out_path, "w") as f:
        json.dump({"video": vid, "n_clusters_input": len(cids), "n_events_output": len(events),
                   "merge_threshold": args.merge_threshold, "event_candidates": events},
                  f, ensure_ascii=False, indent=2)
    print("Saved: %s" % out_path)


if __name__ == "__main__":
    main()
