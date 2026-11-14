"""Leave-one-out accuracy on the enrolled samples. Measured, never estimated."""
from __future__ import annotations

import asyncio
import time
from collections import Counter, defaultdict

from .classifier import Config, Recogniser


def evaluate_user(samples: list[dict], cfg: Config, with_gemini: bool = False) -> dict | None:
    counts = Counter(s["phrase"] for s in samples)
    # A phrase needs >= 2 samples so that one can be held out and still be matchable.
    usable = [s for s in samples if counts[s["phrase"]] >= 2]
    phrases = sorted({s["phrase"] for s in usable})
    if len(phrases) < 2:
        return None
    rec = Recogniser(usable, cfg)
    n = top1 = top3 = 0
    g_top1 = g_top3 = 0
    g_answered = g_changed = 0  # how many calls Gemini really answered / changed the order
    confusions = Counter()
    per_phrase = defaultdict(lambda: [0, 0, 0])  # n, top1, top3
    for i, s in enumerate(rec.samples):
        ranked = rec.rank(s["frames"], top=5, exclude=i)
        names = [r["phrase"] for r in ranked]
        truth = s["phrase"]
        n += 1
        per_phrase[truth][0] += 1
        if names and names[0] == truth:
            top1 += 1
            per_phrase[truth][1] += 1
        elif names:
            confusions[(truth, names[0])] += 1
        if truth in names[:3]:
            top3 += 1
            per_phrase[truth][2] += 1
        if with_gemini:
            from . import rerank as R

            out = asyncio.run(R.rerank(ranked, {"time": time.strftime("%H:%M"), "history": [], "note": ""}))
            if out["mode"] == "gemini":
                g_answered += 1
                if out["choices"][:3] != names[:3]:
                    g_changed += 1
            if out["choices"] and out["choices"][0] == truth:
                g_top1 += 1
            if truth in out["choices"][:3]:
                g_top3 += 1
    res = {
        "samples": n,
        "phrases": len(phrases),
        "top1": round(top1 / n, 4),
        "top3": round(top3 / n, 4),
        "chance_top1": round(1 / len(phrases), 4),
        "chance_top3": round(min(1.0, 3 / len(phrases)), 4),
        "per_phrase": {
            p: {"n": v[0], "top1": round(v[1] / v[0], 4), "top3": round(v[2] / v[0], 4)}
            for p, v in sorted(per_phrase.items())
        },
        "confusions": [
            {"said": a, "predicted": b, "count": c} for (a, b), c in confusions.most_common(10)
        ],
    }
    if with_gemini:
        res["gemini_top1"] = round(g_top1 / n, 4)
        res["gemini_top3"] = round(g_top3 / n, 4)
        res["gemini_answered"] = g_answered  # the rest fell back to offline (classifier top 3)
        res["gemini_changed_order"] = g_changed
    return res


def evaluate_all(samples_by_user: dict[str, list[dict]], cfg: Config | None = None, with_gemini=False) -> dict:
    cfg = cfg or Config()
    users = {}
    tot_n = tot1 = tot3 = 0
    for user, samples in sorted(samples_by_user.items()):
        r = evaluate_user(samples, cfg, with_gemini)
        if r is None:
            continue
        users[user] = r
        tot_n += r["samples"]
        tot1 += round(r["top1"] * r["samples"])
        tot3 += round(r["top3"] * r["samples"])
    overall = None
    if tot_n:
        overall = {"samples": tot_n, "top1": round(tot1 / tot_n, 4), "top3": round(tot3 / tot_n, 4)}
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "leave-one-out, per user (each sample is matched only against that user's other samples)",
        "config": cfg.as_dict(),
        "with_gemini": with_gemini,
        "overall": overall,
        "users": users,
        "pooled": evaluate_pooled(samples_by_user, cfg),
        "starter": evaluate_starter(samples_by_user, cfg),
    }


def _score(rec: Recogniser, queries: list[dict], exclude_ids: bool) -> dict | None:
    ids = [s.get("id") for s in rec.samples]
    n = t1 = t3 = 0
    for q in queries:
        ex = ids.index(q["id"]) if exclude_ids and q.get("id") in ids else None
        names = [r["phrase"] for r in rec.rank(q["frames"], top=3, exclude=ex)]
        n += 1
        t1 += bool(names) and names[0] == q["phrase"]
        t3 += q["phrase"] in names
    if not n:
        return None
    return {"samples": n, "top1": round(t1 / n, 4), "top3": round(t3 / n, 4)}


def evaluate_pooled(samples_by_user: dict[str, list[dict]], cfg: Config) -> dict:
    """As deployed: one pool of everyone's recordings; each sample is held out and matched against
    everything else (own and other people's)."""
    pool = [s for ss in samples_by_user.values() for s in ss]
    out = {"method": "leave-one-out over the pooled recordings of every user (how /api/predict works)", "users": {}}
    if len({s["phrase"] for s in pool}) < 2:
        return out
    rec = Recogniser(pool, cfg)
    for user, ss in sorted(samples_by_user.items()):
        r = _score(rec, ss, exclude_ids=True)
        if r:
            out["users"][user] = r
    return out


def evaluate_starter(samples_by_user: dict[str, list[dict]], cfg: Config) -> dict:
    """No calibration: each user's samples are matched only against OTHER people's recordings,
    on the phrases those other people recorded."""
    out = {"method": "cross-person: each user's samples matched only against other users' recordings", "users": {}}
    for user, ss in sorted(samples_by_user.items()):
        others = [s for u, other in samples_by_user.items() if u != user for s in other]
        phrases = {s["phrase"] for s in others}
        if len(phrases) < 2:
            continue
        rec = Recogniser(others, cfg)
        queries = [q for q in ss if q["phrase"] in phrases]
        r = _score(rec, queries, exclude_ids=False)
        if r:
            r["phrases"] = len(phrases)
            r["chance_top1"] = round(1 / len(phrases), 4)
            r["chance_top3"] = round(min(1.0, 3 / len(phrases)), 4)
            out["users"][user] = r
    return out
