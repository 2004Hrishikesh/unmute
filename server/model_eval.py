"""Is the trained model better than the DTW matcher for a NEW patient? Measured, never assumed.

Leave-one-person-out (LOPO): for each person, train the model and build the matcher on everyone ELSE, then
test on that person's recordings. That is exactly a brand-new patient on day one. Scored for the matcher alone,
the model alone, and the two fused at several weights.
"""
from __future__ import annotations

from collections import Counter

from .classifier import Config, Recogniser
from .neural import LipNet, NetConfig, dtw_proba, fuse

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)   # 0 = matcher only, 1 = trained model only
TAUS = (0.05, 0.1, 0.2)


def _hits(names: list[str], truth: str) -> tuple[int, int]:
    return int(bool(names) and names[0] == truth), int(truth in names[:3])


def lopo(samples_by_user: dict[str, list[dict]], net_cfg: NetConfig | None = None, cfg: Config | None = None,
         log=None) -> dict:
    net_cfg, cfg = net_cfg or NetConfig(), cfg or Config()
    users = [u for u, ss in samples_by_user.items() if ss]
    if len(users) < 2:
        return {"error": "need recordings from at least 2 people", "users": {}}
    tally: dict[str, list[int]] = {}
    calibrated: dict[str, list[int]] = {}  # same people, but WITH their own other recordings in the matcher pool
    per_user = {}
    pool = [s for ss in samples_by_user.values() for s in ss]
    rec_all = Recogniser(pool, cfg)
    pos = {id(s): i for i, s in enumerate(rec_all.samples)}

    def add(key, h1, h3, *buckets):
        for d in buckets:
            v = d.setdefault(key, [0, 0, 0])
            v[0] += 1; v[1] += h1; v[2] += h3

    for u in sorted(users):
        others = [s for v, ss in samples_by_user.items() if v != u for s in ss]
        phrases = {s["phrase"] for s in others}
        queries = [q for q in samples_by_user[u] if q["phrase"] in phrases]
        if len({s["phrase"] for s in others}) < 2 or not queries:
            continue
        if log:
            log(f"{u}: training on {len(others)} recordings from {len(users) - 1} other people, testing {len(queries)}")
        net = LipNet.train(others, net_cfg)
        rec = Recogniser(others, cfg)
        mine: dict[str, list[int]] = {}
        for q in queries:
            ranked = rec.rank(q["frames"], top=999)
            pn = net.proba(q["frames"], q.get("dims"))
            add("matcher", *_hits([r["phrase"] for r in ranked], q["phrase"]), mine, tally)
            add("model", *_hits([p for p, _ in sorted(pn.items(), key=lambda kv: -kv[1]) if p in phrases],
                                 q["phrase"]), mine, tally)
            for tau in TAUS:
                pd = dtw_proba(ranked, tau)
                for a in ALPHAS[1:-1]:
                    add(f"fused a={a} tau={tau}", *_hits([p for p, _ in fuse(pn, pd, a)], q["phrase"]), mine, tally)
        per_user[u] = {k: _fmt(v) for k, v in mine.items()} | {"phrases": len(phrases), "chance_top1": round(1 / len(phrases), 4)}
        # Calibrated: the matcher pool holds everything except this one recording (incl. the person's own
        # other samples); the model still never saw this person. Fair for "does fusing hurt a calibrated person".
        for q in samples_by_user[u]:
            i = pos.get(id(q))
            if i is None:
                continue
            ranked = rec_all.rank(q["frames"], top=999, exclude=i)
            pn = net.proba(q["frames"], q.get("dims"))
            add("matcher", *_hits([r["phrase"] for r in ranked], q["phrase"]), calibrated)
            for tau in TAUS:
                pd = dtw_proba(ranked, tau)
                for a in ALPHAS[1:-1]:
                    add(f"fused a={a} tau={tau}", *_hits([p for p, _ in fuse(pn, pd, a)], q["phrase"]), calibrated)
    results = {k: _fmt(v) for k, v in tally.items()}
    return {"method": "leave-one-person-out: train on everyone else, test on the person left out",
            "people": len(users), "overall": results, "users": per_user,
            "calibrated": {k: _fmt(v) for k, v in calibrated.items()}}


def best_setting(res: dict) -> tuple[str, dict]:
    """The best scorer by top-3 then top-1 (top-3 = the right sentence is on a card)."""
    items = sorted(res["overall"].items(), key=lambda kv: (kv[1]["top3"], kv[1]["top1"]), reverse=True)
    return items[0]


def _fmt(v):
    return {"n": v[0], "top1": round(v[1] / v[0], 4), "top3": round(v[2] / v[0], 4)}
