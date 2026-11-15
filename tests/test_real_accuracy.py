"""Accuracy on the team's REAL recordings in data/enrollments (skipped when there are none).

These are regression guards, not the headline numbers (scripts/evaluate.py prints those). The floors sit a
little under what was measured on 14 November 2026 (pooled leave-one-out over 142 recordings: kesh 77.5% /
92.2%, rishi 70.0% / 97.5%), so a change that makes lip reading clearly worse fails here.
"""
import json
from pathlib import Path

import pytest

from server.classifier import Config
from server.evaluation import evaluate_pooled, evaluate_user

ENROLL = Path(__file__).resolve().parent.parent / "data" / "enrollments"


def _real():
    by_user = {}
    for f in sorted(ENROLL.glob("*.json")):
        s = json.loads(f.read_text())
        by_user.setdefault(s["user"], []).append(s)
    return by_user


REAL = _real() if ENROLL.exists() else {}
pytestmark = pytest.mark.skipif(sum(map(len, REAL.values())) < 40, reason="no real recordings on this machine")


def test_personal_model_accuracy_floor():
    tot = t1 = t3 = 0
    for ss in REAL.values():
        r = evaluate_user(ss, Config())
        if r:
            tot += r["samples"]; t1 += r["top1"] * r["samples"]; t3 += r["top3"] * r["samples"]
            # every person far above chance
            assert r["top3"] >= 3 * r["chance_top3"] or r["top3"] >= 0.85
    assert tot >= 40
    assert t1 / tot >= 0.70 and t3 / tot >= 0.88, (t1 / tot, t3 / tot)


def test_as_deployed_pooled_accuracy_floor():
    res = evaluate_pooled(REAL, Config())
    n = sum(v["samples"] for v in res["users"].values())
    top3 = sum(v["top3"] * v["samples"] for v in res["users"].values()) / n
    top1 = sum(v["top1"] * v["samples"] for v in res["users"].values()) / n
    assert top1 >= 0.65 and top3 >= 0.88, (top1, top3)


def test_urgent_phrases_are_reliably_in_the_three_cards():
    """'I cannot breathe' must not hide: urgent phrases in the top 3 at least 85% of the time (as deployed)."""
    from server.classifier import Recogniser

    urgent = set(json.loads((ENROLL.parent / "phrases.json").read_text())["urgent"])
    pool = [s for ss in REAL.values() for s in ss]
    rec = Recogniser(pool, Config())
    hits = n = 0
    for i, s in enumerate(rec.samples):
        if s["phrase"] in urgent:
            names = [r["phrase"] for r in rec.rank(s["frames"], top=3, exclude=i)]
            n += 1; hits += s["phrase"] in names
    assert n >= 10
    assert hits / n >= 0.85, hits / n
