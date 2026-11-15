"""Accuracy on the team's REAL recordings in data/enrollments (skipped when there are none).

These are regression guards, not the headline numbers (scripts/evaluate.py prints those).

Two sets of floors:
- STRICT, on the original team (kesh, rishi; 143 recordings): a little under what was measured on 14 Nov 2026
  (kesh 80.6% / 90.3%, rishi 75.0% / 95.0%), so a code change that makes lip reading worse fails here.
- EVERYONE (4 people, 385 recordings, 20 phrases): a little under the measured 64.9% / 84.2% personal and
  62.3% / 85.2% pooled. Lower because one volunteer (pankaj) is a much harder lip-reading case (43.8% / 68.6%);
  that is a fact about the data, not the code, and still ~4.5x chance on 20 phrases.
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
TEAM = {u: ss for u, ss in REAL.items() if u in ("kesh", "rishi")}
pytestmark = pytest.mark.skipif(sum(map(len, REAL.values())) < 40, reason="no real recordings on this machine")


def _personal(by_user):
    tot = t1 = t3 = 0
    for ss in by_user.values():
        r = evaluate_user(ss, Config())
        if r:
            tot += r["samples"]; t1 += r["top1"] * r["samples"]; t3 += r["top3"] * r["samples"]
            # every person far above chance
            assert r["top3"] >= 3 * r["chance_top3"] or r["top3"] >= 0.85, (r["top3"], r["chance_top3"])
    return tot, t1 / tot, t3 / tot


def _pooled(by_user):
    res = evaluate_pooled(by_user, Config())
    n = sum(v["samples"] for v in res["users"].values())
    return (sum(v["top1"] * v["samples"] for v in res["users"].values()) / n,
            sum(v["top3"] * v["samples"] for v in res["users"].values()) / n)


def _urgent_top3(by_user):
    from server.classifier import Recogniser

    urgent = set(json.loads((ENROLL.parent / "phrases.json").read_text())["urgent"])
    rec = Recogniser([s for ss in by_user.values() for s in ss], Config())
    hits = n = 0
    for i, s in enumerate(rec.samples):
        if s["phrase"] in urgent:
            names = [r["phrase"] for r in rec.rank(s["frames"], top=3, exclude=i)]
            n += 1; hits += s["phrase"] in names
    assert n >= 10
    return hits / n


@pytest.mark.skipif(len(TEAM) < 2, reason="original team recordings missing")
def test_team_personal_accuracy_floor():
    tot, top1, top3 = _personal(TEAM)
    assert tot >= 40
    assert top1 >= 0.74 and top3 >= 0.88, (top1, top3)


def test_everyone_personal_accuracy_floor():
    tot, top1, top3 = _personal(REAL)
    assert top1 >= 0.60 and top3 >= 0.80, (top1, top3)


@pytest.mark.skipif(len(TEAM) < 2, reason="original team recordings missing")
def test_team_as_deployed_pooled_accuracy_floor():
    top1, top3 = _pooled(TEAM)
    assert top1 >= 0.65 and top3 >= 0.88, (top1, top3)


def test_everyone_as_deployed_pooled_accuracy_floor():
    top1, top3 = _pooled(REAL)
    assert top1 >= 0.58 and top3 >= 0.82, (top1, top3)


@pytest.mark.skipif(len(TEAM) < 2, reason="original team recordings missing")
def test_team_urgent_phrases_are_reliably_in_the_three_cards():
    """'I cannot breathe' must not hide: urgent phrases in the top 3 at least 90% of the time (as deployed)."""
    assert _urgent_top3(TEAM) >= 0.90


def test_everyone_urgent_phrases_in_the_three_cards():
    assert _urgent_top3(REAL) >= 0.80
