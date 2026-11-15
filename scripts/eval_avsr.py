"""Measure the open-source lip reader (Auto-AVSR) on OUR mouths, and pick its blend weights.

Uses calibration recordings that have mouth crops (recorded on the Calibrate page while the lip reader was
installed; saved in data/crops/, never in git). For each one, three situations:
  - lip reader alone: zero-shot, no calibration, no training, scored over the whole phrase list
  - BRAND-NEW patient: the matcher only has OTHER people's recordings (leave-one-person-out), blended with
    the lip reader. This is day one in the ICU.
  - calibrated patient: the matcher has everything except this one recording (incl. the person's own others)
Each blend is tried at several weights; the best (by "right sentence on a card", then first card) goes to
data/avsr_config.json as "weight_new" and "weight", which the server reads.

Also checks the free text: how often it would be shown, and how often it would WRONGLY take card 1 (every
recording here is a list phrase, so free text as card 1 is always a mistake on this data).

The lip reader's outputs are cached in data/crops/_lip_cache.json (by recording id), so a re-run is seconds.
Needs at least 20 recordings with crops.

  .venv/bin/python scripts/eval_avsr.py [--no-cache]
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import avsr  # noqa: E402
from server.app import AVSR_CONFIG, CROPS_DIR, PHRASES, blend_with_lip_reader, load_samples  # noqa: E402
from server.classifier import Config, Recogniser  # noqa: E402

WEIGHTS = (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)
CACHE = CROPS_DIR / "_lip_cache.json"


def _lip_outputs(have, phrases, use_cache=True):
    cache = {}
    if use_cache and CACHE.exists():
        try:
            cache = json.loads(CACHE.read_text())
            if cache.get("_phrases") != phrases:
                cache = {}
        except Exception:
            cache = {}
    cache["_phrases"] = phrases
    todo = [s for s in have if s["id"] not in cache]
    t0 = time.time()
    for i, s in enumerate(todo):
        crops = np.load(CROPS_DIR / f"{s['id']}.npz")["crops"]
        lip = avsr.read_lips(crops, phrases)
        cache[s["id"]] = {k: lip[k] for k in ("scores", "proba", "text", "text_conf", "text_total",
                                               "text_shown", "text_first")}
        if (i + 1) % 25 == 0:
            print(f"  lip reader: {i + 1}/{len(todo)} ({(time.time() - t0) / (i + 1):.2f}s each)", flush=True)
            CACHE.write_text(json.dumps(cache))
    CACHE.write_text(json.dumps(cache))
    if todo:
        print(f"  lip reader ran on {len(todo)} new recordings in {time.time() - t0:.0f}s "
              f"({(time.time() - t0) / len(todo):.2f}s each incl. free text); {len(have) - len(todo)} from cache")
    return cache


def _score(tally, w, names, truth):
    v = tally.setdefault(w, [0, 0, 0])
    v[0] += 1; v[1] += names[0] == truth; v[2] += truth in names[:3]


def _table(title, tally, extra=None):
    print(f"\n  {title}")
    print("  weight on lip reader                 first card   in the 3 cards")
    for w in WEIGHTS:
        n, h1, h3 = tally[w]
        label = {0.0: " (matcher alone)", 1.0: " (lip reader alone, zero-shot)"}.get(w, "")
        print(f"  {w:<5}{label:32s} {h1 / n:6.1%}      {h3 / n:6.1%}   (n={n})")


def _best(tally):
    return max(WEIGHTS, key=lambda w: (tally[w][2], tally[w][1]))


def main():
    if not avsr.available():
        print("Lip reader not installed: bash scripts/fetch_avsr.sh"); return
    by_user = load_samples()
    pool = [s for ss in by_user.values() for s in ss]
    phrases = PHRASES["phrases"]
    have = [s for s in pool if (CROPS_DIR / f"{s['id']}.npz").exists() and s["phrase"] in phrases]
    print(f"{len(have)} recordings with mouth crops (of {len(pool)}), from {len({s['user'] for s in have})} people")
    if not have:
        print("Record on the Calibrate page with the server running (it saves crops automatically)."); return
    lips = _lip_outputs(have, phrases, use_cache="--no-cache" not in sys.argv)

    rec_all = Recogniser(pool, Config())
    pos = {id(s): i for i, s in enumerate(rec_all.samples)}
    rec_others = {}
    new, cal = {}, {}
    per_user = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    free = dict.fromkeys(("exact", "shown", "first", "n", "off_first_right", "off_first_wrong", "off_shown_right"), 0)
    confs_right, confs_wrong = [], []
    for s in have:
        lip = lips[s["id"]]
        u = s["user"]
        if u not in rec_others:
            others = [o for v, ss in by_user.items() if v != u for o in ss]
            rec_others[u] = Recogniser(others, Config()) if len({o["phrase"] for o in others}) >= 2 else None
        r_new = rec_others[u].rank(s["frames"], top=999) if rec_others[u] else []
        r_cal = rec_all.rank(s["frames"], top=999, exclude=pos.get(id(s)))
        for w in WEIGHTS:
            names = [c["phrase"] for c in blend_with_lip_reader(r_new, lip, w)]
            _score(new, w, names, s["phrase"])
            _score(per_user[u], w, names, s["phrase"])
            _score(cal, w, [c["phrase"] for c in blend_with_lip_reader(r_cal, lip, w)], s["phrase"])
        right = avsr._norm(lip["text"]) == avsr._norm(s["phrase"])
        place = avsr.place_free_text(lip["text"], lip["text_conf"], lip["text_total"], lip["scores"])
        free["n"] += 1; free["exact"] += right
        free["shown"] += place["show"]; free["first"] += place["first"]
        # Off the list, simulated: take the true phrase OFF the list. Free text should now be card 1, and right.
        off = avsr.place_free_text(lip["text"], lip["text_conf"], lip["text_total"],
                                   {p: v for p, v in lip["scores"].items() if p != s["phrase"]})
        close = avsr._similar(lip["text"] or "", s["phrase"]) >= 0.75
        free["off_first_right"] += off["first"] and close
        free["off_first_wrong"] += off["first"] and not close
        free["off_shown_right"] += off["show"] and close
        if lip["text_conf"] is not None:
            (confs_right if right else confs_wrong).append(lip["text_conf"])

    n = len(have)
    print(f"\n{n} recordings, all phrases from the list of {len(phrases)} (chance: first card "
          f"{1 / len(phrases):.0%}, in the 3 cards {3 / len(phrases):.0%})")
    _table("BRAND-NEW patient (matcher has only other people's recordings) + lip reader:", new)
    _table("Calibrated patient (matcher has their own other recordings) + lip reader:", cal)
    w_new, w_cal = _best(new), _best(cal)
    print("\n  Brand-new patient, per person (lip reader alone | best blend):")
    for u, t in sorted(per_user.items()):
        a, b = t[1.0], t[w_new]
        print(f"    {u:8s} n={a[0]:3d}   {a[1] / a[0]:6.1%} / {a[2] / a[0]:6.1%}   |   {b[1] / b[0]:6.1%} / {b[2] / b[0]:6.1%}")

    def med(x):
        return f"{np.median(x):.2f}" if x else "-"
    print(f"\n  Free text exactly the phrase: {free['exact']}/{n}  (median confidence: right {med(confs_right)}, "
          f"wrong {med(confs_wrong)}; shown when > {avsr.HEARD_MIN_LOGPROB})")
    print(f"  On list phrases, free text shown as an extra card: {free['shown']}/{n}; WRONGLY as card 1: "
          f"{free['first']}/{n}  (a misreading of a list phrase is hidden: that card is already there)")
    print(f"  Off the list (simulated: true phrase removed from the list): free text is card 1 and right "
          f"{free['off_first_right']}/{n}, card 1 and wrong {free['off_first_wrong']}/{n}, shown and right at all "
          f"{free['off_shown_right']}/{n}")

    if n >= 20:
        cfg = {"weight": w_cal, "weight_new": w_new, "n": n, "people": len(per_user),
               "measured_at": time.strftime("%Y-%m-%d %H:%M"),
               "new_patient": {str(w): {"top1": new[w][1] / n, "top3": new[w][2] / n} for w in WEIGHTS},
               "calibrated": {str(w): {"top1": cal[w][1] / n, "top3": cal[w][2] / n} for w in WEIGHTS},
               "free_text": {**free, "wrong_first_rate": free["first"] / n}}
        AVSR_CONFIG.write_text(json.dumps(cfg, indent=2))
        print(f"\nSaved weights: brand-new patient {w_new}, calibrated {w_cal} to "
              f"{AVSR_CONFIG.relative_to(ROOT)} (the server picks them up on the next prediction).")
    else:
        print(f"\nOnly {n} recordings: not saving weights (need 20). Default stays 0.5.")


if __name__ == "__main__":
    main()
