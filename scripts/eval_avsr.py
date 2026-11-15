"""Measure the open-source lip reader (Auto-AVSR) on OUR mouths, and pick its blend weight.

Uses calibration recordings that have mouth crops (recorded on the Calibrate page while the lip reader was
installed; saved in data/crops/, never in git). For each one:
  - lip reader alone: zero-shot, no calibration, no training, scored over the whole phrase list
  - matcher alone: pooled leave-one-out, as deployed
  - blended at several weights
The best weight (by "right sentence on a card", then first card) goes to data/avsr_config.json, which the
server reads. Needs at least 20 recordings with crops.

  .venv/bin/python scripts/eval_avsr.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import avsr  # noqa: E402
from server.app import AVSR_CONFIG, CROPS_DIR, PHRASES, blend_with_lip_reader, load_samples  # noqa: E402
from server.classifier import Config, Recogniser  # noqa: E402

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main():
    if not avsr.available():
        print("Lip reader not installed: bash scripts/fetch_avsr.sh"); return
    pool = [s for ss in load_samples().values() for s in ss]
    have = [s for s in pool if (CROPS_DIR / f"{s['id']}.npz").exists()]
    print(f"{len(have)} recordings with mouth crops (of {len(pool)}), from "
          f"{len({s['user'] for s in have})} people")
    if not have:
        print("Record on the Calibrate page with the server running (it saves crops automatically)."); return
    rec = Recogniser(pool, Config())
    pos = {id(s): i for i, s in enumerate(rec.samples)}
    phrases = PHRASES["phrases"]
    tally = {w: [0, 0] for w in WEIGHTS}
    free_ok = n = 0
    t0 = time.time()
    rows = []
    for s in have:
        crops = np.load(CROPS_DIR / f"{s['id']}.npz")["crops"]
        lip = avsr.read_lips(crops, phrases)
        ranked = rec.rank(s["frames"], top=999, exclude=pos.get(id(s)))
        n += 1
        for w in WEIGHTS:
            names = [c["phrase"] for c in blend_with_lip_reader(ranked, lip, w)]
            tally[w][0] += names[0] == s["phrase"]
            tally[w][1] += s["phrase"] in names[:3]
        free_ok += lip["text"].strip().lower() == s["phrase"].lower()
        top = max(lip["proba"], key=lip["proba"].get)
        rows.append((s["user"], s["phrase"], top, f'{lip["text"]} [{lip["text_conf"]}]'))
    print(f"\n{n} recordings, {time.time() - t0:.0f}s ({(time.time() - t0) / n:.2f}s each incl. free text)\n")
    for u, said, top, text in rows[:15]:
        print(f"  {u:8s} said {said!r:24s} lip reader's best phrase {top!r:24s} free text {text!r}")
    print("\n  weight on lip reader   first card   in the 3 cards")
    for w in WEIGHTS:
        label = {0.0: " (matcher alone)", 1.0: " (lip reader alone, zero-shot)"}.get(w, "")
        print(f"  {w:<4}{label:32s} {tally[w][0] / n:6.1%}      {tally[w][1] / n:6.1%}")
    print(f"  free text exactly right: {free_ok}/{n}")
    best = max(WEIGHTS, key=lambda w: (tally[w][1], tally[w][0]))
    if n >= 20:
        AVSR_CONFIG.write_text(json.dumps({"weight": best, "n": n, "measured_at": time.strftime("%Y-%m-%d %H:%M"),
                                           "results": {str(w): {"top1": tally[w][0] / n, "top3": tally[w][1] / n} for w in WEIGHTS}}, indent=2))
        print(f"\nSaved weight {best} to {AVSR_CONFIG.relative_to(ROOT)} (the server picks it up on the next prediction).")
    else:
        print(f"\nOnly {n} recordings: not saving a weight (need 20). Default stays {0.5}.")


if __name__ == "__main__":
    main()
