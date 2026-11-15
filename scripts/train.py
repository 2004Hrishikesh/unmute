"""Train the many-people lip-reading model, test it fairly, and deploy it only if it measures better.

Usage (from the repo root), after volunteers have recorded on the Calibrate page:
  .venv/bin/python scripts/train.py            # test (leave-one-person-out), train on everyone, save
  .venv/bin/python scripts/train.py --quick    # fewer epochs/nets: a fast look while collecting data

The test: for each person, train on everyone else and test on that person = a brand-new patient. The matcher
alone, the model alone, and both fused are scored on the same recordings. The best setting by "right sentence
on a card" (top-3), then first card (top-1), is deployed. If the matcher alone wins, the model is saved but
switched off. The running server picks a new model up by itself; no restart.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.app import MODEL_PATH, load_samples  # noqa: E402
from server.model_eval import best_setting, lopo  # noqa: E402
from server.neural import LipNet, NetConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = NetConfig(epochs=50, ensemble=2) if args.quick else NetConfig()

    by_user = load_samples()
    people = {u: len(s) for u, s in by_user.items() if s}
    n = sum(people.values())
    print(f"{n} recordings from {len(people)} people: " + ", ".join(f"{u} {c}" for u, c in sorted(people.items())))
    if len(people) < 2:
        print("Need at least 2 people. Ask volunteers to record on the Calibrate page (Volunteer setting).")
        return

    t0 = time.time()
    res = lopo(by_user, cfg, log=lambda m: print("  " + m))
    print(f"\nBrand-new patient (leave-one-person-out), {res['people']} people, {time.time() - t0:.0f}s:")
    rows = sorted(res["overall"].items(), key=lambda kv: (-kv[1]["top3"], -kv[1]["top1"]))
    for k, v in rows:
        print(f"  {k:28s} first card {v['top1']:6.1%}   in the 3 cards {v['top3']:6.1%}   ({v['n']} recordings)")
    best, score = best_setting(res)
    m = res["overall"]["matcher"]
    use = best != "matcher" and (score["top3"], score["top1"]) > (m["top3"], m["top1"])
    alpha, tau = 0.0, 0.1
    if best == "model":
        alpha = 1.0
    elif best.startswith("fused"):
        alpha = float(best.split("a=")[1].split()[0]); tau = float(best.split("tau=")[1])

    print(f"\nTraining the deployed model on all {n} recordings…")
    net = LipNet.train([s for ss in by_user.values() for s in ss], cfg)
    cal = res.get("calibrated") or {}
    personal = {"matcher": cal.get("matcher"), "fused": cal.get(best)} if use and best.startswith("fused") and cal.get(best) else None
    use_personal = bool(personal) and (personal["fused"]["top3"], personal["fused"]["top1"]) >= (
        personal["matcher"]["top3"], personal["matcher"]["top1"])
    net.meta.update({
        "use": use, "alpha": alpha, "tau": tau, "use_for_calibrated": use_personal, "lopo": res,
        "best": best, "picked_note": (
            "The fusion weight was picked on the same leave-one-person-out results it is reported with; "
            "with few people that is optimistic. It gets fairer with every volunteer."),
        "personal_check": personal,
    })
    net.save(MODEL_PATH)
    print(f"Saved {MODEL_PATH.relative_to(ROOT)}.npz/.json ({net.meta['train_s']}s training)")
    if use:
        print(f"DEPLOYED: {best} (alpha {alpha}, tau {tau}): first card {score['top1']:.1%} vs matcher {m['top1']:.1%}, "
              f"in the 3 cards {score['top3']:.1%} vs {m['top3']:.1%} for a brand-new patient.")
        if personal:
            print(f"People who also have their own recordings (model trained without them): matcher "
                  f"{personal['matcher']['top1']:.1%} / {personal['matcher']['top3']:.1%}, fused "
                  f"{personal['fused']['top1']:.1%} / {personal['fused']['top3']:.1%} -> "
                  f"{'fused' if use_personal else 'matcher only'} for them.")
    else:
        print("NOT deployed: the matcher alone is as good or better on this data. Record more people and re-run.")


if __name__ == "__main__":
    main()
