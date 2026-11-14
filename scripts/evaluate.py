"""Leave-one-out accuracy on the enrolled samples.

Usage (from the repo root):
  .venv/bin/python scripts/evaluate.py              # writes data/results.json
  .venv/bin/python scripts/evaluate.py --sweep      # compare a few classifier settings (prints only)
  .venv/bin/python scripts/evaluate.py --with-gemini  # also measure top-3 after Gemini re-ranking

Every number printed or saved here is measured on real enrolled samples.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.app import RESULTS_FILE, load_samples  # noqa: E402
from server.classifier import Config  # noqa: E402
from server.evaluation import evaluate_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--with-gemini", action="store_true")
    ap.add_argument("--user", default=None)
    args = ap.parse_args()

    by_user = load_samples(args.user)
    if not by_user:
        print("No enrollments found in data/enrollments/. Calibrate first.")
        return

    if args.sweep:
        configs = [
            Config(),
            Config(k=2),
            Config(trim=False),
            Config(center=False),
            Config(deltas=False),
            Config(bs_weight=1.0),
            Config(length=40),
            Config(window=10),
            Config(length=40, k=2, deltas=False, center=False, trim=False, bs_weight=1.0),  # original
        ]
        print(f"{'config':70s} top1   top3")
        for c in configs:
            r = evaluate_all(by_user, c)
            o = r["overall"]
            if o:
                print(f"{json.dumps(c.as_dict()):70s} {o['top1']:.3f}  {o['top3']:.3f}")
        return

    res = evaluate_all(by_user, Config(), with_gemini=args.with_gemini)
    RESULTS_FILE.write_text(json.dumps(res, indent=2))
    if not res["overall"]:
        print("Not enough data: each user needs at least 2 phrases with 2+ samples each.")
        return
    o = res["overall"]
    print(f"Overall: {o['samples']} samples, top-1 {o['top1']:.1%}, top-3 {o['top3']:.1%}")
    for u, r in res["users"].items():
        line = (f"  {u}: {r['samples']} samples / {r['phrases']} phrases, top-1 {r['top1']:.1%} "
                f"(chance {r['chance_top1']:.1%}), top-3 {r['top3']:.1%} (chance {r['chance_top3']:.1%})")
        if "gemini_top3" in r:
            line += f", with Gemini top-1 {r['gemini_top1']:.1%} top-3 {r['gemini_top3']:.1%}"
        print(line)
    print(f"Saved to {RESULTS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
