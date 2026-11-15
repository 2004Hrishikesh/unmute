#!/usr/bin/env python3
"""Demo readiness check. Run it 10 minutes before presenting:

    .venv/bin/python scripts/demo_check.py [presenter]

Prints one line per item with OK / WARN / FAIL and what to do about it. Spends no Gemini quota.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://localhost:8000"
presenter = (sys.argv[1] if len(sys.argv) > 1 else "rishi").strip().lower()


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=5) as r:
        return json.load(r)


def line(status, what, fix=""):
    print(f"[{status:4s}] {what}" + (f"\n       -> {fix}" if fix else ""))


ok = True
try:
    get("/health")
    line("OK", f"Server answering at {BASE}")
except Exception as e:
    line("FAIL", f"Server not answering at {BASE} ({e})",
         "run: .venv/bin/uvicorn server.app:app --port 8000")
    sys.exit(1)

st = get("/api/status")
g = st["gemini"]
if g["available"]:
    line("OK", f"Gemini on: {g['model']}")
else:
    line("WARN", f"Gemini off: {g.get('error')}",
         "Fine for the demo: say 'offline mode, same pipeline'. Key goes in .env as GEMINI_API_KEY=...")
if g.get("resting_s"):
    line("WARN", "Some Gemini models are resting (quota/overload): " + ", ".join(
        f"{m} {s}s" for m, s in g["resting_s"].items()), "Don't run evaluate.py --with-gemini before the demo.")
line("OK", f"ntfy topic: {st['ntfy_topic']}  (subscribe to it in the ntfy app on the nurse's phone)")

vendor = ROOT / "web" / "vendor"
need = [vendor / "face_landmarker.task", vendor / "tasks-vision" / "vision_bundle.mjs",
        vendor / "tasks-vision" / "wasm" / "vision_wasm_internal.wasm"]
if all(p.exists() and p.stat().st_size > 1000 for p in need):
    line("OK", "Face tracker files are local: works with no internet")
else:
    line("WARN", "Face tracker will load from the internet (web/vendor is empty)",
         "run: bash scripts/fetch_vendor.sh   (22 MB, once)")

info = get(f"/api/enrollments?user={presenter}")
enr = info["counts"]
if info.get("starter_available"):
    line("OK", f"Starter model: {info['starter_samples']} recordings from {info['starter_users']} other people, so a new name works with no calibration")
else:
    line("WARN", "No starter model: nobody else has recorded on this laptop", "the 'judge' moment in the demo script needs at least one other person's recordings")
lr = st.get("lip_reader") or {}
if lr.get("installed") and lr.get("error") is None:
    line("OK", f"Open-source lip reader installed ({lr.get('device') or 'not loaded yet'}), blend weight {lr.get('weight')} (new patient {lr.get('weight_new')})"
         + (" (measured)" if lr.get("weight_measured") else " (default: not measured on our mouths yet)"))
    if not lr.get("weight_measured"):
        line("WARN", "Lip reader not measured on our mouths", "record on Calibrate (crops are saved), then: .venv/bin/python scripts/eval_avsr.py")
elif lr.get("error"):
    line("WARN", f"Lip reader failed to load: {lr['error']}", "the app still works without it")
else:
    line("WARN", "Open-source lip reader not installed", "bash scripts/fetch_avsr.sh  (1 GB, once)")
mi = st.get("model") or {}
if mi.get("active"):
    line("OK", f"Many-people model on: {mi['samples']} recordings from {mi['people']} people, trained {mi['trained_at']}")
else:
    line("WARN", "Many-people model off (not trained, or it did not beat the matcher)",
         "record volunteers (Calibrate -> Volunteer), then: .venv/bin/python scripts/train.py")
phrases = get("/api/phrases")
demo = phrases["demo_set"]
if not enr:
    line("FAIL", f"No calibration for presenter '{presenter}'", "open /enroll.html, type the name, record the demo set")
    ok = False
else:
    missing = [p for p in demo if enr.get(p, 0) < 3]
    low = [f"{p} ({enr.get(p, 0)})" for p in demo if 3 <= enr.get(p, 0) < 6]
    n = sum(enr.values())
    line("OK" if not missing else "FAIL", f"'{presenter}' has {n} recordings; demo phrases with < 3 samples: {missing or 'none'}")
    if low:
        line("WARN", "Demo phrases with fewer than 6 samples: " + ", ".join(low),
             "more samples measured better (2: 51%, 3: 64%, 4: 75% top-1 for rishi)")
    ok = ok and not missing

res_path = ROOT / "data" / "results.json"
if res_path.exists():
    res = json.loads(res_path.read_text())
    u = res.get("users", {}).get(presenter)
    if u:
        line("OK", f"Measured for '{presenter}' ({res.get('generated_at', '')[:16]}): top-1 {u['top1']:.0%}, top-3 {u['top3']:.0%} on {u['samples']} samples")
        best = sorted(u["per_phrase"].items(), key=lambda kv: (-kv[1]["top1"], -kv[1]["top3"]))
        line("OK", "Best phrases to demo: " + ", ".join(f'"{p}" ({v["top1"]:.0%})' for p, v in best[:5]))
        weak = [f'"{p}" ({v["top1"]:.0%})' for p, v in best if v["top1"] < 0.5]
        if weak:
            line("WARN", "Avoid on stage: " + ", ".join(weak))
    else:
        line("WARN", f"No measured results for '{presenter}'", "run: .venv/bin/python scripts/evaluate.py")
else:
    line("WARN", "No data/results.json yet", "run: .venv/bin/python scripts/evaluate.py")

reqs = get("/api/requests?limit=5")
if reqs.get("unacknowledged_urgent"):
    line("WARN", f"{reqs['unacknowledged_urgent']} urgent request(s) still unacknowledged on the Nurse page",
         "open /nurse.html and click Acknowledge, so the demo starts clean")
else:
    line("OK", "Nurse page is clear (no open urgent requests)")

print()
print("Also, by hand: Brave at http://localhost:8000, camera allowed and built-in camera picked, close Zoom/FaceTime,"
      " front light on, click once on the page so sound can play, Nurse page open on a second screen with"
      " 'Enable alarm sound' clicked, ntfy app subscribed on the phone.")
sys.exit(0 if ok else 1)
