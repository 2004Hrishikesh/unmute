#!/usr/bin/env bash
# Everything to run before the demo, in order. The server must already be running in another terminal:
#   .venv/bin/uvicorn server.app:app --reload --port 8000
# Usage: bash scripts/before_demo.sh <presenter-name>     (e.g. rishi)
set -uo pipefail
cd "$(dirname "$0")/.."
WHO="${1:-rishi}"
step() { printf "\n\033[1m== %s\033[0m\n" "$1"; }

step "1/5 Tests (should say 'passed', no 'failed')"
UNMUTE_NO_AVSR=1 .venv/bin/python -m pytest -q 2>&1 | tail -2

step "2/5 Accuracy of the matcher on everyone's recordings (updates the Proof page)"
.venv/bin/python scripts/evaluate.py

step "3/5 Train the many-people model (about 1 minute per person; only switched on if it beats the matcher)"
.venv/bin/python scripts/train.py | tail -8

step "4/5 Open-source lip reader on our own mouths (needs Calibrate recordings made with the lip reader installed)"
if ls data/crops/*.npz >/dev/null 2>&1; then .venv/bin/python scripts/eval_avsr.py 2>&1 | grep -v -i warning | tail -36
else echo "No mouth crops yet: record on the Calibrate page first, then re-run this script."; fi

step "5/5 Demo readiness check for '$WHO'"
.venv/bin/python scripts/demo_check.py "$WHO"
