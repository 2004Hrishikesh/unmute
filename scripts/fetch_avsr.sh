#!/usr/bin/env bash
# Install the open-source lip reader (Auto-AVSR, Imperial College London, Apache 2.0): code + 1 GB of weights.
# Everything lands in third_party/ (gitignored). Run once, with internet; after that it works offline.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/pip install -q torch torchvision torchaudio sentencepiece opencv-python-headless
mkdir -p third_party/weights
[ -d third_party/auto_avsr ] || git clone -q --depth 1 https://github.com/mpc001/auto_avsr third_party/auto_avsr
# Upstream assumes CUDA or CPU; make the CTC scorer follow the input's device so it runs on Apple GPUs (MPS).
python3 - <<'PY'
import re
p = "third_party/auto_avsr/espnet/nets/ctc_prefix_score.py"; s = open(p).read()
s = re.sub(r'self\.device = \(\s*torch\.device\("cuda:%d" % x\.get_device\(\)\)\s*if x\.is_cuda\s*else torch\.device\("cpu"\)\s*\)',
           'self.device = x.device  # unmute: works on Apple GPUs (mps) too', s)
open(p, "w").write(s)
PY
W=third_party/weights/vsr_trlrs2lrs3vox2avsp_base.pth
if [ ! -f "$W" ]; then
  echo "Downloading weights (1 GB)…"
  curl -sSL -o "$W" "https://drive.usercontent.google.com/download?id=1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_&export=download&confirm=t"
fi
# Published checksum prefix for this file: 49f77
md5 -q "$W" 2>/dev/null | grep -q '^49f77' || md5sum "$W" | grep -q '^49f77' || { echo "Checksum mismatch: delete $W and retry"; exit 1; }
echo "Lip reader installed. Restart the server."
