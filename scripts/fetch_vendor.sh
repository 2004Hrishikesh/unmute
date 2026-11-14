#!/usr/bin/env bash
# Downloads MediaPipe Face Landmarker files into web/vendor/ so Unmute works with no internet.
# (~15 MB, the same files the browser otherwise loads from the CDN.)
set -euo pipefail
cd "$(dirname "$0")/.."
V=0.10.14
CDN="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@$V"
mkdir -p web/vendor/tasks-vision/wasm
curl -fsSL "$CDN/vision_bundle.mjs" -o web/vendor/tasks-vision/vision_bundle.mjs
for f in vision_wasm_internal.js vision_wasm_internal.wasm vision_wasm_nosimd_internal.js vision_wasm_nosimd_internal.wasm; do
  curl -fsSL "$CDN/wasm/$f" -o "web/vendor/tasks-vision/wasm/$f"
done
curl -fsSL "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
  -o web/vendor/face_landmarker.task
du -sh web/vendor
echo "Done. Reload the page: it now loads the face tracker from local files."
