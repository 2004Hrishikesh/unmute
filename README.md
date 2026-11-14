# Unmute

**Unmute helps hospital patients who can't make a sound, such as ICU patients with a breathing tube or a tracheostomy, talk again.**
The patient silently mouths a sentence at a laptop camera. Unmute reads their lips and shows the 3 most likely
sentences. The patient nods to confirm one and the laptop says it out loud. Urgent requests ("I cannot breathe") turn
the screen red and buzz the nurse's phone.

> The one tool that exists understands 40 fixed phrases. Patients have more to say than that.

Unmute is a communication aid. It is not a medical device and it does not diagnose anything.

## How it works

```
 Browser (web/)                                         Python backend (server/, FastAPI)
 ┌──────────────────────────────────────┐               ┌───────────────────────────────────────────┐
 │ webcam → MediaPipe Face Landmarker   │   frames of   │ /api/predict                              │
 │   per frame: 28 mouth blendshapes +  │   numbers     │   z-normalise with this user's stats      │
 │   40 lip points (centred, de-rolled, │ ────────────► │   DTW k-NN vs. the user's own samples     │
 │   scaled by eye distance)            │  (no video)   │   → top 5 phrases + confidence            │
 │ hold Space = record                  │               │   Gemini re-rank with bedside context     │
 │ head pitch/yaw → nod / shake         │ ◄──────────── │   (time, last 5 requests, care note)      │
 │ 3 choice cards → speechSynthesis     │   3 choices   │   3 s timeout → offline: classifier top 3 │
 │ brightness check → "bed light"       │               │ /api/log → urgent? → ntfy.sh push         │
 └──────────────────────────────────────┘               └───────────────────────────────────────────┘
```

- **Calibration, about a minute per person.** Each person records each phrase 3 to 5 times. There is no model training.
  Every recording becomes a template, and a new recording is matched against them with dynamic time warping, so the
  speed of mouthing doesn't matter.
- **Only numbers leave the camera.** Face tracking runs in the browser, and the server receives about 108 numbers per frame, never video.
- **Bedside context.** Gemini sees the top 5 lip-reading candidates plus the time, the last 5 requests and an
  optional care note. It reorders them only to break near-ties, and it may only pick from the candidates.
- **Works offline.** If there is no key, no network, or Gemini takes longer than 3 seconds, Unmute uses the
  classifier's top 3 and shows an "offline mode" badge. Speech uses the browser's built-in voices. To load the face
  tracker without internet, run `scripts/fetch_vendor.sh` once.
- **Confirm by nod.** Nod (pitch down and back within 1.5 s) to confirm the highlighted choice. Shake to move to the next
  one. Keys `1` `2` `3`, arrows + `Enter`, and mouse clicks also work.
- **Bed-light check.** If the camera image is too dark, the screen says "Turn on the bed light: too dark to read lips".

## Setup

Requires Python 3.10+ and Chrome.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY=...        # optional; without it Unmute runs in offline mode
.venv/bin/uvicorn server.app:app --reload
```

Open <http://localhost:8000> in Chrome and allow the camera.

Optional environment variables: `GEMINI_MODEL` (by default the newest Flash model the key can use is picked by listing
models), `NTFY_TOPIC` (by default a random topic saved in `data/config.json`), `BED_NAME` (default `Bed 1`).

**Nurse phone alerts.** Install the free **ntfy** app and subscribe to the topic shown on the Nurse page.
ntfy.sh topics are public, so the topic name is random and hard to guess.

## Using it

1. **Calibrate** (`/enroll.html`): type your name, hold Space, mouth the phrase shown, release. Repeat until the bar is
   full. Use the demo set of 10 phrases, or all 20.
2. **Patient** (`/`): enter the same name under Settings, hold Space, mouth a sentence, release, then nod.
3. **Nurse** (`/nurse.html`): live request log, red rows for urgent requests, Acknowledge button. Click
   "Enable alarm sound" once.
4. **Proof** (`/proof.html`): measured accuracy. Or run `.venv/bin/python scripts/evaluate.py`
   (`--sweep` compares classifier settings, `--with-gemini` adds the re-ranked numbers).

## Measured results

Leave-one-out per person: each calibration sample is held out and matched against the same person's other samples.
Run `scripts/evaluate.py` and copy the numbers here. **Only measured numbers go here.**

| Person | Samples | Phrases | Top-1 | Top-3 |
| --- | --- | --- | --- | --- |
| _not measured yet_ | | | | |

## API contract

All JSON. A frame is a flat list of floats: `[mouth blendshapes (dims.bs) | lip x,y pairs (dims.lip)]`.

| Endpoint | Request | Response |
| --- | --- | --- |
| `POST /api/enroll` | `{user, phrase, frames: [[...]], dims: {bs, lip}, fps}` | `{ok, id}` |
| `GET /api/enrollments?user=` | | `{user, counts: {phrase: n}}` |
| `DELETE /api/enrollments/last?user=` | | `{ok, deleted}` |
| `POST /api/predict` | `{user, frames, note, local_time, use_gemini}` | `{candidates: [{phrase, distance, confidence}] (top 5), choices: [3], mode: "gemini"\|"offline", reason, timing}` |
| `POST /api/log` | `{user, sentence, method, mode}` | the stored row, incl. `urgent`, `push` |
| `GET /api/requests` | | `{requests: [...newest first], unacknowledged_urgent}` |
| `POST /api/ack` | `{id}` (or `"all"`) | `{ok}` |
| `GET /api/results` / `POST /api/evaluate` | | leave-one-out results |
| `GET /api/status` | | Gemini availability, ntfy topic, bed |

## Layout

```
server/  app.py (API + static), features.py, classifier.py (DTW k-NN), rerank.py (Gemini), alerts.py (ntfy), evaluation.py
web/     index.html + app.js (patient), enroll.*, nurse.*, proof.*, common.js, style.css
scripts/ evaluate.py, fetch_vendor.sh
data/    phrases.json, enrollments/*.json, requests.jsonl, results.json
```

## Prior art and what's next

SRAVI (Liopa) recognises a fixed list of about 40 phrases and needs Wi-Fi. Unmute learns each patient's own lips,
uses bedside context, and runs offline. The next step is open sentences: a pretrained visual speech recognition
model (Auto-AVSR) whose noisy transcript Gemini corrects into 3 candidate sentences using the same context.
