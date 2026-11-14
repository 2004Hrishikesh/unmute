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

- **No calibration needed to start.** Unmute matches against every recording on the laptop (the shared starter
  model), not only the patient's own. Measured cross-person on the Proof page: a new person matched only against
  another team member's recordings of the 10 demo phrases got 69% top-1 / 94% top-3 (chance 10% / 30%); against a
  20-phrase set, 35% / 60% (chance 5% / 15%).
- **It learns from use.** Every confirmed sentence is saved as a personal sample of that phrase (Settings → "Learn
  from every confirmed sentence", at most 8 per phrase). Measured: starter plus 2 personal samples per phrase gave
  68% top-1 / 98% top-3 on rishi's other recordings; pooling other people's recordings with a calibrated person's own
  did not hurt (75% / 95% own only, 75% / 97.5% pooled).
- **Calibration, when someone has a minute.** "Quick start" is the 5 urgent phrases plus Yes/No (about 20 recordings);
  the demo set is 10 phrases. There is no model training. Every recording becomes a template, and a new recording is
  matched against them with dynamic time warping, so the speed of mouthing doesn't matter.
- **Only numbers leave the camera.** Face tracking runs in the browser, and the server receives about 108 numbers per frame, never video.
- **Bedside context.** Gemini sees the top 5 lip-reading candidates plus the time, the last 5 requests and an
  optional care note. It reorders them only to break near-ties, and it may only pick from the candidates.
  **Safety rule, enforced in code:** Gemini can never remove the lip-reading top match, or an urgent phrase in the
  lip-reading top 3, from the 3 choices. In testing it once dropped "I need water" because water had "just been
  given"; that can no longer happen.
- **Gemini model choice.** The server lists the Flash models your key can use (this costs no quota) and prefers
  the fast "lite" ones: we measured about 0.5–1.3 s per call against 1.3–7 s for the full models, which were
  often overloaded. When a model refuses, it rests: until its quota resets on 429 (free keys get as few as 20
  requests a day per model), 60 s on 503, and a week on 404 (retired). The next model is tried within the same
  3 s, and resting times are remembered in `data/gemini_state.json`.
- **Works offline.** If there is no key, no network, or Gemini takes longer than 3 seconds, Unmute uses the
  classifier's top 3 and shows an "offline mode" badge. Speech uses the browser's built-in voices. To load the face
  tracker without internet, run `scripts/fetch_vendor.sh` once (22 MB into `web/vendor/`, gitignored); the page then
  makes no requests to the internet at all, which we checked in the browser's network log.
- **Confirm with whatever movement the patient has** (Settings → "How does the patient confirm"):
  - **Nod**: pitch down and back within 1.5 s confirms; shake moves to the next choice.
  - **Button / switch**: the choices light up one by one; one press picks the lit one. Any USB button or
    assistive "jelly bean" switch that types Space or Enter works, with no driver.
  - **Long blink**: the same scanning; closing both eyes for about 0.8 s picks. Normal blinks are ignored.
  - Recording can be "hold while mouthing", "tap to start, tap to stop", or **hands-free**: it starts when the lips
    start moving and stops when they have been still for 0.7 s (thresholds measured from our own recordings: a still
    face sits near 0.0004 lip-change per frame, mouthing peaks at 0.003–0.006). In hands-free mode the first card can
    speak by itself after a countdown (default 4 s, shown as a bar); a shake, a long blink, or any key cancels it, and a
    nod confirms sooner. The patient never has to touch anything.
  - Carers can always use keys `1` `2` `3`, arrows + `Enter`, or a mouse click.
- **Calibration quality check.** After each sample, Unmute checks whether it looks like the person's other samples of
  the same phrase and asks them to redo it if not. Recordings where the lips barely moved are rejected.
- **Pre-alert before confirmation.** The moment the best guess is an urgent phrase, the nurse page and phone get a
  "Possible (not yet confirmed)" alert (Settings toggle, on by default). The sentence the patient then confirms
  supersedes it; a false alarm costs the nurse a glance, a missed "I cannot breathe" costs more.
- **Bed-light check.** If the camera image is too dark, the screen says "Turn on the bed light: too dark to read lips".

## Setup

Requires Python 3.10+ and Chrome.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env            # optional: paste GEMINI_API_KEY=... into .env (gitignored)
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
4. **Before a demo**: `.venv/bin/python scripts/demo_check.py <name>` prints OK / WARN / FAIL for the server, Gemini,
   offline files, the presenter's calibration, the best measured phrases to show, and the nurse log.
5. **Proof** (`/proof.html`): measured accuracy. Or run `.venv/bin/python scripts/evaluate.py`
   (`--sweep` compares classifier settings, `--with-gemini` adds the re-ranked numbers).

## Measured results

Leave-one-out per person: each calibration sample is held out and matched against the same person's other samples.
Run `scripts/evaluate.py` and copy the numbers here. **Only measured numbers go here.**

| Person | Samples | Phrases | Top-1 | Top-3 |
| --- | --- | --- | --- | --- |
| rishi | 40 | 10 | 75.0% | 95.0% |
| pankaj | 40 | 10 | 20.0% | 52.5% |
| **all** | 80 | 10 | 47.5% | 73.8% |

Later the same day, kesh re-recorded all 20 phrases, 5 samples each, same phrase in a row:

| Person | Samples | Phrases | Top-1 | Top-3 | Chance |
| --- | --- | --- | --- | --- | --- |
| kesh | 102 | 20 | 81.4% | 91.2% | 5% / 15% |

Shared starter model, same data (Proof page): kesh matched only against rishi's recordings of the 10 demo phrases,
no calibration: 70.6% top-1 / 90.2% top-3 (51 samples). rishi matched only against kesh's 20 phrases: 42.5% / 75.0%
(chance 5% / 15%). As deployed (pooled, each sample held out): kesh 77.5% / 92.2%, rishi 70.0% / 97.5%.

Chance is 10% top-1 and 30% top-3. Measured on 14 November 2026 with the default `Config` in
`server/classifier.py`. The first settings scored 18.8% top-1 and 35.0% top-3 on the same data. Measured step by step:

| Change | Top-1 | Top-3 |
| --- | --- | --- |
| First settings | 18.8% | 35.0% |
| + centre each recording on its own mean, add velocity, trim still frames, lip points only | 40.0% | 66.2% |
| + hubness correction (stops short phrases like "Yes" soaking up wrong answers) | 47.5% | 73.8% |

These settings were picked using these same 80 samples, so the numbers are optimistic until they are re-measured
on new recordings. pankaj's samples of the same phrase differed a lot from each other; his recordings were deleted
on 14 November for re-calibration, so the Proof page now shows the current recordings, not this table.

**More samples per phrase help, measured.** Random subsets of rishi's recordings, leave-one-out:
2 samples per phrase 51% top-1 / 70% top-3; 3 samples 64% / 92%; 4 samples 75% / 95%. The calibration default is
now 6 samples per phrase, recorded in a row, which has not been measured yet.

**When one person's samples don't match each other.** For one calibrating team member, two recordings of the same
phrase were on average as far apart as recordings of different phrases (distance ratio 1.00, against 0.79 for
rishi). No feature setting we tried (smoothing, no velocity, per-recording scaling, longer resampling, wider DTW
band) changed that. The fix is in the recording, not the maths: mouth each sample the same way, keep the head still
and facing the camera, and record the samples of a phrase one after another.

**Gemini in this test.** Leave-one-out has no bedside context: no care note and no recent requests. In an earlier
run, before the hubness correction, Gemini answered 28 of the 80 calls before the free quota ran out (the rest fell
back offline). In all 28 it kept the classifier's order, so accuracy did not change. That shows it does not make
lip reading worse without context. Its benefit with real context has not been measured.

## API contract

All JSON. A frame is a flat list of floats: `[mouth blendshapes (dims.bs) | lip x,y pairs (dims.lip)]`.

| Endpoint | Request | Response |
| --- | --- | --- |
| `POST /api/enroll` | `{user, phrase, frames: [[...]], dims: {bs, lip}, fps}` | `{ok, id, check: {matches, in_top3, looks_like}}`; 422 if the lips barely moved |
| `GET /api/enrollments?user=` | | `{user, counts, own_samples, own_from_use, personal, starter_available, starter_samples, starter_users}` |
| `DELETE /api/enrollments/last?user=` | | `{ok, deleted}` |
| `POST /api/predict` | `{user, frames, note, local_time, use_gemini}` | `{candidates: [{phrase, distance, confidence}] (top 5), choices: [3], mode: "gemini"\|"offline", reason, source: "personal"\|"starter", timing}` |
| `POST /api/log` | `{user, sentence, method, mode, learn?, frames?, dims?, fps?}` | the stored row, incl. `urgent`, `push`, `learned` |
| `GET /api/requests` | | `{requests: [...newest first], unacknowledged_urgent}` |
| `POST /api/ack` | `{id}` (or `"all"`) | `{ok}` |
| `GET /api/results` / `POST /api/evaluate` | | leave-one-out results |
| `GET /api/status` | | Gemini availability, ntfy topic, bed |

## Layout

```
server/  app.py (API + static), features.py, classifier.py (DTW k-NN), rerank.py (Gemini), alerts.py (ntfy), evaluation.py
web/     index.html + app.js (patient), enroll.*, nurse.*, proof.*, common.js, style.css
scripts/ evaluate.py, demo_check.py, fetch_vendor.sh
data/    phrases.json, enrollments/*.json, requests.jsonl, results.json
```

## Prior art and what's next

See [docs/COMPETITION.md](docs/COMPETITION.md) for the researched landscape with sources (SRAVI, eye-gaze devices,
AAC apps, open-source lip readers, other hackathon projects), [docs/PITCH.md](docs/PITCH.md) for the story,
[docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) for the 4-minute demo, and [docs/JUDGE_FAQ.md](docs/JUDGE_FAQ.md).

SRAVI (Liopa) recognises a fixed list of about 40 phrases and needs Wi-Fi. Unmute learns each patient's own lips,
uses bedside context, and runs offline. The next step is open sentences: a pretrained visual speech recognition
model (Auto-AVSR) whose noisy transcript Gemini corrects into 3 candidate sentences using the same context.
