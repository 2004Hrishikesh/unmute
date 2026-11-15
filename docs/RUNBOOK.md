# Unmute: the one-page runbook

## The situation (tell it like this)

**Mr. Sharma, 64, Bed 7, ICU, 3 a.m.** He had throat surgery yesterday and has a tracheostomy, a tube in his neck.
His lips work, but no air reaches his voice box, so he cannot make a sound. He is weak; his hands shake.

1. A laptop on the bed table faces him. The nurse set it up yesterday with one question: *"What can this patient
   do?"* → **"Can nod, but not use hands."** He never calibrated. He never touched anything.
2. He silently mouths **"I am in pain."** Recording starts by itself when his lips move and stops when they rest.
3. Three big cards appear: **I am in pain** / I am cold / Thank you. It's urgent, so the nurse's phone already says
   *"Possible: I am in pain (not yet confirmed)."*
4. He nods. The laptop says "I am in pain" out loud. The screen turns red and the nurse station alarms.
5. Next time he mouths "Thank you" and the cards show *Yes / No / I love you*. Wrong. He shakes his head: the
   cards move. Still wrong. The nurse taps the **round touch button**, then **Thank you**. It's spoken, and Unmute
   saves that recording as *his* "Thank you", so next time it's on the first card.

**Who it's for:** patients whose lips move but whose voice doesn't: tracheostomy, after throat or neck surgery, some
stroke and ALS patients. **Not** patients with a breathing tube *in the mouth* (the lips can't move).

**The honest part about a brand-new patient (measured 14 Nov 2026).** With none of his own recordings, the
open-source lip reader plus other people's recordings puts the right sentence on the **first card about 7 times in
10** and **among the 3 cards about 7 times in 8** (71.3% / 87.1%, 240 silent recordings by 2 people it never saw,
20 phrases, chance 5%). Without the lip reader it was 37.9% / 65.0%. When it's wrong: the touch board (him if he can
touch, the nurse or family if not). Every correction is learned; with 2-3 of his own samples of a phrase the
matcher alone has it in the 3 cards ~85-90% of the time.

## What a weak patient uses (Settings → "What can this patient do?")

| Choice | Recording | Picking a card | For |
| --- | --- | --- | --- |
| Can see and touch a screen | tap the button | tap the card, or the touch button for any sentence | most awake patients |
| Can nod, not use hands | **hands-free**: starts when lips move, stops when they rest | nod = yes, shake = next; the first card **speaks itself after 4 s** unless he shakes | weak arms |
| Can press one switch | tap the switch to start/stop | cards light up one by one; press when the right one is lit | one finger, a cheek, a foot |
| Can only blink | hands-free | cards light up one by one; **close both eyes ~1 s** on the right one | locked-in, very weak |

The touch button stays on screen in every mode, for the nurse or family.

## Commands (copy one at a time)

Start the server (leave this terminal open):
```bash
cd /Users/hrishi_xox/Fall_2026/Miscellaneous/project_hackathon
.venv/bin/uvicorn server.app:app --reload --port 8000
```
If it says the port is in use, a server is already running: just use it, or stop it with `lsof -ti tcp:8000 -sTCP:LISTEN | xargs kill` (only the server, not the browser).

In a second terminal, everything before the demo in one go (tests, accuracy, training, lip reader, readiness check):
```bash
cd /Users/hrishi_xox/Fall_2026/Miscellaneous/project_hackathon
bash scripts/before_demo.sh rishi
```
Open **Brave** at <http://localhost:8000>. Top right should say **lip reader on · MPS** and **Gemini context on**
(or "offline mode": also fine).

## Numbers to say (all on the Proof page)

| Say | Number | Where it comes from |
| --- | --- | --- |
| Brand-new patient, no calibration: right sentence in the 3 cards | **87%** (first card 71%) | lip reader + others' recordings, 240 silent recordings, 2 unseen people |
| Lip reader alone, zero-shot | 83% in the 3 cards (67% first) | same |
| Without the lip reader | 65% (38% first) | same recordings |
| After the patient has used it (calibrated) + lip reader | 94% in the 3 cards (78% first) | same |
| Chance with 20 phrases | 5% first card, 15% in the 3 cards | |

Don't say "accurate lip reading of any sentence": free text was exactly right 56/240 times. Say "20 bedside
phrases, plus a bonus free-text card when it's sure".

## Before the demo (re-run any time you record more)

1. More people make every number fairer: Calibrate → name → Phrases: **Volunteer** → hold Space, mouth, let go, 60
   times → **Next volunteer**. Mouth crops are saved for the lip reader.
2. `bash scripts/before_demo.sh rishi`: all 5 steps should finish; step 1 "passed", no "failed".
3. Commit new recordings: `git add data/enrollments && git commit -m "More recordings" && git push`.

## The 4-minute demo

| Time | Do | Say |
| --- | --- | --- |
| 0:00 | Nothing on screen yet | Mr. Sharma's story, two sentences: tube in the neck, lips move, no voice; nurses guess. |
| 0:30 | Patient page. Settings → "Can nod, not use hands". Mouth **"I need water"** | "No button. It starts when my lips move. No calibration for a new patient." Nod. It speaks. |
| 1:15 | Mouth **"I cannot breathe"** and don't confirm | After a 2-second red bar the alarm goes off by itself (if no bar, nod). Teammate holds up the phone (ntfy). "Urgent sentences don't wait for a nod; a misread is cancelled with one tap and the nurse is told." |
| 1:45 | Mouth **"Thank you"** (weak phrase); if wrong, drag the round button, tap it, tap Thank you | "When it's wrong there's always a way out, and the mistake teaches it his lips." |
| 2:30 | Settings → "Can only blink". Mouth **"No"**, close eyes on the lit card | "Whatever he can still move: head, finger, eyelid." |
| 3:00 | Optional: mouth something not on the list, e.g. **"Where is my phone"**, big and slow | Free text is card 1 (dashed, "lip reader heard") only if the lip reader is very sure and no phrase fits; else an extra card or hidden. Free text was exactly right only 56/240 times, so **try it before the demo; if it doesn't show, skip this beat**. Either way say: "An open-source lip reader trained on 3,000 hours of video, offline on this laptop, reads a new patient with no calibration: right sentence in the 3 cards 87% of the time." |
| 3:20 | Proof page | "Every number here is measured, the fair way: tested on a person the model never saw." |
| 3:45 | Close | "Offline, no setup, learns each patient, and gets the nurse when it matters." |

Phrases that worked best for Rishi: *I need water, I cannot breathe, I am in pain, Call my family, No*. Avoid *Yes* and
*Thank you* as the first try (except the deliberate correction beat).

## Tests before the demo (tick each)

Automatic (`bash scripts/before_demo.sh rishi`):
- [ ] Tests: "74 passed" (or more), 0 failed
- [ ] Readiness check: no FAIL lines

By hand, in Brave:
- [ ] Camera: built-in Mac camera picked, teal lip outline on your face, light on your face
- [ ] Badges: "lip reader on · MPS"; "Gemini context on" or "offline mode"
- [ ] Click once on the page (unlocks sound)
- [ ] "I need water" → right card → nod → spoken out loud
- [ ] "I cannot breathe" → red screen + alarm → Nurse page row red → phone buzzes (ntfy subscribed) → "Nurse is here" clears it
- [ ] Touch button: drag it, tap it, tap a sentence → spoken
- [ ] Hands-free: lips move → records by itself; first card speaks after the countdown
- [ ] Long blink picks a lit card
- [ ] Dark: cover the camera halfway / turn the light off → banner + white screen edge; turn the light on → it goes away (or press "Screen light off")
- [ ] Wi-Fi off → still works, badge says "offline mode" (phone push needs internet; the Nurse page doesn't)
- [ ] Nurse page → "Acknowledge all" so the log is clean
- [ ] Proof page shows the numbers you'll quote

## If something goes wrong on stage

| Problem | Fix |
| --- | --- |
| Wrong card | Shake (or press 2/3). Or the touch button. That *is* the feature. |
| No card at all / "too short" | Mouth bigger and slower; in Hold mode hold Space for the whole sentence. |
| Hands-free won't start | Settings → lower "Lip sensitivity", or switch to "Can touch a screen". |
| Black camera | Camera picker under the video → built-in camera → Retry. Close Zoom/FaceTime. |
| No sound | Click the page once, confirm again. |
| "offline mode" | Say "it works without internet; that's the point". |
| Server died | `.venv/bin/uvicorn server.app:app --port 8000`, reload the page. |
