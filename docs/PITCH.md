# Unmute: the pitch

*Unmute is a communication aid. It is not a medical device and it does not diagnose anything.*

## The problem

Every day, ICU patients wake up unable to make a sound. A tracheostomy tube, a breathing tube, or surgery on the
voice box takes their voice away, often for days or weeks. They can still move their lips. So they mouth words at
the nurse, and the nurse guesses. "Water?" "Pain?" "Your family?" When the guess is wrong, the patient tries
again, gets tired, and gives up. Letter boards are slow and need a steady hand. Writing needs a free hand and
strength that many patients don't have.

The one commercial tool for this, SRAVI by Liopa, understands a fixed list of about 40 phrases and needs a
network connection. Patients have more to say than 40 fixed phrases, and hospital Wi-Fi is not something you want
between a patient and "I cannot breathe".

## What Unmute does

The patient silently mouths a sentence at an ordinary laptop camera. Unmute reads their lips and shows the three
most likely sentences as big cards. The patient confirms one with a nod, a button press, or a long blink, and the
laptop says it out loud. If the sentence is urgent, the screen turns red, an alarm sounds, the nurse page lights
up, and the nurse's phone buzzes. Unmute learns each patient's own lips in about a minute of calibration, needs no
training data, and works with no internet.

## How it works, in five steps

1. **Face tracking in the browser.** MediaPipe Face Landmarker runs on the laptop. Each frame becomes about 108
   numbers: 40 lip points (centred on the mouth, head tilt removed, scaled by eye distance) and 28 mouth
   blendshapes. Only these numbers go to the server. No video ever leaves the camera.
2. **Matching against the patient's own calibration.** Every calibration recording is a template. A new recording
   is compared with each template by dynamic time warping (DTW), so mouthing fast or slow doesn't matter. The
   phrase with the closest template wins. There is no model training at all.
3. **Bedside context from Gemini.** The top 5 candidates go to Gemini with the time of day, the last 5 confirmed
   requests, and an optional care note ("post-surgery, nil by mouth"). Gemini may reorder the candidates to break
   near-ties. It cannot add a sentence. If it takes more than 3 seconds, or there is no key or no network, Unmute
   uses the lip-reading order and shows an "offline mode" badge.
4. **Confirm with whatever movement the patient has.** Nod to confirm and shake for the next choice. Or, if the
   patient can't move their head: the cards light up one by one, and a press on any button or switch picks the lit
   one. Or a long blink (eyes closed for about 0.8 s) picks it. Carers can press 1, 2, 3.
5. **Speak and escalate.** The browser speaks the sentence. Urgent phrases ("I cannot breathe", "I am in pain",
   "Help me", "Call the nurse", "I feel sick") turn the screen red, sound an alarm, mark the request red on the
   nurse page, and send a push notification to the nurse's phone through ntfy.

## What makes it different

- **A personal lip model in one minute.** Unmute learns *this* patient's lips: swollen, taped, weak, or lopsided
  mouths included. No dataset, no training, no GPU. Add a personal phrase ("Where is Priya?") in a minute.
- **Context that can only reorder, never invent.** Gemini sees the bedside situation but may only choose among the
  phrases the lips produced. The lip-reading top match and any urgent phrase in the top 3 can never be removed.
  The patient always confirms before anything is spoken.
- **Confirmation with whatever the patient can still move.** Head, finger, or eyelid. Each patient picks their
  method in Settings.
- **Works offline on any laptop.** No cloud dependency for the core path. Gemini is a bonus, not a requirement.
- **Privacy by design.** The server receives numbers, not pictures.
- **Urgent requests reach a human.** Red screen, alarm, nurse page, phone push.

### Unmute next to SRAVI (Liopa)

| | SRAVI | Unmute |
| --- | --- | --- |
| Who it understands | Anyone, no setup (speaker-independent) | One patient at a time, after about a minute of calibration (speaker-dependent) |
| Phrases | Fixed list of about 40 | 20 built in, any phrase can be added per patient |
| Unusual mouths (swelling, tape, palsy) | Trained on typical speakers | Learns the patient's own mouth |
| Needs network | Yes | No; Gemini context is optional |
| Bedside context | No | Yes, Gemini re-ranks with time, history, care note |
| Confirm method | App buttons | Nod, button/switch, long blink, keys |
| Urgent escalation | No | Red screen, alarm, nurse page, phone push |
| Runs on | Phone app | Any laptop with a webcam, in a browser |

SRAVI's strength is that it works for a new patient instantly. Unmute's strength is that it fits the patient in
front of it. We do not claim Unmute is more accurate than SRAVI; we have not measured that.

## Measured results

Leave-one-out per person: each calibration recording is held out and matched against the same person's other
recordings. Nothing here is estimated. Measured on 14 November 2026. Chance is 10% top-1 and 30% top-3.

| Person | Samples | Phrases | Top-1 | Top-3 (correct sentence among the 3 cards) |
| --- | --- | --- | --- | --- |
| rishi | 40 | 10 | 75.0% | 95.0% |
| pankaj | 40 | 10 | 20.0% | 52.5% |
| **all** | 80 | 10 | 47.5% | 73.8% |

How the recogniser improved during the hackathon, on the same 80 recordings:

| Change | Top-1 | Top-3 |
| --- | --- | --- |
| First settings | 18.8% | 35.0% |
| + centre each recording on its own mean, add velocity, trim still frames, lip points only | 40.0% | 66.2% |
| + hubness correction (stops short phrases like "Yes" soaking up wrong answers) | 47.5% | 73.8% |

Caveats, said out loud:

- These settings were chosen using these same 80 recordings, so the numbers are optimistic until they are
  re-measured on new recordings.
- The gap between rishi and pankaj is real. pankaj's recordings of the same phrase differ a lot from each other.
  Calibration now warns about this while recording, and a recording where the lips barely moved is rejected.
- Gemini's benefit with real bedside context has not been measured. In the leave-one-out test there is no care note
  and no history, and Gemini kept the lip-reading order in every call it answered. That shows it does not make lip
  reading worse without context.
- A third person, kesh, is still calibrating (26 usable recordings of 13 phrases so far, 1–2 per phrase, which is
  too few to judge).

## Honest limitations

- **Speaker-dependent.** Each patient calibrates. A patient who is already too unwell to calibrate needs someone
  else's recordings, which we have not tested (see roadmap).
- **Best fit is a free mouth.** A patient with a breathing tube *through the mouth* can barely move their lips.
  Unmute suits tracheostomy patients, patients just after extubation, and laryngectomy patients.
- **Consistency matters.** Accuracy depends on mouthing the phrase the same way each time.
- **Short phrases look alike.** "Yes", "No", and "Thank you" are the most confused.
- **Free Gemini quota is small** (as few as 20 requests a day per model). Unmute switches models and falls back
  offline on its own, but the demo may run in offline mode.

## Can we pull training data from SRAVI or others?

No. SRAVI's data is proprietary. Public lip-reading datasets exist (GRID, LRW, LRS2, LRS3), but they use different
vocabularies, need licence agreements, and training a model on them is a multi-day job with GPUs, not a
hackathon-night job. Unmute's design deliberately needs no dataset: the patient's own minute of calibration is the
data.

## Roadmap

1. **Shared starter templates.** Record volunteers with the same features. A new patient starts from that shared
   library and adds one personal sample per phrase. Our features are already normalised (mouth-centred, tilt
   removed, scaled by eye distance), so recordings from different faces are comparable.
2. **Cross-person evaluation.** Calibrate on persons A and B, test on C. This gives a measured number for "works on
   a stranger", next to today's "works on yourself" number.
3. **Open vocabulary.** A pretrained visual speech recognition model (Auto-AVSR) gives a rough transcript; Gemini
   corrects it into three candidate sentences using the same bedside context.
4. **Clinical validation** with speech-language therapists and ICU nurses, on real patients, before any claim about
   care.

## One line to close

SRAVI recognises a fixed list from everyone's lips. Unmute learns *your* lips, understands *your* situation, and
lets you confirm with *whatever movement you still have*.
