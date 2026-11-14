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
up, and the nurse's phone buzzes. A new patient starts from a shared starter model with no setup; every sentence
they confirm teaches Unmute their own lips. It needs no training data and works with no internet.

## How it works, in five steps

1. **Face tracking in the browser.** MediaPipe Face Landmarker runs on the laptop. Each frame becomes about 108
   numbers: 40 lip points (centred on the mouth, head tilt removed, scaled by eye distance) and 28 mouth
   blendshapes. Only these numbers go to the server. No video ever leaves the camera.
2. **Matching against recordings.** Every recording on the laptop, the patient's own and other people's (the shared
   starter), is a template. A new recording is compared with each template by dynamic time warping (DTW), so
   mouthing fast or slow doesn't matter. The phrase with the closest template wins; the patient's own templates win
   naturally once they exist, and every confirmed sentence becomes one. There is no model training at all.
3. **Bedside context from Gemini.** The top 5 candidates go to Gemini with the time of day, the last 5 confirmed
   requests, and an optional care note ("post-surgery, nil by mouth"). Gemini may reorder the candidates to break
   near-ties. It cannot add a sentence. If it takes more than 3 seconds, or there is no key or no network, Unmute
   uses the lip-reading order and shows an "offline mode" badge.
4. **Confirm with whatever movement the patient has, or none.** Nod to confirm and shake for the next choice. Or,
   if the patient can't move their head: the cards light up one by one, and a press on any button or switch picks
   the lit one. Or a long blink (eyes closed for about 0.8 s) picks it. Or hands-free: recording starts when the
   lips move and stops when they rest, and the top card is spoken by itself after a short countdown unless the
   patient objects or mouths again. Carers can press 1, 2, 3.
5. **Speak and escalate.** The browser speaks the sentence. Urgent phrases ("I cannot breathe", "I am in pain",
   "Help me", "Call the nurse", "I feel sick") turn the screen red, sound an alarm, mark the request red on the
   nurse page, and send a push notification to the nurse's phone through ntfy. When the best guess is urgent, the
   nurse is told "possible, not yet confirmed" the moment the cards appear; the confirmation follows seconds later.

## Does the patient have to calibrate? No.

This was our biggest worry: a sick patient can't be asked to record 40 phrases. So Unmute now has three ways in,
and all three are measured on our own recordings (Proof page, "Shared starter model"):

1. **Zero calibration: the shared starter model.** Unmute matches against *everyone's* recordings on the laptop,
   not just the patient's own. A brand-new person gets the right sentence first time 69% and in the three cards
   94% of the time when matched only against another team member's recordings of the 10 demo phrases (kesh, 16
   recordings; chance 10% / 30%). The other way round, against a 20-phrase set, it is 35% / 60% (rishi, 40
   recordings; chance 5% / 15%). Not perfect, but the patient can start talking in the first minute, with no setup.
2. **It learns from use.** Every sentence the patient confirms (nod, button, blink) is saved as a personal sample of
   that phrase. No calibration session, just use. In a measured experiment, the starter model plus 2 personal samples
   per phrase reached 68% top-1 and 98% top-3 on rishi's remaining recordings; 1 personal sample gave 58% / 86%.
   Adding other people's recordings to a calibrated person's own did not hurt (rishi: 75% / 95% own only,
   75% / 97.5% pooled).
3. **Quick start when a nurse has two minutes.** The Calibrate page has "Quick start: 5 urgent phrases + Yes/No":
   7 phrases, about 20 recordings. The nurse can hold the button ("tap to start, tap to stop") while the patient
   mouths. Planned surgery and tracheostomy patients can calibrate the day before, while they still can, the way
   voice banking works for ALS.

## What makes it different

- **Starts with no setup, then becomes personal.** Day one: the shared starter model. Every confirmed sentence is a
  personal sample, so Unmute learns *this* patient's lips (swollen, taped, weak, or lopsided mouths included) just
  from being used. No dataset, no training, no GPU. Add a personal phrase ("Where is Priya?") in a minute.
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
| Who it understands | Anyone, no setup (speaker-independent) | Anyone from the shared starter, then adapts to the patient from every confirmed sentence |
| Phrases | Fixed list of about 40 | 20 built in, any phrase can be added per patient |
| Unusual mouths (swelling, tape, palsy) | Trained on typical speakers | Learns the patient's own mouth |
| Needs network | Yes | No; Gemini context is optional |
| Bedside context | No | Yes, Gemini re-ranks with time, history, care note |
| Confirm method | App buttons | Nod, button/switch, long blink, keys |
| Urgent escalation | No | Red screen, alarm, nurse page, phone push |
| Runs on | Phone app | Any laptop with a webcam, in a browser |

SRAVI's strength is a model trained on many speakers. Unmute's strength is that it fits the patient in front of
it, and gets better the more they use it. We do not claim Unmute is more accurate than SRAVI; we have not measured that.

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
  too few to judge his personal model). His recordings are what the starter-model numbers above were measured with.
- The "starter + 2 personal samples" numbers come from one person (rishi) with one other person's recordings as
  the starter. More people are needed before that is a general claim.

## Honest limitations

- **The starter model is small.** It is built from the people who have used this laptop (two team members today).
  Its measured accuracy for a new person ranges from 35% to 69% top-1 depending on who recorded what. More
  volunteers make it better; that is a recording session, not a research project.
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

1. **A bigger starter library.** Record more volunteers (nurses, students) with the same features. The starter
   model and the cross-person measurement already exist; more people make the number go up.
2. **Measure learning from use on real patients**: how many confirmed sentences until the personal model beats
   the starter.
3. **Open vocabulary.** A pretrained visual speech recognition model (Auto-AVSR) gives a rough transcript; Gemini
   corrects it into three candidate sentences using the same bedside context.
4. **Clinical validation** with speech-language therapists and ICU nurses, on real patients, before any claim about
   care.

## One line to close

SRAVI recognises a fixed list from everyone's lips. Unmute starts talking with no setup, learns *your* lips from
every sentence you say, understands *your* situation, and lets you confirm with *whatever movement you still have*.
