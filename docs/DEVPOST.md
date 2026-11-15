# Devpost submission: copy-paste

## Project name
Unmute

## Elevator pitch (the one line under the name)
ICU patients with a tube in the neck can move their lips but can't make a sound. Unmute reads their silent lips with a laptop camera, offline, with no setup, and speaks for them.

## Tracks
- **HackUMass track (exactly one): [HACKUMASS] [MEDTECH] Best MedTech Hack**
- **Sponsor prizes (opt in): [MLH] Best Use of Gemini API**
- Grand Prizes: every project is considered automatically.

## Built with (paste one at a time, 20 tags)
python, fastapi, uvicorn, javascript, html5, css3, mediapipe, pytorch, auto-avsr, espnet, opencv, numpy, sentencepiece, gemini-api, ntfy, web-speech-api, apple-silicon, metal, pytest, httpx

## "Try it out" links
- Code: https://github.com/2004Hrishikesh/unmute (make the repo public first, or judges can't open it)

## Video demo link
YouTube, Unlisted is fine (see the shot list at the bottom of this file).

---

## About the project (paste everything between the lines)

---

## Inspiration

Every day, ICU patients wake up unable to make a sound. A tracheostomy tube in the neck, or surgery on the throat, takes their voice away for days or weeks. **Their lips still move.** So they mouth words at the nurse, and the nurse guesses: *"Water? Pain? Your family?"* When the guess is wrong, the patient tries again, gets tired, and gives up. Letter boards need a steady hand. Writing needs strength they don't have.

We wanted the simplest possible thing: an ordinary laptop on the bed table that **reads their lips and says the sentence out loud**, and gets the nurse when it matters. No setup, no internet, no special hardware.

## What it does

1. **The patient silently mouths a sentence** at the laptop camera. In hands-free mode recording starts when the lips move and stops when they rest; no button.
2. **Three big cards** show the most likely sentences out of 20 bedside phrases ("I need water", "I am in pain", "Call my family", "I cannot breathe", ...).
3. **The patient picks one with whatever they can still move.** The nurse answers one question at setup, *"What can this patient do?"*:
   - **Touch:** tap the card.
   - **Nod:** nod = yes, shake = next card; the first card speaks itself after a countdown.
   - **One switch:** the cards light up one by one, and a press on any button picks the lit one.
   - **Blink:** close both eyes for about a second on the lit card (for locked-in or very weak patients).
4. **The laptop says it out loud.**
5. **Urgent sentences escalate.** "I cannot breathe", "I am in pain", "Help me" and the other urgent phrases turn the screen red, sound an alarm, light up the **Nurse page**, and push to the **nurse's phone**. When the best guess is urgent, the nurse is told *"possible, not yet confirmed"* the moment the cards appear.
6. **When it's wrong there is always a way out.** A movable round touch button opens every sentence, for the patient or the nurse or family. Every correction is saved as the patient's own sample, so Unmute learns *their* lips.
7. **Something not on the list?** The lip reader's own free-text reading appears as an extra card, but only when it is sure, and it is never spoken without a confirm.
8. **Dark rooms at 3 a.m.:** the screen becomes a soft light on the face, so the camera can still see the lips.

## How we built it

```
camera ─► MediaPipe Face Landmarker (in the browser)
            ├─► 108 numbers per frame (40 lip points + 28 mouth blendshapes)
            └─► small mouth picture per frame (only while the lip reader is on)
                      │  localhost only (127.0.0.1, same laptop)
                      ▼
          FastAPI server on the laptop
            ├─► Matcher: dynamic time warping k-NN over every recording, own + other people's
            ├─► Auto-AVSR lip reader (250M params, 3,291 h of video) on the laptop GPU (PyTorch, Apple MPS)
            ├─► blend ─► top 5 ─► Gemini reorders by bedside context (optional, 3 s budget)
            ▼
   3 cards ─► nod / switch / blink / touch ─► speech + Nurse page + phone push (ntfy)
```

- **Face tracking in the browser.** MediaPipe Face Landmarker turns each frame into about 108 numbers: lip points centred on the mouth, head tilt removed, scaled by eye distance, plus 28 mouth blendshapes.
- **Matcher, no training.** Every recording is a template. A new one is compared with all of them by **dynamic time warping**, so mouthing fast or slow doesn't matter, with a hubness correction so a few "average" templates don't win everything. The patient's own templates win naturally once they exist.
- **Open-source lip reader.** We run [Auto-AVSR](https://github.com/mpc001/auto_avsr) (Imperial College London, Apache 2.0) **offline on the laptop's GPU**. Instead of asking it for free text, we ask it how likely **each phrase on the list** is: teacher-forced decoder plus CTC, about 0.1-0.3 s. That turned out to be far more reliable than free text. We blend the two in log space:

$$\log p(\text{phrase}) = w \,\log p_{\text{lip reader}} + (1-w)\,\log p_{\text{matcher}}$$

  with the weight \\( w \\) measured on our own silent mouthing.
- **Gemini as a careful second opinion.** Gemini gets the top 5 candidates, the time of day, the last 5 requests and an optional care note ("post-surgery, nil by mouth"), and may reorder them to break near-ties, as JSON. **Safety rules are enforced in code:** it can only choose from the candidates, and it can never remove the top lip-reading match or an urgent phrase. In testing it once dropped "I need water" because water had "just been given"; that can't happen any more. If it takes more than 3 s, or there's no network, Unmute uses the lip-reading order and shows "offline mode". It rotates across the Flash models the key can use, and rests one that is over quota.
- **Accessibility inputs:** nod and shake detection from head pose, a long-blink detector from the eye blendshapes, switch scanning, and a hands-free start/stop from lip motion.
- **Nurse side:** a Nurse page with a red/green request log and alarm, and **ntfy** push notifications to a phone.
- **76 automated tests**, including accuracy floors on our real recordings and a **privacy test** that captures everything sent to Gemini and ntfy and checks that it's text only.

## Measured, not guessed

Every number below is on the Proof page and re-computed by one script. A "brand-new patient" means **people the system never saw**, with no calibration, and the matcher only has *other* people's recordings. There are 20 phrases, so chance is 5% for the first card and 15% for the 3 cards.

| Brand-new patient, silent mouthing (240 recordings, 2 unseen people) | First card | In the 3 cards |
| --- | --- | --- |
| Matcher alone (other people's recordings) | 37.9% | 65.0% |
| Open-source lip reader alone, zero-shot | 66.7% | 83.3% |
| **Lip reader + matcher (what runs)** | **71.3%** | **87.1%** |
| After the patient has used it (own samples) + lip reader | 78.3% | 93.8% |

The lip reader takes about 0.7 s per sentence on an M5 Pro laptop, including the free-text reading; the cards appear about 1-2 s after the lips stop.

## Privacy: the video never leaves the laptop

- The server listens on **127.0.0.1 only**: nothing on the network can reach it.
- The browser sends it numbers per frame, plus small mouth pictures while the lip reader is on. They are processed in memory and **never saved** for a patient.
- Only **two things may go to the internet, both optional and both text:** the candidate sentences and bedside context to Gemini, and *"Bed 7: I am in pain"* to the nurse's phone. **A test captures both and checks they hold no picture, no face numbers and no name.**
- **With Wi-Fi off everything still works:** lip reading, cards, speech, alarm and Nurse page. Only the phone push needs a network, and ntfy is open source and can run on the hospital's own network (one setting: `NTFY_URL`).

## Challenges we ran into

- **Silent mouthing is not speech.** The lip reader learned from people talking aloud, and nobody could tell us whether it would read *silent* mouthing. So we measured it on 240 of our own silent recordings: 83.3% in the 3 cards zero-shot, about 5.5 times chance.
- **Our own number was wrong, and we caught it.** The first rule for free text was "show it first when it is more likely than any list phrase". Measured on real mouthing, it would have put a wrong sentence on card 1 in **124 of 240** recordings, because its reading is usually a near-miss ("called my family"). The new rule hides near-misses and only goes first when no phrase fits: **4 of 240**.
- **A model that looked great, until more people.** We trained our own small neural network on volunteers' recordings, with heavy augmentation and an ensemble. With 2 people, tested leave-one-person-out, it beat the matcher. With 4 people it didn't, so our training script **switched it off by itself**. We only ship what measures better.
- **People are different.** One volunteer's own recordings matched each other far less well than anyone else's (43.8% on the first card, against 69-81% for the others). We now report per person and keep honest accuracy floors in the tests.
- **Weak patients.** A patient may not be able to calibrate, press or even nod. That is why there are four input modes, a hands-free countdown, and a touch board that the nurse can use too.
- **Engineering.** The lip reader's CTC scorer didn't run on Apple GPUs (a one-line device fix). Our pairwise DTW was O(N²) in memory (now block-wise, 1 GB at 300 recordings). And a dark ICU room meant using the screen as a light.

## Accomplishments that we're proud of

- A brand-new patient, with **no setup at all**, gets the right sentence in the 3 cards **87% of the time** (out of 20 phrases).
- It is **fully offline** on an ordinary laptop: a 250-million-parameter lip reader running on the laptop's own GPU.
- **Anyone can use it, whatever they can still move:** touch, nod, one switch, or a blink.
- **Every claim is backed by a test or a measured number**, including the ones that told us we were wrong.

## What we learned

- **Test on people the model has never seen.** Testing on the same person flattered every number we had.
- **Asking a big model "which of these 20?" beats asking it "what did they say?"** Constrained scoring was much more reliable than free text.
- Accessibility is mostly about the last step: picking a card is harder for a weak patient than the AI is for us.
- Put the LLM where it is safe: reordering, with hard rules in code, never inventing.

## What's next

- Test with speech-language pathologists and real tracheostomy patients.
- More volunteers and other languages, plus fine-tuning the lip reader on silent mouthing.
- Sentences the care team adds per patient.
- Integration with hospital nurse-call systems.
- A tablet build for the bed rail.

*Unmute is a communication aid. It is not a medical device and does not diagnose anything.*

---

## Image gallery (in docs/gallery/, 3:2, upload in this order)

1. `1-cover.jpg`: title card (the thumbnail)
2. `2-proof.jpg`: measured accuracy, brand-new patient (live Proof page)
3. `6-how.jpg`: how it works, and what stays on the laptop
4. `4-touch.jpg`: the touch board (live app)
5. Optional: a still of the Patient page with the 3 cards, taken from your demo video or with Cmd+Shift+4 in Brave.
   Don't upload the Nurse page as it is: it shows the ntfy topic name, which would let anyone read or send alerts.

## Demo video: shot list (aim for 2 minutes, record with Cmd+Shift+5, microphone on)

| Time | Show | Say |
| --- | --- | --- |
| 0:00 | Cover image, or your face | "After throat surgery or a tracheostomy, ICU patients can move their lips but can't make a sound. Nurses guess. Unmute reads their lips." |
| 0:15 | Patient page, Settings → "Can nod, not use hands". Mouth **"I need water"**, nod | "No button: it starts when my lips move. No calibration. I nod, and it speaks." |
| 0:40 | Mouth **"I cannot breathe"**, nod. Show the red screen, the Nurse page and the phone | "Urgent sentences sound an alarm and buzz the nurse's phone, even before the patient confirms." |
| 1:00 | Mouth "Thank you". If it's wrong, drag the round button, tap it, tap the sentence | "When it's wrong, there's always a way out, and the correction teaches it this patient's lips." |
| 1:20 | Settings → "Can only blink". Mouth **"No"**, close your eyes on the lit card | "Touch, nod, one switch, or a blink: whatever the patient can still move." |
| 1:35 | Proof page | "Tested on people it never saw, with no setup: the right sentence is in the 3 cards 87% of the time. An open-source lip reader runs on this laptop's GPU." |
| 1:50 | Turn Wi-Fi off, mouth one more | "No internet needed. The video never leaves the laptop." |
| 2:00 | Cover | "Unmute." |

Upload to YouTube as **Unlisted**, then paste the link into "Video demo link".
