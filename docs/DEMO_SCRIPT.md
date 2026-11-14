# Unmute: demo script for the judges

Four minutes. **Presenter: Rishi** (best measured accuracy). One teammate holding a phone with the ntfy app.

**Phrases to use on stage** (Rishi's measured top-1, leave-one-out, 4 samples each): "I need water", "I cannot
breathe", "I am in pain", "Call my family", "No" were right first time 100% of the time. **Avoid** "Thank you" and
"Yes" (25% first time; they get confused with each other and with "No"). Re-check on /proof.html after
re-calibrating.

## Setup checklist, 10 minutes before

Do these in order. Tick each one.

- [ ] Start the server from the project folder:
  ```bash
  .venv/bin/uvicorn server.app:app --port 8000
  ```
- [ ] Run the readiness check. It prints OK / WARN / FAIL per item and the best phrases to demo:
  ```bash
  .venv/bin/python scripts/demo_check.py rishi
  ```
- [ ] Quit Zoom, FaceTime, and anything else that might hold the camera.
- [ ] Open **Brave** at <http://localhost:8000> and **allow the camera**.
- [ ] In the camera picker under the video, pick the **built-in Mac camera**, not the iPhone camera. Wait for the
      teal lip outline on your face.
- [ ] Sit facing a window or a lamp. Light on your face, not behind you. If the yellow "Turn on the bed light"
      banner shows, move.
- [ ] **The presenter is Rishi**, the person with the best measured accuracy: 75% top-1, 95% top-3 on the
      10-phrase demo set with 4 samples per phrase. Re-check <http://localhost:8000/proof.html> after re-calibrating.
- [ ] Open **Settings** under the video. Type the presenter's name. Make sure "How does the patient confirm" is on
      **Nod**. Check the top-right badge:
  - "Gemini context on": great.
  - "offline mode": also fine. You will say "offline mode, same pipeline". Do not try to fix it on stage.
- [ ] Click once anywhere on the page. This unlocks browser audio so speech and the alarm can play.
- [ ] Open the **Nurse** page (<http://localhost:8000/nurse.html>) in a second window or on the phone, and click
      **Enable alarm sound**.
- [ ] On the phone, install the free **ntfy** app and subscribe to the topic shown on the Nurse page.
- [ ] Do one practice run: mouth "I need water", nod, hear it. Then mouth "I cannot breathe", confirm, see red,
      click "Nurse is here" to clear it.
- [ ] On the Nurse page, click "Acknowledge all" so the log starts clean.

## The script

**0:00 — The problem (2 sentences)**

> "Every day, ICU patients wake up with a tube in their throat and no voice. They mouth words, nurses guess, and
> 'I cannot breathe' gets mistaken for 'I need water'. Unmute reads the patient's lips and gives them a voice."

**0:30 — Calibrate page: it learns *your* lips in a minute**

Click **Calibrate**. Show the phrase on screen.

> "There is no training data. The patient mouths each phrase a few times, and that becomes their personal
> model. Watch."

Hold **Space**, mouth the phrase, let go. Point at the green line: "matches your other samples ✓".

> "It even checks each recording against the previous ones, so a sloppy sample gets caught right away."

**1:00 — Patient page: mouth, three cards, nod, speak**

Click **Patient**. Hold Space, silently mouth **"I need water"**, let go.

> "The browser turns my face into 108 numbers per frame. No video leaves this laptop. The server matches them
> against my own recordings and shows the three most likely sentences."

Three cards appear. **Nod**. The laptop speaks "I need water".

> "I nodded to confirm. The patient always confirms before anything is said."

If the badge says Gemini context on, point at the "Context:" line: "Gemini sees the time, the last requests and a
care note, and can reorder these three, but it can never invent a sentence."

**1:45 — Patients who can't nod**

Open **Settings**, change "How does the patient confirm" to **Button / switch**.

> "Many ICU patients can't move their head. So the cards light up one by one, and any button picks the lit one.
> A fifty-dollar assistive switch, or a finger on the space bar."

Hold Space, mouth **"Call my family"**, let go. Cards scan. Press **Space** when the right one is lit. It speaks.

Optional if time allows: switch to **Long blink**, mouth "No", close your eyes for a second on the lit card.

> "Or just the eyelids. Whatever movement the patient still has."

Switch back to **Nod**.

**2:30 — The urgent path**

Hold Space, mouth **"I cannot breathe"**, let go. Confirm it.

The screen turns red and the alarm sounds. Teammate holds up the phone: the ntfy notification arrives. The Nurse
page row is red.

> "Urgent requests don't wait for someone to walk past. Red screen, alarm, nurse station, and the nurse's phone."

Teammate clicks **Acknowledge** on the Nurse page. The alarm stops on the patient screen.

> "When the nurse acknowledges it, the patient sees that help is coming."

**3:15 — Proof page: measured, not promised**

Click **Proof**.

> "These are measured numbers from our own recordings, leave-one-out. Right now the correct sentence is in the
> three cards 95% of the time for me, and [read the overall top-3 number from the Proof page] across all of us.
> Across the hackathon we took the recogniser from
> 19% to 47% top-1 by fixing normalisation, adding lip velocity, and correcting for short phrases that soak up
> wrong answers. These settings were tuned on the same recordings, so we expect lower numbers on new people until
> we re-measure. We only show what we measured."

**3:45 — Close**

> "SRAVI recognises a fixed list from everyone's lips and needs a network. Unmute learns *your* lips in a minute,
> understands *your* situation, works offline on any laptop, and lets you confirm with *whatever movement you still
> have*. Next: shared starter templates so a new patient needs almost no calibration, and open vocabulary with a
> pretrained visual speech model. Thank you."

## Fallbacks: when something goes wrong on stage

| What happens | What you do | What you say |
| --- | --- | --- |
| The right sentence is card 2 or 3, not card 1 | Shake your head to move, then nod. Or press **2** or **3**. | "It's in the top three; a shake moves to the next one." |
| None of the three is right | Nod on "None of these: try again", or press **Esc**. Mouth it again, bigger and slower. | "Lip reading is hard; the patient just tries again. Nothing wrong gets spoken." |
| The nod isn't detected | Press **1**. Or open Settings and lower the nod threshold. | "Carers can also press a key." |
| The camera shows black or "Starting camera…" | Use the camera picker under the video to choose the built-in camera, then click **Retry**. | "The iPhone camera grabbed it; switching." |
| "Face not visible" | Move so your whole face is in the frame and the light is on your face. | |
| The badge says "offline mode" | Nothing. | "We're in offline mode: no Gemini, same pipeline. That's the point: it works without the cloud." |
| Speech doesn't play | Click once on the page, then confirm again. | "Browsers need one click before they'll talk." |
| The alarm doesn't sound on the Nurse page | Click "Enable alarm sound" on that page. | |
| The phone doesn't buzz | Show the red row on the Nurse page instead. | "The push goes through ntfy; the nurse page is the in-hospital path." |
| The server is down | In a terminal: `.venv/bin/uvicorn server.app:app --port 8000`, then reload. | |

## Do not say

- "medical device", "diagnose", "clinically proven".
- "More accurate than SRAVI." We have not measured that.
- Any number that is not on the Proof page.
- "It works for anyone." It works for the person who calibrated.
