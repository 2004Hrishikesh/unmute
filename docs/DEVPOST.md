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

## About the project

Paste the whole of `docs/devpost/about.txt`. It's raw Markdown in Devpost's template order, with no tables (Devpost
may not render them).

- Fastest: in Terminal, `pbcopy < docs/devpost/about.txt`, then in Devpost's box press Cmd+A, then Cmd+V.
- Don't copy it from a rendered view (GitHub or a Markdown preview): that drops the `##`, `-` and `1.` markers
  Devpost needs.
- After saving, open the project page once to check it. If the math line shows as raw text, delete that line.

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
