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

## Demo video: shot list (2 minutes)

Narrate only in "Can see and touch a screen" mode (hold Space to record): in the hands-free modes, talking moves
your lips and starts a recording. Use your own name in Settings. Best phrases for kesh and rishi: I need water,
I am in pain, Call my family, No (rishi: also I cannot breathe; kesh: also Help me).

| Time | Do | Say |
| --- | --- | --- |
| 0:00 | Patient page, your face on camera | "After a tracheostomy, ICU patients can move their lips but can't make a sound, so nurses guess. This is Unmute: it reads silent lips, offline, on this laptop." |
| 0:15 | "Watch." Hold Space, silently mouth **I need water**, let go, press 1 | "Three cards. I pick one, and it's spoken out loud." |
| 0:35 | "Now an emergency." Hold Space, mouth **I am in pain**, let go, press 1. Hold the phone up to the camera | "Urgent requests turn the screen red, sound an alarm and buzz the nurse's phone." |
| 0:55 | Tap the round touch button, tap **Call my family** | "If it's ever wrong, there's always a way out, and every correction teaches it this patient's lips." |
| 1:10 | Say "For a patient who can only blink:", then Settings → **Can only blink**. **Silent from here:** mouth **No**, close your eyes about 1 s on the lit card. Switch back to touch mode before speaking | (silent) |
| 1:30 | Wi-Fi off from the menu bar. Hold Space, mouth **Call my family**, let go, press 1 | "No internet: it still works, and the video never leaves this laptop." |
| 1:45 | Proof page | "Tested on people it never saw, with no setup: the right sentence is in the 3 cards 87% of the time, with an open-source lip reader running on this laptop's GPU. That's Unmute." |

Record with Cmd+Shift+5 → Record Selected Portion (around Brave) → Options → Microphone: MacBook Pro Microphone. Do
2-3 takes, keep the best, trim the ends in QuickTime (Cmd+T). Wi-Fi back on, then YouTube → Upload → Visibility:
Unlisted → paste the link in Devpost. Don't show the Nurse page (its header shows the ntfy topic name).
