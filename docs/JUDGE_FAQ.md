# Unmute: judge FAQ

Short answers to the questions judges are most likely to ask. Every number here is measured and is on the Proof
page; anything we have not measured says so.

**1. How does it work without training a model?**
Every calibration recording becomes a template: a short sequence of lip shapes. A new recording is compared with
each template using dynamic time warping, which lines up two sequences even if one was mouthed faster. The phrase
with the closest template wins. Nothing is trained; adding a phrase is just recording it a few times.

**2. Why DTW and not a neural network?**
A neural lip reader needs thousands of speakers and days of GPU time, and still struggles with unusual mouths.
DTW nearest-neighbour needs one patient and one minute. It runs on a laptop CPU in a few milliseconds per
prediction. For a bedside tool with 10 to 20 phrases per patient, that trade is worth it.

**3. A sick patient can't sit through calibration. So how does this work in real life?**
They don't have to. A new patient starts on the shared starter model: Unmute matches against every recording on the
laptop, not just their own. Measured: the right sentence was in the three cards 94% of the time for a team member
matched only against another member's recordings of the 10 demo phrases (69% first time; chance 30% / 10%).
Then every sentence the patient confirms is saved as a personal sample, so the model becomes theirs just from use:
starter plus 2 personal samples per phrase measured 98% top-3. A nurse can also do a 2-minute "quick start" of the
5 urgent phrases plus Yes/No, holding the button for the patient.

**4. How is it better than SRAVI?**
We don't claim it is more accurate; we have not measured that. It is different: it starts with no setup and then
learns the patient's own lips from use, phrases can be added per patient, bedside context can only reorder,
confirmation is by nod, button, or blink, it works offline, and urgent requests reach the nurse's phone. SRAVI has
a fixed list of about 40 phrases, one model for everyone, and needs a connection.

**5. What is the accuracy, and how did you measure it?**
Leave-one-out per person: each calibration recording is held out and matched against that person's other
recordings. On 80 recordings from two people, 10 phrases each: 47.5% top-1 and 73.8% top-3 overall; the best
person is at 75% top-1 and 95% top-3. Chance is 10% and 30%. Measured 14 November 2026, shown on the Proof page.

**6. Why only 47.5%?**
Two reasons. First, the settings were tuned on these same 80 recordings, so even that number is optimistic.
Second, one person's recordings of the same phrase vary a lot; the same recogniser gets 75% on the other person.
Calibration now warns when a recording doesn't match the previous ones. The number that matters at the bedside is
top-3, because the patient picks from three cards: 74% overall, 95% for the best person.

**7. What about patients who can't nod, or can't press anything at all?**
Settings offers three confirm methods. Nod and shake. Button or switch: the cards light up one by one and a press
on any button or an assistive switch picks the lit one. Long blink: eyes closed for about 0.8 seconds picks the
lit card. And a hands-free mode: recording starts when the lips move and stops when they rest, and the top card is
spoken by itself after a countdown unless the patient shakes, presses, or mouths again. If the best guess is
urgent, the nurse is alerted immediately as "possible", before confirmation.

**8. What about privacy? It's a camera in an ICU.**
Face tracking runs inside the browser. The server receives about 108 numbers per frame (lip point positions and
mouth shape values), never an image. The request log stores sentences and times, not video or numbers.

**9. What if there is no internet?**
The core path, mouth to cards to speech, runs on the laptop. Gemini is optional: without a key, a network, or an
answer within 3 seconds, Unmute uses the lip-reading order and shows an "offline mode" badge. The face tracker
files can be downloaded once with `scripts/fetch_vendor.sh` so even the first load needs no internet.

**10. What does Gemini actually add? Can it hallucinate a sentence?**
It gets the top 5 lip-reading candidates plus the time, the last 5 confirmed requests, and a care note, and may
reorder them to break near-ties. It cannot add a sentence: every choice is validated against the candidate list,
and the lip-reading top match and any urgent phrase in the top 3 can never be dropped. In our test it answered 28
calls and never changed the order, which is right when there is no context; its benefit with real context is not
measured yet.

**11. How long does calibration take?**
Zero if the laptop already has other people's recordings: the patient just starts, and confirmed sentences become
their calibration. Quick start (5 urgent phrases + Yes/No, 3 each) is about two minutes. The full 10-phrase demo
set at 6 samples each is about six minutes. More samples measured better: 2 per phrase 51% top-1, 3 gave 64%,
4 gave 75%.

**12. How would you get more data and make it generalise?**
Not from SRAVI; their data is proprietary. Public datasets (GRID, LRW, LRS2/LRS3) have different vocabularies and
need licences. Our route is already built and measured: the shared starter library grows with every volunteer who
records on the laptop (features are normalised for face size and tilt, so different faces are comparable), and the
cross-person number on the Proof page shows how good it is for a stranger. Patients having planned surgery can
record the day before, the way voice banking works.

**13. Is this a medical device?**
No. Unmute is a communication aid. It does not diagnose, treat, or monitor anything. The patient confirms every
sentence before it is spoken, and urgent requests go to a human nurse.

**14. What does it cost to deploy?**
A laptop with a webcam and a browser. The software is free: Python, FastAPI, MediaPipe, and ntfy. Gemini is
optional and has a free tier. An assistive switch for patients who can't nod costs tens of dollars.

**15. What is next?**
A bigger starter library from volunteers; measuring how many confirmed sentences a real patient needs before their
personal model beats the starter; open vocabulary using a pretrained visual speech model (Auto-AVSR) whose rough
transcript Gemini corrects into candidates; and validation with speech-language therapists and ICU nurses before
any claim about care.
