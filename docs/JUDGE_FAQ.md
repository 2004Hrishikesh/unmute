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
laptop, not just their own. Measured: the right sentence was in the three cards 90.2% of the time for a team member
matched only against another member's recordings of the 10 demo phrases (70.6% first time; chance 30% / 10%).
Then every sentence the patient confirms is saved as a personal sample, so the model becomes theirs just from use:
starter plus 2 personal samples per phrase measured 98% top-3. A nurse can also do a 2-minute "quick start" of the
5 urgent phrases plus Yes/No, holding the button for the patient.

**4. How is it better than SRAVI?**
We don't claim it is more accurate; we have not measured that. It is different: it starts with no setup and then
learns the patient's own lips from use, phrases can be added per patient, bedside context can only reorder,
confirmation is by nod, button, or blink, it works offline, and urgent requests reach the nurse's phone. SRAVI has
a fixed list of about 40 phrases, one model for everyone, and needs a connection.

**5. What is the accuracy, and how did you measure it?**
Leave-one-out: each recording is held out and matched against the others. kesh, all 20 phrases, 5–6 samples each:
81.4% top-1 and 91.2% top-3 (chance 5% / 15%). rishi, 10 phrases, 4 each: 75.0% / 95.0% (chance 10% / 30%).
Overall 79.6% / 92.2% on 142 recordings. Measured 14 November 2026, live on the Proof page.

**6. Why not higher, and what moved the number?**
Earlier today a first 80-recording set scored 47.5% top-1, and one person's recordings of the same phrase varied
a lot. The recogniser settings were tuned on that set (so that number was optimistic), the Calibrate page was
changed to record the samples of a phrase in a row and to warn when one looks like a different phrase, and kesh's
102 new recordings under those rules measure 81.4% top-1. The number that matters at the bedside is top-3, because
the patient picks from three cards: 92.2% overall.

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

**16. What if the lip reading picks the wrong sentence?**

Nothing is spoken until the patient confirms, so a wrong card is never said by itself. If none of the three is right,
the patient taps the floating touch button (it can be dragged anywhere the hand reaches, like a phone's assistive-touch
button) and taps the right sentence on a board of big buttons, urgent ones first. That sentence is spoken *and* the
recording is saved as a sample of it, so the mistake teaches the model. If they already confirmed the wrong card, the
board offers "Said the wrong thing?": the recording is moved to the right phrase instead of being learned wrong. The
Proof page shows live accuracy from the request log: first card, in the three cards, or fixed on the board.

**17. What happens in the dark?**

The camera needs some light on the face; there is no infrared camera in a laptop. Three things happen when the image
gets too dark: a banner says so; the screen edge turns into a warm white ring light (at about 50 cm a laptop screen
lights a face usefully); and the face tracker gets a brightened copy of the image (the camera feed itself is not
changed). The light stays on until the room is clearly bright without it, so it doesn't flicker. And the touch board
needs no light at all, so the patient can always talk. Honest limit: we have not measured accuracy in a dark room yet.

**18. How many recordings does a patient need?**

Zero to start (the shared starter model), and accuracy climbs with use. Replayed on our recordings, each predicted
before it is learned: 0 own samples of a phrase 55% first card / 85% in the three cards; 2 own samples 80% / 93%;
4 or more 91% / 100%. Urgent phrases do better: as deployed they are in the three cards 97.3% of the time (37
recordings). More people recording more phrases makes the zero-sample start better; the matcher's memory now grows
linearly, so thousands of recordings are fine.
