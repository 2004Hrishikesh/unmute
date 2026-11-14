# Unmute: competition and prior art

Researched by web search on 14 November 2026. Every product claim has a source link. Where we could not confirm
something it says "unverified". Nothing here is a measurement of Unmute against another product.

**We have NOT measured Unmute against any of these products; no accuracy comparison is claimed.**

## 1. What already exists

### Products and clinical tools

| Name | Who makes it | What it does | How it recognises | Needs network? | Needs calibration? | Cost / availability | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **SRAVI** (Speech Recognition App for the Voice Impaired) | Liopa Ltd, Belfast (Queen's University spin-out) | Phone/tablet app. Patient mouths a phrase from a fixed list ("more than 40 phrases"); app shows text within seconds. Trialled on tracheostomy patients in NHS critical care. | Two engines in the 2023 paper: a speaker-independent deep neural net trained on >500,000 samples from thousands of speakers (75% rank-1, 86% within top 3 on 14 patients) and a DTW template matcher (48% / 74%). | Yes. Needs Wi-Fi or 4G; staff reported poor connectivity. Processing appears to be off-device (moving it to the handset is listed as future work). | A short per-user set-up to locate the lip region. The DTW mode "works best with user-provided training data" but a no-data version was built. | **Company in creditors' voluntary liquidation since 8 April 2024** (deficiency £1.49M). Companies House status: Liquidation. App Store pages (US and GB) returned 404 on 14 Nov 2026. A second QUB study (31 participants) completed April 2024. CE marking: not found, unverified. | [Critical Care 2023 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10332046), [Silicon Republic](https://siliconrepublic.com/?p=1016429), [Irish News](https://ie.irishnews.com/news/business/liquidator-appointed-to-belfast-lip-reading-technology-firm-liopa-WT4PUOU4VZBI3HSYDODEC3AOPU), [Companies House](https://find-and-update.company-information.service.gov.uk/company/NI635074), [NCT06027866](https://clinicaltrials.gov/study/NCT06027866), [App Store (404)](https://apps.apple.com/gb/app/sravi/id1518790556) |
| **Vocable AAC** | WillowTree | Free AAC app (iOS/Android). Pick saved phrases or type on a QWERTY keyboard; the device speaks. For MS, stroke, ALS, spinal injury. | Head tracking (ARKit on iPhone/iPad with TrueDepth camera; ARCore on Android) plus gaze-dwell. Touch also works. No speech or lip recognition. | Not stated in the README (unverified). Tracking runs on-device. | None described (unverified). | Free, open source (MIT). Needs a TrueDepth iOS device for head tracking. | [GitHub README](https://github.com/willowtreeapps/vocable-ios), [App Store](https://apps.apple.com/us/app/-/id1497040547) |
| **TD I-Series** (TD I-13 / I-16) | Tobii Dynavox | Dedicated Windows speech-generating device with built-in eye tracker. Runs TD Snap (symbols) and Communicator 5 (text). | Eye gaze; also "fingers or a switch". Full keyboard, so any sentence. | Not stated; a stand-alone device (offline use unverified). | Yes, eye-gaze calibration (1 to 16 points, usually set up by a specialist). | A third-party guide lists the I-Series at **$11,000+**; Tobii shows no price. Typically funded via insurance (unverified). | [Tobii TD I-Series](https://ca.tobiidynavox.com/pages/TD-i-series), [Tobii support](https://us.tobiidynavox.com/pages/product-support-i-series), [Eye Gaze 101 (prices)](https://undivided.io/resources/eye-gaze-101-3636) |
| **Look to Speak** | Google | Free Android app. Look left, right or up to pick a phrase from an editable list; the phone speaks it. | Eye gestures from the front camera. No special hardware; phone must be mounted. | No: "all of the data is private and never leaves the phone". | Sensitivity adjustment, not a full calibration. | Free, Android 9+. | [Google blog](https://blog.google/outreach-initiatives/accessibility/look-to-speak/) |
| **Project Relate / Euphonia** | Google Research | Android beta app for people with non-standard *speech*: live captions, "Repeat" in a clear synthetic voice, dictation. | Audio speech recognition. User records **500 phrases**, waits "a few days" for a personal model. **Needs the patient's voice, so it does not apply to silent patients.** | Yes for model building (cloud). | Yes, 500 recorded phrases. | Beta, **not accepting new users** (checked 14 Nov 2026). | [Project Relate](https://sites.research.google/relate/), [Euphonia](https://sites.research.google/euphonia/about) |
| **Whispp** | Whispp (Netherlands) | iOS/Android app and phone-call feature that turns whispered or vocal-cord-impaired speech into a natural voice in real time. Can clone the user's own voice. | Audio-to-audio AI. **Needs an audible whisper or rough speech**; not stated to work with silent mouthing. | Yes for real-time calls (unverified for other modes). | Optional: voice cloning from recordings. | Subscription: $9.99/month early-bird at CES 2024. | [TechCrunch](https://techcrunch.com/2024/01/11/whispp-voice-app) |
| **Vidatak EZ Board** | Vidatak LLC | Laminated two-sided board with pictures, words and letters. Patient points. Designed with ventilated patients at UCLA. Single-patient use. | None: the patient points; the nurse reads. | No | No | About **$26** per board. Sold by medical suppliers; English, Spanish, Chinese. | [Instruction sheet](https://vidatak.com/wp-content/uploads/2021/07/VidatakEZBoard_InstructionSheet.pdf), [Pocket Nurse](https://www.pocketnurse.com/default/09-31-1040-vidatak-ez-board-patient-communication-board) |
| **VidaTalk** | Vidatak LLC | Tablet app version of the EZ Board: >100 preset phrases, pain level/location, keyboard, finger drawing, speaks aloud, >10 languages. | Touch. | Not stated (unverified). | No | Pilot: 12 of 13 intubated patients said they would choose it (company press release). Price unverified. | [UCLA Health](https://www.uclahealth.org/news/vidatalk-app-gives-patients-with-communication-challenges), [Press release](https://www.newswire.com/news/new-study-shows-intubated-patients-would-choose-vidatalk-to-help-them-20405733) |
| **Passy Muir Valve** | Passy-Muir Inc. | Speaking valve on the tracheostomy tube (some models in-line with a ventilator). Redirects exhaled air through the vocal folds so the patient speaks with their own voice. | Not software. A clinical device. | No | No; needs a doctor's order and clinical candidacy assessment. | Prescription device. Price unverified. | [Passy-Muir valves](https://www.passy-muir.com/valves_page) |
| **EyeControl** | EyeControl (Israel) | Head-mounted camera and bone-conduction speaker; turns eye gestures into speech for ventilated ICU patients. | Eye-gesture ML on a small processing unit. | Not stated (unverified). | Company claims "under 20 minutes" training (unverified). | Pilot of 15 ICU patients (Beilinson); Emory trial NCT04582149 completed, results not found. Price unverified. | [EU CORDIS](https://cordis.europa.eu/article/id/442638-innovative-platform-helps-acute-care-patients-communicate), [NCT04582149](https://clinicaltrials.gov/study/NCT04582149) |
| **LiRA** (Lip Reading Assistant) | UNC Chapel Hill student startup | Planned smartphone app: lip movements to text and speech for voiceless hospital patients. | ML trained on crowdsourced "selfie" videos (LipTrain). | Unverified | Unverified | Won $25k pitch prizes (2021, 2023); planned user testing by end of 2023. No 2024-2026 news found; status unverified. | [WRAL TechWire 2023](https://wraltechwire.com/2023/08/02/chapel-hill-startup-lira-is-crowdsourcing-selfie-videos-to-improve-tech-for-the-voiceless/) |

### Open-source, research and hackathon projects (what judges may have seen)

| Name | Who | What it does | How it recognises | Network? | Calibration? | Availability | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Two-stage VSR for ICU patients** | RWTH Aachen University Hospital (Sci. Reports, Jan 2023) | Research system; audio-visual dataset recorded in the Aachen ICU with a clinician-picked phrase corpus; two bedside webcams. **6.3% word error rate.** | Deep model: face frames to audio features to text. Speaker-independent. | n/a (research) | No | Paper and dataset description; not a product. | [Nature Sci. Reports](https://www.nature.com/articles/s41598-022-26155-5) |
| **Auto-AVSR** | Imperial College (mpc001) | Open-source visual speech recognition; pretrained models; **~19-20% WER on LRS3** (open sentences). | Deep VSR, speaker-independent. | No (local GPU) | No | Code open; LRS3-trained weights are research-use (per Lipflow's note). This is Unmute's planned open-vocabulary step. | [GitHub](https://github.com/mpc001/auto_avsr) |
| **chaplin** | amanvirparhar | "Reads your lips in real-time and types whatever you silently mouth"; LLM (Ollama) cleans the text. Desktop dictation, not bedside. | Auto-AVSR LRS3 model, fully local. | No | No | MIT, 777 stars. No accuracy figures given. | [GitHub](https://github.com/amanvirparhar/chaplin) |
| **Open-Alterego** | ebowwa | Silent speech to text with a per-speaker fine-tuning workflow; Gradio and HTTP API. | Auto-AVSR plus personal fine-tuning (needs CUDA GPU). | No | Optional fine-tune; "no evaluated personalized checkpoint is included yet". | MIT, 22 stars. | [GitHub](https://github.com/ebowwa/Open-Alterego) |
| **Lipflow** | unknown (macOS) | Mouth silently at the webcam; text appears at the cursor. Apple Silicon VSR. | Deep VSR (LRS3 weights). | No | No | Described as MIT code with non-commercial weights; repo URL not found, unverified. | [AlphaSignal](https://alphasignal.ai/news/lipflow-lets-mac-users-silently-mouth-words-into-any-app) |
| **LipNet** (Keras) | rizkiarm | Sentence-level lip reading on the GRID lab corpus (fixed grammar); 14.19% WER unseen speakers. | Deep model, speaker-independent. | No | No | MIT, 692 stars, pretrained weights. A classic demo; not usable for ICU phrases as-is. | [GitHub](https://github.com/rizkiarm/LipNet) |
| **LipLearner** (CHI 2023) | rkmtlab, iPhone | Users register their own silent commands; on-device, real time. | Few-shot: contrastive pretraining on LRW, then user-registered commands. | No (on-device) | Yes, record keyword and command samples. | MIT; iOS Xcode project. Closest in spirit to "learn the user's own phrases". | [GitHub](https://github.com/rkmtlab/LipLearner) |
| **SilentVoice** (Aug 2026) | Yash Sharda, Hackster | Browser MediaPipe Face Mesh, 40 mouth landmarks per frame, coordinates only to a Flask server, 1D CNN; **7 words** (help, water, pain, doctor, breathe, yes, no); for medical emergencies. | Small CNN trained on 98 samples from one person; ~45% training accuracy. | Server on LAN/tunnel | Trained per author. | MIT, proof of concept. Pipeline very close to Unmute's front end. | [Hackster](https://www.hackster.io/yashsharda2006/silentvoice-d03305) |
| **Ventriloquist** (Meta Global AI Developer Hackathon, Oct-Nov 2026) | solo, ER intern (Indonesia) | Phone on the bed rail asks yes/no questions for ventilated ICU patients; speaks only after the patient confirms; 3 quick YES = alarm; WhatsApp family relay. **Does not read lips.** | Patient picks their own signal (finger, blink, eyebrow); 3 YES + 3 NO template clips; SAM 3.1 tracks it, a Muse Spark "blind judge" double-checks; asks again on disagreement. | Yes for live mode (Meta API); mock mode offline. | Yes, short template calibration. | MIT. "No patient has used it yet." | [GitHub](https://github.com/0xNoramiya/ventriloquist) |
| **Blink Speech** | Devpost entry | Blink patterns and eye gaze for ICU, post-surgery and locked-in patients. | Eye/blink, not lips. | Unverified | Unverified | Devpost page not fetchable; details unverified beyond the blurb. | [Devpost](https://devpost.com/software/blink-speech) |
| **E.C.H.O.** | Makers Asylum (Rhiaan Jhaveri) | ICU communication device: SRAVI lip reading plus large preset buttons (water, food, change position). ~6-month maker project. | SRAVI (see above) plus buttons. | Yes (SRAVI) | Unverified | Maker project page. | [Makers Asylum](https://www.makersasylum.com/project/e-c-h-o/) |
| **Generic lip-reading hackathon entries** | Devpost: Viseme, "Silent Speech Recognition for the Voiceless" (x2), Speakinto Space | Lip video to text, some with TTS; aimed at noisy rooms or voiceless users in general, not ICU. Teams report accuracy as the hard part. | Deep models trained by the team. | Varies | No | Prototypes. | [Viseme](https://devpost.com/software/viseme), [SSR for the Voiceless](https://devpost.com/software/silent-speech-recognition-for-the-voiceless-ktvwyp), [Speakinto Space](https://devpost.com/software/speakinto-space) |

Not found: any hackathon project that combines lip reading, a browser front end, nurse escalation and patient-chosen
confirmation for ICU patients. The two nearest things judges may have seen are **SilentVoice** (same MediaPipe
landmarks-only pipeline, 7 words, one person's data) and **Ventriloquist** (same ICU setting and confirm-before-speak
idea, but yes/no questions instead of lip reading).

## 2. Where Unmute is different

Only differences we can read off our own README are listed. None is an accuracy claim.

- **No setup to start, then personal.** A new patient is matched against the shared starter pool of everyone's
  recordings on the laptop, and every confirmed sentence is saved as a personal template (cap 8 per phrase). Learning
  comes from use, not a recording session. SRAVI's paper does describe a DTW mode that "can learn from the user", so
  per-user learning itself is not new; learning from confirmed sentences during normal use is what we did not find elsewhere.
- **Bedside context that can only reorder.** Gemini sees the time, the last 5 requests and a care note, and may only
  reorder the lip-reading candidates. It can never add a sentence, never drop the lip-reading top match, and never drop
  an urgent phrase from the top 3. None of the products above use situational context in recognition (as far as we found).
- **Confirmation with whatever moves.** Nod, button/switch, long blink, hands-free auto-speak, or carer keys. Tobii
  (eyes, fingers, switch) and Ventriloquist (patient-chosen signal) also offer a choice of input; pairing it with lip
  reading is what we did not find.
- **Offline core path.** No key, no network or a slow Gemini falls back to the classifier's top 3. SRAVI required Wi-Fi
  or 4G and the paper reports poor connectivity on the ward. (Look to Speak, Vocable, chaplin also run locally.)
- **Privacy.** Only about 108 numbers per frame leave the browser; no video. SRAVI's processing appears to be
  off-device; SilentVoice and Look to Speak are also local-first.
- **Urgent escalation to a human.** Red screen, alarm, nurse web page, ntfy phone push, and an early "possible, not yet
  confirmed" alert. We found no commercial product above that pushes to a nurse's phone. Ventriloquist has an alarm and
  family relay.
- **Runs in a browser on any laptop.** No app install, no TrueDepth phone, no $11k device.
- **The only lip-reading product for this group, SRAVI, is in liquidation** (April 2024) and its app is no longer in the
  App Store. There is a gap. (Our README line "the one tool that exists" should read "the one lip-reading product", since
  eye-gaze devices and boards exist and are in daily use.)

## 3. What they do better than us

- **SRAVI** had a speaker-independent neural model trained on >500,000 samples from thousands of speakers, a
  peer-reviewed feasibility study on 14 tracheostomy patients (75% rank-1, 86% top-3), a second completed study with 31
  participants, an NHS deployment at Royal Preston (2021) and Innovate UK funding. Our numbers are from two team members,
  not patients. CE marking: not found, unverified either way.
- **Eye gaze (Tobii, EyeControl, Look to Speak) works with zero mouth movement**, including orally intubated patients,
  who our README says are a poor fit for Unmute. Tobii has decades of AAC practice, mounting, specialist support and
  funding routes.
- **AAC apps (Tobii, Vocable, VidaTalk) have full keyboards and large vocabularies.** The patient can say anything;
  Unmute has 20 phrases plus whatever is added per patient. VidaTalk ships in more than 10 languages.
- **Open-vocabulary lip reading exists** (Auto-AVSR, chaplin, Lipflow; Aachen reached 6.3% WER on an ICU corpus). Unmute
  is a closed phrase set today; open sentences are a roadmap item.
- **Letter boards cost $26, need no power, no camera, no setup, and are already in every ICU.**
- **The Passy Muir Valve gives eligible tracheostomy patients their actual voice.** No software matches that.
- **Whispp and Project Relate give fluent, natural-voice output** for patients who can still make some sound.
- **Ventriloquist** handles uncertainty more carefully (two independent detectors; asks again when they disagree) and
  states plainly that no patient has used it. We should be equally plain.
- **LipLearner** is a published (CHI 2023), on-device, few-shot method; our DTW k-NN is simpler and less tested.

## 4. How to answer "why is this unique?" in 30 seconds

"Three things exist for a patient who can't make a sound: a $26 letter board, which needs a steady hand; an eye-gaze
device at around eleven thousand dollars; and one lip-reading app, SRAVI, which needed Wi-Fi, recognised a fixed list of
about forty phrases, and whose company went into liquidation last year. Unmute runs in a browser on any laptop, works
offline, and sends numbers, not video. A new patient starts talking with no setup from a shared starter pool, and every
sentence they confirm teaches it their own lips. Bedside context can reorder the three choices but never invent one. The
patient confirms with a nod, a switch, or a long blink, and 'I cannot breathe' reaches the nurse's phone. We have not
measured it against any other product, and it is a communication aid, not a medical device."
