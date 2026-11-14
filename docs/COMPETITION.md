# Unmute next to what already exists

Researched 14 November 2026 from primary sources where we could reach them (papers, trial registries, company
and GitHub pages). Every external fact has its URL. "Could not verify" means we only saw it in a search snippet
or a secondary article. Unmute's own numbers are from `data/results.json` / the Proof page (see README.md and
docs/PITCH.md). **We have not measured Unmute against any of these products, so we never say we are more accurate.**

## 1. The field at a glance

| Solution | What it is | Input | Who it serves | Setup per patient | Needs network? | Cost (if known) | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **SRAVI** (Liopa, Belfast) | Phrase lip-reading app showing 3 ranked phrases | Lips | Tracheostomy / voice-impaired ICU patients | None for the DTW method ("without the user having to provide training data"); app may keep data to adapt to the user | Not stated in the paper; app-based | Not sold to individuals: hospitals contacted Liopa (listing now 404) | [Critical Care 2023 paper](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1186/s13054-023-04420-x&resultType=core&format=json), [trial NCT06027866](https://clinicaltrials.gov/api/v2/studies/NCT06027866) |
| **Liopa (company)** | SRAVI's maker; Queen's University Belfast spin-out | – | – | – | – | In creditors' voluntary liquidation since 8 April 2024 | [The Gazette](https://www.thegazette.co.uk/notice/4599675), [Irish News](https://ie.irishnews.com/news/business/liquidator-appointed-to-belfast-lip-reading-technology-firm-liopa-WT4PUOU4VZBI3HSYDODEC3AOPU) |
| **Two-stage VSR for ICU** (Uni Hospital Aachen, Sci Rep 2023) | Research lip reader: silent video → predicted audio features → text | Lips | ICU patients (intended); tested on healthy adults only | Pooled model, no per-patient step described | No (research code, not released) | Not a product; authors co-founded Clinomic GmbH | [PMC9844948](https://pmc.ncbi.nlm.nih.gov/articles/PMC9844948/) |
| **Greek multimodal lip reading** (Computers 2022) | Personalised RNN trained on video recorded *before* voice loss | Lips (+ audio for training labels) | Tracheostomy patients | Pre-loss video of the patient | Not stated | Research | [Crossref record](https://api.crossref.org/works/10.3390/computers11030034) |
| **Auto-AVSR** (open source) | Sentence-level lip-reading model, 250M params | Lips | Research / developers | None (trained on 3,291 h of public video) | No, but needs a capable machine | Free (Apache 2.0) | [GitHub](https://github.com/mpc001/auto_avsr) |
| **Chaplin** (open source) | Desktop tool: types what you silently mouth, Auto-AVSR + local LLM | Lips | Anyone with a camera | None | No ("runs fully locally") | Free (MIT) | [GitHub](https://github.com/amanvirparhar/chaplin) |
| **SilentVoice** (Hackster prototype) | Browser MediaPipe mouth landmarks → Flask → small CNN, 7 words | Lips | Emergencies (proof of concept) | Trained on the author's own recordings; others record 10–20 samples per word and retrain | Phone access via a Cloudflare tunnel | Free (MIT) | [Hackster](https://www.hackster.io/yashsharda2006/silentvoice-d03305) |
| **Tobii Dynavox I-Series** | Eye-gaze speech-generating device | Eyes / touch | AAC users incl. hospitals | Eye calibration, mounting | No | I-110 listed at $5,999 (2023 directory); "price on application" | [Closing The Gap directory](https://online.flippingbook.com/view/353809829/80/), [Living Made Easy](https://livingmadeeasy.org.uk/product/tobii-dynavox-i-series-communication-aid) |
| **Eyegaze Edge** (LC Technologies) | Eye-gaze communication system, ICU screens available | Eyes | AAC users incl. ICU | Eye calibration, mounting | No | $8,700–$11,150 (2023 directory) | [Closing The Gap directory](https://online.flippingbook.com/view/353809829/69/), [eyegaze.com](https://eyegaze.com/) |
| **EyeControl** (Israel) | Wearable, screenless eye-movement device with audio feedback | Eyes | Ventilated ICU patients | "Average training time under 20 minutes" (company claim) | Not stated | Not public | [Trial NCT04582149](https://clinicaltrials.gov/study/NCT04582149), [CORDIS](https://cordis.europa.eu/article/id/442638-innovative-platform-helps-acute-care-patients-communicate) |
| **VidaTalk** (Vidatak) | Tablet communication app: icons, speech-generated messages, drawing, keyboard; from the EZ Board | Touch | Intubated / non-vocal ICU patients | Pick from icons | Not stated | Not public (hospital licence) | [PMC10833611](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10833611/), [trial protocol](https://cdn.clinicaltrials.gov/large-docs/76/NCT02921776/Prot_SAP_002.pdf) |
| **Vocable AAC** (WillowTree) | Free open-source AAC app, head tracking or touch | Head pose / touch | People who cannot speak | None | No | Free (MIT) | [GitHub](https://github.com/willowtreeapps/vocable-ios), [App Store](https://apps.apple.com/us/app/-/id1497040547) |
| **AlterEgo** (MIT Media Lab / startup) | Wearable that reads jaw/face EMG for silent speech | EMG | Research; consumer launch 2025 | User-specific calibration | Not stated | Not public | [2018 paper slides](https://www.slideshare.net/slideshow/alterego-a-personalized-wearable-silent-speech-interface/93236141), [TechCrunch](https://techcrunch.com/?p=1618117) |
| **Google Project Relate / Euphonia** | Personalised speech recognition for atypical *voiced* speech | Voice | People who can speak but are hard to understand | Record 500 phrases, model ready days later | Yes (Android app) | Free beta | [Euphonia](https://sites.research.google/euphonia/about), [9to5Google](https://9to5google.com/2021/11/09/google-project-relate-app/) |

## 2. The six that matter most

**SRAVI (Liopa).** The closest product to Unmute, and the one judges may know. In the peer-reviewed feasibility
study (14 tracheostomy patients, 616 recordings, 516 phrase-identifiable) the app showed 3 ranked phrases; the
deep-network method scored 75% at rank 1 and 86% across the 3 ranks, the DTW method 48% / 74%. In the follow-up
trial of 31 patients (NCT06027866, results posted September 2025) 135 of 468 videos returned the correct phrase,
about 29%. SRAVI is registered with the MHRA and CE marked, runs on a phone or tablet camera, and was only
available through hospitals. Liopa entered creditors' voluntary liquidation on 8 April 2024. SRAVI has a fixed
phrase list (Liopa said "more than 40": could not verify from a primary source), no bedside context, no
nod/switch/blink confirmation, and no nurse escalation. Note that SRAVI's own DTW method is the same family of
algorithm as Unmute's, and its real-patient numbers (48% rank 1, 74% top 3) are in the same range as ours on
healthy team members. That is sobering, and honest to say.

**Two-stage VSR for ICU (Aachen).** The strongest academic result: WER 6.3% on 238 ICU sentences, about 6,900
videos. But it was recorded with 6 young healthy adults, "did not involve any patients", the same speakers appear
in train and test, no code or product was released, and real-time operation was not demonstrated. It has no
confirmation step, no context, no alerting. Unmute is a working bedside tool; this is a model.

**Auto-AVSR / Chaplin.** Open-vocabulary lip reading exists as open source: Auto-AVSR reports 20.3% WER on LRS3
(a 250M-parameter model trained on 3,291 hours), and Chaplin wraps it into a local "types what you mouth" tool
with an LLM clean-up step. Neither is built for patients: no fixed safe phrase list, no confirmation before
speaking, no escalation, and a heavy model. This is exactly Unmute's roadmap item 3, which is why we cite it as
"what's next", not as a competitor we beat.

**SilentVoice (Hackster).** The closest hackathon-style project to our architecture: MediaPipe mouth landmarks in
the browser, numbers (not video) to a Flask server, a small CNN over 30-frame windows. It knows 7 words, was
trained on one author's recordings, reports about 45% training accuracy with validation between 10% and 40% on
98 samples, and has no confirmation, no context, no nurse alert, and no offline story. Unmute shares the privacy
idea and goes much further: starter model across people, learning from use, DTW with no training, Gemini context,
three confirm methods, escalation, offline.

**Eye-gaze devices (Tobii Dynavox, Eyegaze Edge, EyeControl).** These work even when the mouth cannot move at
all, which is a real advantage over any lip reader. They cost thousands of dollars ($5,999 to $11,150 in 2023
directories), need mounting and eye calibration, and are selection tools: the patient picks from a grid, letter
by letter or icon by icon, rather than just saying the sentence. EyeControl removes the screen but still needs
"under 20 minutes" of training per its maker. Unmute runs on the laptop that is already in the room.

**VidaTalk and Vocable AAC.** Touch or head-pose picking from boards. Vocable is free and open source and uses
ARKit head tracking; VidaTalk is validated in ICU usability studies (12 of 13 intubated patients would use it
again, manufacturer-reported). Both need the patient to navigate menus. Neither reads lips, so a patient who can
only mouth words and cannot lift a hand or hold a head pose cannot use them; Unmute's long-blink mode is for
exactly that patient.

## 3. Unique claims we can defend

1. **Starts with no setup and becomes personal from use.** Matches against a shared pool of everyone's
   recordings, and every confirmed sentence is saved as a personal sample. Measured cross-person: 69% top-1 /
   94% top-3 for one team member on 10 demo phrases, 35% / 60% for another on 20 phrases. *No competitor found
   does both:* SRAVI adapts per user per its listing (could not verify beyond a search snippet); SilentVoice needs
   retraining; eye-gaze needs calibration. **Verified** for all except the SRAVI adaptation detail.
2. **Bedside context that can only reorder, never invent.** Gemini sees time, last requests and a care note, may
   only choose among the lip-reading candidates, and cannot drop the top match or an urgent phrase. *No other
   solution in the table uses any context.* **Verified** (none of the sources mention it).
3. **Confirm with whatever movement the patient has: nod, any switch, or a long blink.** SRAVI and VidaTalk use
   on-screen buttons; Vocable uses head pose or touch; eye-gaze needs eyes on a screen. *No lip-reading product
   offers scanning plus switch or blink.* **Verified** for the sources read.
4. **Urgent phrases reach a human: red screen, alarm, nurse page, phone push.** *None of the lip-reading or AAC
   sources describe escalation.* **Verified** (absence in sources; a hospital nurse-call integration may exist for
   eye-gaze devices that we did not find).
5. **Privacy: only about 108 numbers per frame leave the browser, never video.** SilentVoice does the same (so
   not unique to us); SRAVI's data handling is not described in the paper. **Verified** for the comparison.
6. **Fully offline on any laptop, free software.** Chaplin and Vocable are also free and local; eye-gaze devices
   cost thousands; SRAVI was hospital-only. **Verified.**
7. **Measured and shown, with caveats, on a Proof page.** Our personal-model best is 75% top-1 / 95% top-3 (one
   person, 4 samples per phrase, settings tuned on the same data). SRAVI's published numbers are 48–75% rank 1
   on real patients. We do not claim to beat them; we claim we show our numbers the same way. **Verified.**

## 4. Claims to avoid

- "More accurate than SRAVI" or than anything else. Not measured. SRAVI's real-patient 48% / 74% (DTW) is the
  same range as ours on healthy people.
- "Speaker-independent" or "works for anyone". Our starter model is two team members; the measured range for a
  new person is 35–69% top-1.
- "The only lip-reading tool for ICU." SRAVI exists (CE marked, trialled on 31 patients), and Aachen built one.
- "SRAVI needs the cloud / Wi-Fi." The paper does not say where processing runs. Could not verify.
- "SRAVI only knows 40 phrases." The ">40" figure is from press, not a primary source.
- "Clinically validated", "medical device", "diagnoses". Unmute is a communication aid tested on three
  team members.
- Any Aachen-style WER number for Unmute. We do phrase accuracy, not word error rate.

## 5. Where competitors beat us

- **Trained on thousands of speakers.** Auto-AVSR (3,291 hours) and SRAVI's deep network generalise far beyond
  our two-person starter pool. SRAVI's DNN reached 75% rank 1 on real patients; our cross-person number for a
  new person is 35–69% on healthy team members.
- **Real patients, regulators, trials.** SRAVI: 14-patient feasibility study, 31-patient trial, MHRA registration,
  CE mark. VidaTalk: ICU usability studies and a randomised trial. We have three team members and no patients.
- **Works with no mouth movement at all.** Eye-gaze devices and EyeControl serve patients who are intubated
  through the mouth, paralysed, or sedated-but-aware. Lip reading cannot.
- **Open vocabulary.** Auto-AVSR and Chaplin read any sentence; Unmute reads 20 phrases plus personal additions.
- **Hardware polish.** Tobii and Eyegaze ship mounts, batteries, and support contracts. We ship a browser tab.
- **Independent accuracy evidence.** Every number we have was measured by us on us, with settings chosen on the
  same recordings.
