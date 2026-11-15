// Patient view: mouth -> 3 choices -> nod to confirm -> speak (+ nurse alert if urgent).
import {
  $, api, settings, createLandmarker, CameraControl, FaceTracker, drawMouth, PushToRecord,
  GestureDetector, LongBlink, AutoRecorder, tapMode, BrightnessCheck, speak, alarm, unlockAudio, beep, localTimeString,
} from "/common.js";

const video = $("#video"), overlay = $("#overlay"), cam = $("#cam");
const statusEl = $("#status"), subEl = $("#substatus"), choicesEl = $("#choices");
const hintEl = $("#hint"), reasonEl = $("#reason"), spokenEl = $("#spoken"), camMsg = $("#camMsg");
const recordBtn = $("#recordBtn");

const RETRY = "None of these: try again";
const MIN_FRAMES = 8;

let state = "loading";
let choices = [], confidences = {}, selected = 0, lastMode = null, choiceShownAt = 0;
let calibrated = false, calInfo = null, lastRec = null, URGENT = [];
let lastFrame = null;
let urgentId = null, urgentPoll = null;
let spokenTimer = null;
// For the touch board: the recording just confirmed, what was learned from it, and its log row, so a wrong
// confirmation can be fixed (relabelled) instead of teaching the model the wrong phrase.
let lastLearnedId = null, lastLogId = null, lastConfirmAt = 0, PHRASES = [];
let atOpen = false;
const FIX_WINDOW_MS = 60000;

// ---------------------------------------------------------------- settings UI

const gest = new GestureDetector({ pitchT: settings.get("pitchT", 8), yawT: settings.get("yawT", 12) });
function bindRange(id, key, dflt, apply) {
  const el = $("#" + id), val = $("#" + id + "Val");
  el.value = settings.get(key, dflt); val.textContent = el.value; apply(+el.value);
  el.addEventListener("input", () => { val.textContent = el.value; settings.set(key, +el.value); apply(+el.value); });
}
bindRange("pitchT", "pitchT", 8, (v) => (gest.pitchT = v));
bindRange("yawT", "yawT", 12, (v) => (gest.yawT = v));
let darkT = 55;
bindRange("darkT", "darkT", 55, (v) => (darkT = v));

// Confirm modes: "nod" (head), "switch" (one big button: choices are scanned, press to pick),
// "blink" (choices are scanned, close both eyes ~1 s to pick). Switch and blink need no head movement.
const blinker = new LongBlink({ holdMs: 800 });
let scanMs = 2500;
bindRange("scanS", "scanS", 2.5, (v) => (scanMs = v * 1000));
bindRange("blinkS", "blinkS", 0.8, (v) => (blinker.holdMs = v * 1000));
const confirmSel = $("#confirmMode"), recordSel = $("#recordMode");
confirmSel.value = settings.get("confirmMode", "nod");
recordSel.value = settings.get("recordMode", "hold");
const mode = () => confirmSel.value;
function applyMode() {
  document.querySelectorAll(".auto-only").forEach((e) => e.classList.toggle("hidden", !autoMode()));
  document.querySelectorAll(".nod-only").forEach((e) => e.classList.toggle("hidden", mode() !== "nod" || (e.classList.contains("auto-only") && !autoMode())));
  document.querySelectorAll(".scan-only").forEach((e) => e.classList.toggle("hidden", mode() === "nod"));
  document.querySelectorAll(".blink-only").forEach((e) => e.classList.toggle("hidden", mode() !== "blink"));
  if (state === "choosing") { startScan(); startAutoConfirm(); renderChoices(); } else if (["idle", "spoken"].includes(state)) setState("idle", { msg: idleMsg() });
}
confirmSel.addEventListener("change", () => { settings.set("confirmMode", confirmSel.value); applyMode(); });
recordSel.addEventListener("change", () => { settings.set("recordMode", recordSel.value); applyMode(); });
const autoMode = () => recordSel.value === "auto";

// Hands-free: the lips starting to move begins the recording; stillness ends it. Then the first card can
// speak by itself after a countdown unless the patient shakes, blinks long, presses anything, or nods sooner.
let autoConfirmMs = 4000;
bindRange("autoConfirmS", "autoConfirmS", 4, (v) => (autoConfirmMs = v * 1000));
const auto = new AutoRecorder({
  enabled: () => autoMode() && calibrated && !atOpen && ["idle", "spoken"].includes(state),
  onStart: (preroll) => { ptr.begin(); for (const f of preroll) ptr.push(f); },
  onStop: ({ tooShort, lostFace }) => {
    if (!ptr.recording) return;
    if (tooShort || lostFace) { ptr.cancel(); setState("idle", { msg: lostFace ? "I lost your face. Face the camera." : idleMsg() }); return; }
    ptr.end();
  },
});
bindRange("sens", "sens", 6, (v) => (auto.sensitivity = v));
let autoTimer = null;
function startAutoConfirm() {
  stopAutoConfirm();
  if (!autoMode() || mode() !== "nod" || autoConfirmMs <= 0 || state !== "choosing") return;
  autoTimer = setTimeout(() => { if (state === "choosing") confirm(selected, "auto"); }, autoConfirmMs);
}
function stopAutoConfirm() { clearTimeout(autoTimer); autoTimer = null; }

let scanTimer = null;
function startScan() {
  stopScan();
  if (mode() === "nod") return;
  const step = () => {
    if (state !== "choosing") return;
    if (!blinker.closed) moveSelection(1, true); // don't move away while the eyes are closing on a choice
    scanTimer = setTimeout(step, scanMs);
  };
  scanTimer = setTimeout(step, scanMs * 1.4); // a little longer on the first (most likely) choice
}
function stopScan() { clearTimeout(scanTimer); scanTimer = null; }
function recordHint() {
  if (autoMode()) return "Hands-free: just mouth your sentence. Recording starts and stops by itself.";
  return tapMode()
    ? "Press <kbd>Space</kbd> / the button once, mouth the sentence, press again."
    : "Hold <kbd>Space</kbd> (or the button) while you mouth, then let go.";
}

const userInput = $("#userInput"), noteInput = $("#noteInput"), geminiToggle = $("#geminiToggle");
userInput.value = settings.user;
noteInput.value = settings.note;
geminiToggle.checked = settings.get("useGemini", true);
userInput.addEventListener("change", () => { settings.user = userInput.value.trim().toLowerCase(); refreshUser(); });
noteInput.addEventListener("change", () => (settings.note = noteInput.value));
geminiToggle.addEventListener("change", () => settings.set("useGemini", geminiToggle.checked));
const learnToggle = $("#learnToggle");
learnToggle.checked = settings.get("learn", true);
learnToggle.addEventListener("change", () => settings.set("learn", learnToggle.checked));
const preAlertToggle = $("#preAlertToggle");
preAlertToggle.checked = settings.get("preAlert", true);
preAlertToggle.addEventListener("change", () => settings.set("preAlert", preAlertToggle.checked));

// Before the patient confirms: if the best guess is urgent, the nurse is told now, marked "possible".
// The confirmed sentence that follows supersedes it on the nurse page. A false alarm costs a glance;
// a missed "I cannot breathe" costs more.
async function preAlert(sentence) {
  if (!preAlertToggle.checked || !URGENT.includes(sentence)) return;
  try {
    await api("/api/log", { body: { user: settings.user, sentence, method: "possible", mode: lastMode, confirmed: false, learn: false } });
    if (state === "choosing") reasonEl.textContent = `Nurse pre-alerted: possible "${sentence}", not confirmed yet. ` + reasonEl.textContent;
  } catch (e) { console.warn("pre-alert failed", e); }
}

async function refreshUser() {
  const u = settings.user;
  $("#userBadge").textContent = u ? `user: ${u}` : "no user";
  calibrated = false;
  if (!u) { $("#settings").open = true; setState("nouser"); return; }
  try {
    calInfo = await api(`/api/enrollments?user=${encodeURIComponent(u)}`);
    // No calibration needed when other people's recordings exist: that is the shared starter model.
    calibrated = calInfo.personal || calInfo.starter_available;
  } catch {}
  setState(calibrated ? "idle" : "uncalibrated", { msg: idleMsg() });
}

function idleMsg() {
  if (!calInfo) return "";
  if (!calInfo.personal) {
    return `No calibration needed: starter model from ${calInfo.starter_samples} recordings by ${calInfo.starter_users} other `
      + `${calInfo.starter_users === 1 ? "person" : "people"}. Unmute learns from every sentence you confirm.`;
  }
  return calInfo.own_from_use ? `Your model: ${calInfo.own_samples} samples, ${calInfo.own_from_use} learned from use.` : "";
}

async function refreshStatus() {
  try {
    const s = await api("/api/status");
    const b = $("#modeBadge");
    if (lastMode === "offline" || !s.gemini.available || !geminiToggle.checked) {
      b.textContent = "offline mode"; b.className = "badge offline";
      b.title = s.gemini.error || "Gemini off: using the on-device classifier only";
    } else {
      b.textContent = "Gemini context on"; b.className = "badge ok"; b.title = s.gemini.model || "";
    }
  } catch {
    const b = $("#modeBadge"); b.textContent = "server offline"; b.className = "badge offline";
  }
}

// ---------------------------------------------------------------- state + rendering

function setState(s, extra = {}) {
  state = s;
  if (s !== "choosing") { stopScan(); stopAutoConfirm(); }
  cam.classList.toggle("recording", s === "recording");
  $("#recDot").classList.toggle("hidden", s !== "recording");
  recordBtn.classList.toggle("on", s === "recording");
  recordBtn.disabled = !["idle", "choosing", "spoken", "recording"].includes(s);
  spokenEl.classList.add("hidden");
  if (s !== "choosing") choicesEl.innerHTML = "";
  reasonEl.textContent = "";
  hintEl.innerHTML = "";

  if (s === "loading") { statusEl.textContent = "Loading…"; subEl.textContent = extra.msg || ""; }
  if (s === "nouser") { statusEl.textContent = "Who is using Unmute?"; subEl.textContent = "Type a name in Settings (left), then calibrate."; }
  if (s === "uncalibrated") {
    statusEl.textContent = "Calibration needed";
    subEl.innerHTML = `Nobody has calibrated on this laptop yet. <a href="/enroll.html">Open Calibrate</a> (about a minute).`;
  }
  if (s === "idle") {
    statusEl.textContent = autoMode() ? "Just mouth your sentence" : "Mouth your sentence now";
    subEl.textContent = extra.msg || "";
    hintEl.innerHTML = recordHint();
  }
  if (s === "recording") {
    statusEl.textContent = "Recording…";
    subEl.textContent = autoMode() ? "Keep going; it stops when your lips stop." : tapMode() ? "Mouth the sentence clearly, then press again." : "Mouth the sentence clearly, then let go.";
  }
  if (s === "thinking") { statusEl.textContent = "Reading your lips…"; subEl.textContent = ""; }
  if (s === "choosing") renderChoices();
  if (s === "spoken") {
    statusEl.textContent = "Said out loud:";
    subEl.textContent = "";
    spokenEl.textContent = extra.sentence; spokenEl.classList.remove("hidden");
    hintEl.innerHTML = recordHint().replace("while you mouth", "to say something else");
  }
  if (s === "error") { statusEl.textContent = "Something went wrong"; subEl.textContent = extra.msg || ""; }
}

function renderChoices() {
  statusEl.textContent = "Did you say…";
  subEl.textContent = {
    nod: autoMode() && autoConfirmMs > 0 ? "Nod to confirm now · shake for the next one · or wait and the first card speaks by itself" : "Nod to confirm · shake your head for the next one",
    switch: "Press the button when the right one lights up",
    blink: "Close your eyes for a moment when the right one lights up",
  }[mode()];
  const opts = choices.concat([RETRY]);
  choicesEl.innerHTML = "";
  opts.forEach((c, i) => {
    const b = document.createElement("button");
    b.className = "choice" + (i === selected ? " selected" : "") + (c === RETRY ? " retry" : "");
    if (c === RETRY) b.textContent = c;
    else {
      b.innerHTML = `<span class="num">${i + 1}</span><span></span>`;
      b.children[1].textContent = c;
    }
    const countdown = mode() !== "nod" ? scanMs * (performance.now() - choiceShownAt < 300 ? 1.4 : 1)
      : (autoMode() && autoConfirmMs > 0 ? autoConfirmMs : 0);
    if (i === selected && countdown > 0) {
      const bar = document.createElement("span");
      bar.className = "scanbar";
      bar.style.animationDuration = `${countdown}ms`;
      b.appendChild(bar);
    }
    b.addEventListener("click", () => confirm(i, "click"));
    choicesEl.appendChild(b);
  });
  hintEl.innerHTML = mode() === "switch"
    ? "Switch = <kbd>Space</kbd> or <kbd>Enter</kbd> (most USB buttons type one of these). Carers can also press <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd>."
    : "Or press <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd>, arrows + <kbd>Enter</kbd>, <kbd>Esc</kbd> to retry.";
}

function moveSelection(delta, auto = false) {
  const n = choices.length + 1;
  selected = (selected + delta + n) % n;
  beep(auto ? 520 : 660, 60, auto ? 0.06 : 0.12);
  renderChoices();
  if (!auto) startScan(); // a manual move restarts the scan clock
  startAutoConfirm(); // and the hands-free countdown
}

// ---------------------------------------------------------------- record -> predict

const ptr = new PushToRecord({
  button: recordBtn,
  // In switch mode the same button picks a choice, so it must not start a recording while choosing.
  enabled: () => calibrated && !atOpen && (["idle", "spoken"].includes(state) || (state === "choosing" && mode() === "nod") || state === "recording"),
  onStart: () => { speechSynthesis.cancel(); clearTimeout(spokenTimer); lastLearnedId = null; lastLogId = null; setState("recording"); },
  onStop: (rec) => onRecorded(rec),
});

async function onRecorded(rec) {
  if (rec.frames.length < MIN_FRAMES) {
    const why = rec.missed > rec.frames.length ? "I could not see your face. Face the camera." : "Too short. Hold Space for the whole sentence.";
    setState("idle", { msg: why });
    return;
  }
  setState("thinking");
  lastRec = rec;
  try {
    const r = await api("/api/predict", {
      body: {
        user: settings.user, frames: rec.frames, dims: rec.dims, note: settings.note,
        local_time: localTimeString(), use_gemini: geminiToggle.checked,
      },
    });
    lastMode = r.mode;
    refreshStatus();
    choices = r.choices;
    confidences = Object.fromEntries(r.candidates.map((c) => [c.phrase, c.confidence]));
    selected = 0;
    choiceShownAt = performance.now();
    gest.reset(); blinker.reset();
    setState("choosing");
    startScan();
    startAutoConfirm();
    if (r.mode === "gemini" && r.reason) reasonEl.textContent = "Context: " + r.reason;
    if (r.mode === "offline" && r.reason) reasonEl.textContent = "Offline mode: " + r.reason;
    if (r.source === "starter") reasonEl.textContent = (r.recogniser === "model+matcher"
      ? "Model trained on many people (no calibration yet). " : "Starter model (no calibration yet). ") + reasonEl.textContent;
    preAlert(choices[0]);
    renderDebug(r);
  } catch (e) {
    setState("idle", { msg: "Could not read that: " + e.message });
  }
}

async function confirm(i, method) {
  if (state !== "choosing") return;
  const opts = choices.concat([RETRY]);
  const sentence = opts[i];
  if (sentence === undefined) return;
  if (sentence === RETRY) { setState("idle", { msg: "OK, try again." }); nudgeTouch(); return; }
  await sayAndLog(sentence, method, i);
}

// Speak a sentence the patient chose and log it. rank = its position among the cards (0-2), or -1 if it
// was not on the cards (picked on the touch board). The recording is learned under this sentence.
async function sayAndLog(sentence, method, rank, extra = {}) {
  speak(sentence);
  setState("spoken", { sentence });
  clearTimeout(spokenTimer);
  spokenTimer = setTimeout(() => state === "spoken" && setState("idle", { msg: idleMsg() }), 6000);
  lastConfirmAt = Date.now();
  try {
    const learn = $("#learnToggle").checked && !!lastRec && extra.learn !== false;
    const row = await api("/api/log", { body: {
      user: settings.user, sentence, method, mode: lastMode, learn, rank,
      frames: learn ? lastRec.frames : null, dims: learn ? lastRec.dims : {}, fps: learn ? lastRec.fps : null,
      corrects: extra.corrects || null,
    } });
    lastLogId = row.id;
    if (row.learned_id) lastLearnedId = row.learned_id;
    if (row.urgent) showUrgent(row);
    if (row.learned) { try { calInfo = await api(`/api/enrollments?user=${encodeURIComponent(settings.user)}`); } catch {} }
  } catch (e) {
    console.warn("log failed", e);
  }
}

// ---------------------------------------------------------------- touch board (assistive touch)
// A floating button the patient (or a carer) can drag anywhere on the screen, so it sits where their hand
// can reach. Tapping it opens every sentence as a big button. It is the way out whenever lip reading is
// wrong, the room is dark, or the face can't be seen, and what they pick teaches the model.

const atBtn = $("#atBtn"), atPanel = $("#atPanel"), atGrid = $("#atGrid");
const atToggle = $("#atToggle");
atToggle.checked = settings.get("assistiveTouch", true);
function placeTouch(pos) {
  const size = 76, m = 12;
  const W = window.innerWidth, H = window.innerHeight;
  const x = pos.side === "left" ? m : W - size - m;
  const y = Math.min(H - size - m, Math.max(m + 56, pos.y * H));
  atBtn.style.left = `${x}px`; atBtn.style.top = `${y}px`;
}
let atPos = settings.get("atPos", { side: "right", y: 0.62 });
placeTouch(atPos);
atBtn.classList.toggle("hidden", !atToggle.checked);
atToggle.addEventListener("change", () => { settings.set("assistiveTouch", atToggle.checked); atBtn.classList.toggle("hidden", !atToggle.checked); });
window.addEventListener("resize", () => placeTouch(atPos));

let drag = null;
atBtn.addEventListener("pointerdown", (e) => {
  unlockAudio();
  drag = { id: e.pointerId, sx: e.clientX, sy: e.clientY, ox: atBtn.offsetLeft, oy: atBtn.offsetTop, moved: false };
  atBtn.setPointerCapture(e.pointerId);
});
atBtn.addEventListener("pointermove", (e) => {
  if (!drag || e.pointerId !== drag.id) return;
  const dx = e.clientX - drag.sx, dy = e.clientY - drag.sy;
  if (!drag.moved && Math.hypot(dx, dy) < 8) return; // a shaky tap is still a tap
  drag.moved = true;
  atBtn.classList.add("dragging");
  atBtn.style.left = `${drag.ox + dx}px`; atBtn.style.top = `${drag.oy + dy}px`;
});
function endDrag(e, cancelled) {
  if (!drag || e.pointerId !== drag.id) return;
  const wasDrag = drag.moved;
  drag = null;
  atBtn.classList.remove("dragging");
  if (wasDrag) {
    const cx = atBtn.offsetLeft + 38;
    atPos = { side: cx < window.innerWidth / 2 ? "left" : "right", y: atBtn.offsetTop / window.innerHeight };
    settings.set("atPos", atPos);
    placeTouch(atPos); // snap to the nearest side, like a phone's assistive-touch button
  } else if (!cancelled) openTouch();
}
atBtn.addEventListener("pointerup", (e) => endDrag(e, false));
atBtn.addEventListener("pointercancel", (e) => endDrag(e, true));
atBtn.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); openTouch(); } });

function nudgeTouch() {
  if (!atToggle.checked) return;
  atBtn.classList.remove("nudge"); void atBtn.offsetWidth; atBtn.classList.add("nudge");
}

function touchContext() {
  if (state === "choosing" && lastRec) return "correct";
  if (lastLearnedId && Date.now() - lastConfirmAt < FIX_WINDOW_MS && ["spoken", "idle"].includes(state)) return "fix";
  return "say";
}

function openTouch() {
  if (atOpen) return;
  atOpen = true;
  stopScan(); stopAutoConfirm();
  if (state === "choosing") renderChoices(); // drop the countdown bar while the board is open
  const ctx = touchContext();
  $("#atTitle").textContent = { correct: "Not on the cards? Tap what you said.", fix: "Said the wrong thing? Tap what you meant.",
    say: "Tap a sentence to say it out loud" }[ctx];
  $("#atSub").textContent = {
    correct: "Unmute says it out loud and learns your lips for that sentence, so next time it is on the first card.",
    fix: "Unmute says it out loud and moves what it just learned to the right sentence.",
    say: calibrated ? "Works in the dark, without the camera, and with any pointer or switch." : "Works without the camera or calibration.",
  }[ctx];
  const urgent = PHRASES.filter((p) => URGENT.includes(p));
  const rest = PHRASES.filter((p) => !URGENT.includes(p));
  atGrid.innerHTML = "";
  for (const p of [...urgent, ...rest]) {
    const b = document.createElement("button");
    b.className = "at-item" + (URGENT.includes(p) ? " urgent-item" : "") + (ctx === "correct" && choices.includes(p) ? " on-card" : "");
    b.textContent = p;
    if (URGENT.includes(p)) { const t = document.createElement("span"); t.className = "tag"; t.textContent = "alerts the nurse"; b.appendChild(t); }
    else if (ctx === "correct" && choices.includes(p)) { const t = document.createElement("span"); t.className = "tag"; t.textContent = `card ${choices.indexOf(p) + 1}`; b.appendChild(t); }
    b.addEventListener("click", () => pickTouch(p, ctx));
    atGrid.appendChild(b);
  }
  atPanel.classList.remove("hidden");
  const first = atGrid.querySelector("button"); if (first) first.focus({ preventScroll: true });
}

function closeTouch(resume = true) {
  if (!atOpen) return;
  atOpen = false;
  atPanel.classList.add("hidden");
  if (resume && state === "choosing") { choiceShownAt = performance.now(); gest.reset(); blinker.reset(); startScan(); startAutoConfirm(); renderChoices(); }
  atBtn.focus({ preventScroll: true });
}
$("#atClose").addEventListener("click", () => closeTouch());
atPanel.addEventListener("click", (e) => { if (e.target === atPanel) closeTouch(); });

async function pickTouch(sentence, ctx) {
  closeTouch(false);
  beep(784, 80, 0.12);
  if (ctx === "correct") {
    const rank = choices.indexOf(sentence); // -1 = the lip reading did not offer it
    state = "choosing"; // sayAndLog moves on to "spoken"
    await sayAndLog(sentence, "touch", rank);
    if (rank < 0) reasonEl.textContent = `Learned: next time this lip movement reads as "${sentence}".`;
    return;
  }
  if (ctx === "fix") {
    const wrongId = lastLearnedId, wrongRow = lastLogId;
    lastLearnedId = null;
    try { await api("/api/relabel", { body: { user: settings.user, id: wrongId, phrase: sentence } }); } catch (e) { console.warn("relabel failed", e); }
    await sayAndLog(sentence, "touch-fix", -1, { learn: false, corrects: wrongRow });
    reasonEl.textContent = `Fixed: that recording now teaches "${sentence}".`;
    return;
  }
  await sayAndLog(sentence, "touch", null, { learn: false });
}

recordBtn.addEventListener("pointerdown", () => {
  if (state === "choosing" && mode() === "switch") confirm(selected, "switch");
});

// ---------------------------------------------------------------- urgent

function showUrgent(row) {
  urgentId = row.id;
  $("#urgentText").textContent = row.sentence;
  $("#urgentPush").textContent = row.push && row.push.startsWith("ntfy HTTP 2")
    ? "Nurse's phone alerted." : `Nurse station alerted. (Phone push: ${row.push || "not sent"})`;
  $("#urgent").classList.remove("hidden");
  alarm(true);
  clearInterval(urgentPoll);
  urgentPoll = setInterval(async () => {
    try { const r = await api(`/api/requests/${urgentId}`); if (r.acknowledged) clearUrgent(false); } catch {}
  }, 2000);
}
async function clearUrgent(sendAck = true) {
  alarm(false);
  clearInterval(urgentPoll);
  $("#urgent").classList.add("hidden");
  if (sendAck && urgentId) { try { await api("/api/ack", { body: { id: urgentId } }); } catch {} }
  urgentId = null;
}
$("#urgentAck").addEventListener("click", () => clearUrgent(true));

// ---------------------------------------------------------------- keyboard fallback

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
  unlockAudio();
  if (atOpen) { if (e.key === "Escape") { e.preventDefault(); closeTouch(); } return; }
  if (e.key === "t" || e.key === "T") { openTouch(); return; }
  if (state !== "choosing") return;
  if (autoTimer && !["1", "2", "3", "Enter"].includes(e.key)) { stopAutoConfirm(); renderChoices(); } // any other key = "wait, not that"
  if (mode() === "switch" && (e.code === "Space" || e.key === "Enter") && !e.repeat) {
    e.preventDefault(); confirm(selected, "switch"); return;
  }
  if (["1", "2", "3"].includes(e.key)) confirm(+e.key - 1, "key");
  else if (e.key === "ArrowDown" || e.key === "ArrowRight" || e.key === "Tab") { e.preventDefault(); moveSelection(1); }
  else if (e.key === "ArrowUp" || e.key === "ArrowLeft") { e.preventDefault(); moveSelection(-1); }
  else if (e.key === "Enter") confirm(selected, "key");
  else if (e.key === "Escape") setState("idle", { msg: "OK, try again." });
});

// ---------------------------------------------------------------- frames

function onFrame(frame, ts) {
  lastFrame = frame;
  drawMouth(overlay, video, frame, state === "recording" ? "#f87171" : "#5eead4");
  auto.push(frame, ts);
  ptr.push(frame);
  if (!frame) camMsg.textContent = "Face the camera";
  else camMsg.textContent = state === "recording" ? (autoMode() ? "Lips moving: recording" : "Recording") : (autoMode() ? "Listening to your lips" : "Tracking your lips");
  camMsg.classList.toggle("hidden", !!frame && state !== "recording");

  if (frame && state === "choosing" && !atOpen && performance.now() - choiceShownAt > 600) {
    if (mode() === "nod") {
      const g = gest.update(frame.pitch, frame.yaw, ts);
      if (g === "nod") confirm(selected, "nod");
      else if (g === "shake") moveSelection(1);
    } else if (mode() === "blink") {
      const wasClosed = blinker.closed;
      if (blinker.update(frame.blink, ts) === "blink") { beep(988, 120, 0.15); confirm(selected, "blink"); }
      else if (blinker.closed && autoTimer) stopAutoConfirm();
      else if (blinker.closed !== wasClosed || blinker.closed) {
        const sel = choicesEl.querySelector(".choice.selected");
        if (sel) sel.style.setProperty("--blink", blinker.progress.toFixed(2));
        if (sel) sel.classList.toggle("blinking", blinker.closed);
      }
    }
  }
}

// ---------------------------------------------------------------- dark room
// Too dark: the screen turns into a ring light around the edge (lights the face from ~50 cm), and the face
// tracker gets a brightened copy of the image. The light stays on once it is on (otherwise lighting the face
// would read as "bright enough", switch the light off, and flicker), until the room is clearly bright
// without it or someone presses "Screen light off". The touch board needs no light at all.
const faceLightToggle = $("#faceLightToggle");
faceLightToggle.checked = settings.get("faceLight", true);
faceLightToggle.addEventListener("change", () => { settings.set("faceLight", faceLightToggle.checked); if (!faceLightToggle.checked) setFaceLight(false); });
let faceLight = false, brightSince = 0, lightDismissed = false;
function setFaceLight(on) {
  faceLight = on;
  document.body.classList.toggle("facelight", on);
  if (tracker) tracker.gain = on ? 2.2 : 1;
  $("#lightOff").classList.toggle("hidden", !on);
  $("#darkText").textContent = on
    ? "Dark room: the screen is lighting your face. A bed light works better. The touch button always works."
    : "Turn on the bed light: too dark to read lips. The touch button still works.";
}
function onDark(dark) {
  if (dark) $("#darkBanner").classList.remove("hidden");
  if (dark && faceLightToggle.checked && !faceLight && !lightDismissed) setFaceLight(true);
  if (!dark && !faceLight) $("#darkBanner").classList.add("hidden");
  if (dark) nudgeTouch();
}
function checkLightOff() {
  if (!faceLight || !bright || bright.value == null) return;
  // Clearly bright even with our light: a real lamp is on.
  if (bright.value > darkT * 2.2) { if (!brightSince) brightSince = Date.now(); } else brightSince = 0;
  if (brightSince && Date.now() - brightSince > 8000) { setFaceLight(false); $("#darkBanner").classList.add("hidden"); brightSince = 0; }
}
$("#lightOff").addEventListener("click", () => { lightDismissed = true; setFaceLight(false); setTimeout(() => (lightDismissed = false), 120000); });

let lastPred = null;
function renderDebug(pred) {
  if (pred) lastPred = pred;
  const f = lastFrame, d = gest.last || { dp: 0, dy: 0 };
  const lines = [
    `fps ${tracker ? tracker.fps : 0}   brightness ${bright && bright.value != null ? bright.value.toFixed(0) : "-"}`,
    `head Δpitch ${d.dp.toFixed(1)}°  Δyaw ${d.dy.toFixed(1)}°   jawOpen ${f ? f.jawOpen.toFixed(2) : "-"}   eyesClosed ${f ? f.blink.toFixed(2) : "-"}`,
  ];
  if (lastPred) {
    lines.push(`last: ${lastPred.mode}, ${lastPred.timing.total_s}s`);
    lastPred.candidates.forEach((c) => lines.push(`  ${c.confidence.toFixed(2)}  d=${c.distance.toFixed(3)}  ${c.phrase}`));
  }
  $("#debug").textContent = lines.join("\n");
}
setInterval(() => $("#settings").open && renderDebug(), 250);

// ---------------------------------------------------------------- boot

let tracker = null, bright = null;
async function boot() {
  applyMode();
  api("/api/phrases").then((p) => { URGENT = p.urgent || []; PHRASES = p.phrases || []; }).catch(() => {});
  setState("loading", { msg: "Starting camera and face tracker" });
  refreshStatus();
  // Load the face tracker while the camera starts; they don't depend on each other.
  const lmPromise = createLandmarker(() => {});
  let lmReady = false;
  const camera = new CameraControl({
    video, after: recordBtn, msgEl: camMsg,
    onReady: () => { if (lmReady) refreshUser(); },
    onError: (help) => setState("error", { msg: help }),
  });
  const camOk = await camera.start();
  try {
    const lm = await lmPromise;
    tracker = new FaceTracker(video, lm, onFrame);
    tracker.start();
    lmReady = true;
  } catch (e) {
    setState("error", { msg: "Face tracker failed to load (needs internet once, or run scripts/fetch_vendor.sh): " + e.message });
    return;
  }
  bright = new BrightnessCheck(video, onDark, () => darkT);
  setInterval(checkLightOff, 1000);
  if (camOk) await refreshUser();
  setInterval(refreshStatus, 15000);
}
boot();
