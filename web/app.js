// Patient view: mouth -> 3 choices -> nod to confirm -> speak (+ nurse alert if urgent).
import {
  $, api, settings, createLandmarker, CameraControl, FaceTracker, drawMouth, PushToRecord,
  GestureDetector, BrightnessCheck, speak, alarm, unlockAudio, beep, localTimeString,
} from "/common.js";

const video = $("#video"), overlay = $("#overlay"), cam = $("#cam");
const statusEl = $("#status"), subEl = $("#substatus"), choicesEl = $("#choices");
const hintEl = $("#hint"), reasonEl = $("#reason"), spokenEl = $("#spoken"), camMsg = $("#camMsg");
const recordBtn = $("#recordBtn");

const RETRY = "None of these: try again";
const MIN_FRAMES = 8;

let state = "loading";
let choices = [], confidences = {}, selected = 0, lastMode = null, choiceShownAt = 0;
let calibrated = false;
let lastFrame = null;
let urgentId = null, urgentPoll = null;
let spokenTimer = null;

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

const userInput = $("#userInput"), noteInput = $("#noteInput"), geminiToggle = $("#geminiToggle");
userInput.value = settings.user;
noteInput.value = settings.note;
geminiToggle.checked = settings.get("useGemini", true);
userInput.addEventListener("change", () => { settings.user = userInput.value.trim().toLowerCase(); refreshUser(); });
noteInput.addEventListener("change", () => (settings.note = noteInput.value));
geminiToggle.addEventListener("change", () => settings.set("useGemini", geminiToggle.checked));

async function refreshUser() {
  const u = settings.user;
  $("#userBadge").textContent = u ? `user: ${u}` : "no user";
  calibrated = false;
  if (!u) { $("#settings").open = true; setState("nouser"); return; }
  try {
    const r = await api(`/api/enrollments?user=${encodeURIComponent(u)}`);
    calibrated = Object.keys(r.counts).length >= 2;
  } catch {}
  setState(calibrated ? "idle" : "uncalibrated");
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
    subEl.innerHTML = `Unmute learns your lips in about a minute. <a href="/enroll.html">Open Calibrate</a>`;
  }
  if (s === "idle") {
    statusEl.textContent = "Mouth your sentence now";
    subEl.textContent = extra.msg || "";
    hintEl.innerHTML = "Hold <kbd>Space</kbd> (or the button) while you mouth, then let go.";
  }
  if (s === "recording") { statusEl.textContent = "Recording…"; subEl.textContent = "Mouth the sentence clearly, then let go."; }
  if (s === "thinking") { statusEl.textContent = "Reading your lips…"; subEl.textContent = ""; }
  if (s === "choosing") renderChoices();
  if (s === "spoken") {
    statusEl.textContent = "Said out loud:";
    subEl.textContent = "";
    spokenEl.textContent = extra.sentence; spokenEl.classList.remove("hidden");
    hintEl.innerHTML = "Hold <kbd>Space</kbd> to say something else.";
  }
  if (s === "error") { statusEl.textContent = "Something went wrong"; subEl.textContent = extra.msg || ""; }
}

function renderChoices() {
  statusEl.textContent = "Did you say…";
  subEl.textContent = "Nod to confirm · shake your head for the next one";
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
    b.addEventListener("click", () => confirm(i, "click"));
    choicesEl.appendChild(b);
  });
  hintEl.innerHTML = "Or press <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd>, arrows + <kbd>Enter</kbd>, <kbd>Esc</kbd> to retry.";
}

function moveSelection(delta) {
  const n = choices.length + 1;
  selected = (selected + delta + n) % n;
  beep(660, 60, 0.12);
  renderChoices();
}

// ---------------------------------------------------------------- record -> predict

const ptr = new PushToRecord({
  button: recordBtn,
  enabled: () => calibrated && ["idle", "choosing", "spoken"].includes(state),
  onStart: () => { speechSynthesis.cancel(); clearTimeout(spokenTimer); setState("recording"); },
  onStop: (rec) => onRecorded(rec),
});

async function onRecorded(rec) {
  if (rec.frames.length < MIN_FRAMES) {
    const why = rec.missed > rec.frames.length ? "I could not see your face. Face the camera." : "Too short. Hold Space for the whole sentence.";
    setState("idle", { msg: why });
    return;
  }
  setState("thinking");
  try {
    const r = await api("/api/predict", {
      body: {
        user: settings.user, frames: rec.frames, note: settings.note,
        local_time: localTimeString(), use_gemini: geminiToggle.checked,
      },
    });
    lastMode = r.mode;
    refreshStatus();
    choices = r.choices;
    confidences = Object.fromEntries(r.candidates.map((c) => [c.phrase, c.confidence]));
    selected = 0;
    choiceShownAt = performance.now();
    gest.reset();
    setState("choosing");
    if (r.mode === "gemini" && r.reason) reasonEl.textContent = "Context: " + r.reason;
    if (r.mode === "offline" && r.reason) reasonEl.textContent = "Offline mode: " + r.reason;
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
  if (sentence === RETRY) { setState("idle", { msg: "OK, try again." }); return; }
  speak(sentence);
  setState("spoken", { sentence });
  spokenTimer = setTimeout(() => state === "spoken" && setState("idle"), 6000);
  try {
    const row = await api("/api/log", { body: { user: settings.user, sentence, method, mode: lastMode } });
    if (row.urgent) showUrgent(row);
  } catch (e) {
    console.warn("log failed", e);
  }
}

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
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  unlockAudio();
  if (state !== "choosing") return;
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
  ptr.push(frame);
  if (!frame) camMsg.textContent = "Face the camera";
  else camMsg.textContent = state === "recording" ? "Recording" : "Tracking your lips";
  camMsg.classList.toggle("hidden", !!frame && state !== "recording");

  if (frame && state === "choosing" && performance.now() - choiceShownAt > 600) {
    const g = gest.update(frame.pitch, frame.yaw, ts);
    if (g === "nod") confirm(selected, "nod");
    else if (g === "shake") moveSelection(1);
  }
}

let lastPred = null;
function renderDebug(pred) {
  if (pred) lastPred = pred;
  const f = lastFrame, d = gest.last || { dp: 0, dy: 0 };
  const lines = [
    `fps ${tracker ? tracker.fps : 0}   brightness ${bright && bright.value != null ? bright.value.toFixed(0) : "-"}`,
    `head Δpitch ${d.dp.toFixed(1)}°  Δyaw ${d.dy.toFixed(1)}°   jawOpen ${f ? f.jawOpen.toFixed(2) : "-"}`,
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
  bright = new BrightnessCheck(video, (dark) => $("#darkBanner").classList.toggle("hidden", !dark), () => darkT);
  if (camOk) await refreshUser();
  setInterval(refreshStatus, 15000);
}
boot();
