// Calibration: record each phrase a few times for the current user.
import {
  $, api, settings, createLandmarker, CameraControl, FaceTracker, drawMouth, PushToRecord,
  BrightnessCheck, beep, unlockAudio, tapMode,
} from "/common.js";

const video = $("#video"), overlay = $("#overlay"), cam = $("#cam"), camMsg = $("#camMsg");
const phraseEl = $("#phrase"), msgEl = $("#msg"), recordBtn = $("#recordBtn");
const MIN_FRAMES = 8;

let PH = null, counts = {}, current = null, lastIndex = -1, busy = false, ready = false;

const userInput = $("#userInput"), setSelect = $("#setSelect"), targetSelect = $("#targetSelect"), orderSelect = $("#orderSelect");
userInput.value = settings.user;
setSelect.value = settings.get("enrollSet", "demo");
targetSelect.value = String(settings.get("enrollTarget", 6));
orderSelect.value = settings.get("enrollOrder", "block");
orderSelect.addEventListener("change", () => { settings.set("enrollOrder", orderSelect.value); pickNext(); render(); });
userInput.addEventListener("change", () => { settings.user = userInput.value.trim().toLowerCase(); loadCounts(); });
setSelect.addEventListener("change", () => { settings.set("enrollSet", setSelect.value); pickNext(); render(); });
targetSelect.addEventListener("change", () => { settings.set("enrollTarget", +targetSelect.value); pickNext(); render(); });

// Volunteer: someone healthy records every phrase a few times for the many-people model (scripts/train.py).
// Mixed order on purpose: varied takes teach the model more than the same take repeated.
const volunteer = () => setSelect.value === "volunteer";
const phraseList = () => ({ demo: PH.demo_set, quick: PH.quick_set }[setSelect.value] || PH.phrases);
const target = () => (volunteer() ? 3 : +targetSelect.value);
const orderMode = () => (volunteer() ? "mixed" : orderSelect.value);

async function loadCounts() {
  const u = settings.user;
  $("#userBadge").textContent = u ? `user: ${u}` : "no user";
  counts = {};
  if (u) {
    try { counts = (await api(`/api/enrollments?user=${encodeURIComponent(u)}`)).counts; } catch {}
  }
  pickNext(); render();
}

// "block": stay on the same phrase until it has enough samples (people mouth it the same way when the
// previous attempt is fresh). "mixed": round-robin, the phrase with the fewest samples next.
function pickNext(skip = false) {
  const list = phraseList();
  const need = list.filter((p) => (counts[p] || 0) < target());
  if (!need.length) { current = null; return; }
  if (orderMode() === "block" && !skip) {
    if (current && need.includes(current)) return;
    const n = list.length;
    for (let k = 1; k <= n; k++) { const p = list[(lastIndex + k) % n]; if (need.includes(p)) { current = p; lastIndex = list.indexOf(p); return; } }
  }
  const minC = Math.min(...need.map((p) => counts[p] || 0));
  const n = list.length;
  for (let k = 1; k <= n; k++) {
    const p = list[(lastIndex + k) % n];
    if ((counts[p] || 0) < target() && ((counts[p] || 0) === minC || skip)) {
      current = p; lastIndex = list.indexOf(p); return;
    }
  }
  current = need[0]; lastIndex = list.indexOf(current);
}

function render() {
  const list = phraseList(), t = target();
  const done = list.reduce((a, p) => a + Math.min(counts[p] || 0, t), 0);
  const total = list.length * t;
  $("#bar").style.width = `${(100 * done) / total}%`;
  $("#progressText").textContent = `${done} / ${total} samples`;
  if (!settings.user) { phraseEl.textContent = "Type your name first"; }
  else if (current) phraseEl.textContent = `“${current}”`;
  else { phraseEl.textContent = "Calibration complete ✓"; msgEl.innerHTML = "Unmute has learned your lips. Try it on the Patient view."; }
  $("#doneLink").classList.toggle("hidden", !!current || volunteer());
  targetSelect.disabled = orderSelect.disabled = volunteer();
  $("#volunteerBox").classList.toggle("hidden", !(volunteer() && settings.user && !current));
  if (volunteer() && settings.user && !current) phraseEl.textContent = "All done ✓";
  recordBtn.disabled = !ready || !settings.user || !current;
  const chips = $("#chips"); chips.innerHTML = "";
  for (const p of list) {
    const c = document.createElement("span");
    const n = counts[p] || 0;
    c.className = "chip" + (n >= t ? " done" : "") + (p === current ? " current" : "");
    c.textContent = `${p} · ${n}`;
    chips.appendChild(c);
  }
}

const ptr = new PushToRecord({
  button: recordBtn,
  enabled: () => ready && !!settings.user && !!current && !busy,
  onStart: () => {
    cam.classList.add("recording"); $("#recDot").classList.remove("hidden"); recordBtn.classList.add("on");
    msgEl.textContent = "Recording… mouth it now";
  },
  onStop: async (rec) => {
    cam.classList.remove("recording"); $("#recDot").classList.add("hidden"); recordBtn.classList.remove("on");
    if (rec.frames.length < MIN_FRAMES) {
      msgEl.innerHTML = `<span class="flash-bad">${rec.missed > rec.frames.length ? "Face not visible." : "Too short."} Try again.</span>`;
      return;
    }
    busy = true;
    try {
      const said = current;
      const r = await api("/api/enroll", { body: { user: settings.user, phrase: said, frames: rec.frames, dims: rec.dims, fps: rec.fps } });
      counts[said] = (counts[said] || 0) + 1;
      beep(1046, 90, 0.12);
      const c = r.check;
      if (c && !c.in_top3) {
        // Clearly unlike this person's other samples of the phrase. Don't nag: explain, and let them choose.
        msgEl.innerHTML = `<span class="check-warn"></span>`;
        msgEl.firstChild.textContent = `Saved. This “${said}” looked more like “${c.looks_like}” than like your other `
          + `${c.others} samples of it. If that was a slip, click “Redo last”; otherwise keep going and mouth the two phrases differently.`;
      } else {
        msgEl.innerHTML = `<span class="flash-ok"></span>`;
        msgEl.firstChild.textContent = `Saved “${said}” (${rec.frames.length} frames, ${rec.secs.toFixed(1)}s)`
          + (c ? (c.matches ? " · matches your other samples ✓" : " · close to your other samples") : "");
      }
      pickNext(); render();
      if (!current) refreshPool();
    } catch (e) {
      msgEl.innerHTML = `<span class="flash-bad">Save failed: ${e.message}</span>`;
    } finally { busy = false; }
  },
});

$("#undoBtn").addEventListener("click", async () => {
  try {
    const r = await api(`/api/enrollments/last?user=${encodeURIComponent(settings.user)}`, { method: "DELETE" });
    counts[r.deleted] = Math.max(0, (counts[r.deleted] || 1) - 1);
    current = r.deleted; lastIndex = phraseList().indexOf(current);
    msgEl.textContent = `Removed the last sample of “${r.deleted}”. Record it again.`;
    render();
  } catch (e) { msgEl.textContent = "Nothing to undo."; }
});
$("#skipBtn").addEventListener("click", () => { pickNext(true); render(); });
$("#nextVolunteer").addEventListener("click", () => {
  settings.user = ""; userInput.value = ""; current = null; loadCounts(); refreshPool();
  msgEl.textContent = "Next volunteer: type your name (any nickname), then start.";
  userInput.focus();
});
async function refreshPool() {
  try {
    const u = await api("/api/users");
    const people = Object.keys(u).length, recs = Object.values(u).reduce((a, b) => a + b, 0);
    const line = `Shared pool: ${recs} recordings from ${people} ${people === 1 ? "person" : "people"}. `
      + "More different people = better for patients who can't record. Then run scripts/train.py.";
    $("#poolLine").textContent = line; $("#poolText").textContent = line;
  } catch {}
}
window.addEventListener("keydown", unlockAudio);

async function boot() {
  PH = await api("/api/phrases");
  await loadCounts();
  refreshPool();
  const lmPromise = createLandmarker(() => {});
  let lmReady = false;
  const camera = new CameraControl({
    video, after: cam, msgEl: camMsg,
    onReady: () => { ready = lmReady; msgEl.innerHTML = tapMode() ? "Press <kbd>Space</kbd> (or the button), mouth the phrase, press again." : "Hold <kbd>Space</kbd> (or the button), mouth the phrase, let go."; render(); },
    onError: (help) => { ready = false; msgEl.innerHTML = `<span class="flash-bad"></span>`; msgEl.firstChild.textContent = help; render(); },
  });
  await camera.start();
  let lm;
  try { lm = await lmPromise; }
  catch (e) { msgEl.textContent = "Face tracker failed to load (needs internet once, or run scripts/fetch_vendor.sh): " + e.message; return; }
  lmReady = true;
  new FaceTracker(video, lm, (frame) => {
    drawMouth(overlay, video, frame, ptr.recording ? "#f87171" : "#5eead4");
    ptr.push(frame);
    camMsg.textContent = frame ? "" : "Face the camera";
    camMsg.classList.toggle("hidden", !!frame);
  }).start();
  new BrightnessCheck(video, (dark) => $("#darkBanner").classList.toggle("hidden", !dark), () => settings.get("darkT", 55));
  ready = camera.ok; render();
}
boot();
