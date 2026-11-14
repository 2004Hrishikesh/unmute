// Calibration: record each phrase a few times for the current user.
import {
  $, api, settings, createLandmarker, CameraControl, FaceTracker, drawMouth, PushToRecord,
  BrightnessCheck, beep, unlockAudio, tapMode,
} from "/common.js";

const video = $("#video"), overlay = $("#overlay"), cam = $("#cam"), camMsg = $("#camMsg");
const phraseEl = $("#phrase"), msgEl = $("#msg"), recordBtn = $("#recordBtn");
const MIN_FRAMES = 8;

let PH = null, counts = {}, current = null, lastIndex = -1, busy = false, ready = false;

const userInput = $("#userInput"), setSelect = $("#setSelect"), targetSelect = $("#targetSelect");
userInput.value = settings.user;
setSelect.value = settings.get("enrollSet", "demo");
targetSelect.value = String(settings.get("enrollTarget", 4));
userInput.addEventListener("change", () => { settings.user = userInput.value.trim().toLowerCase(); loadCounts(); });
setSelect.addEventListener("change", () => { settings.set("enrollSet", setSelect.value); pickNext(); render(); });
targetSelect.addEventListener("change", () => { settings.set("enrollTarget", +targetSelect.value); pickNext(); render(); });

const phraseList = () => (setSelect.value === "demo" ? PH.demo_set : PH.phrases);
const target = () => +targetSelect.value;

async function loadCounts() {
  const u = settings.user;
  $("#userBadge").textContent = u ? `user: ${u}` : "no user";
  counts = {};
  if (u) {
    try { counts = (await api(`/api/enrollments?user=${encodeURIComponent(u)}`)).counts; } catch {}
  }
  pickNext(); render();
}

// Round-robin: the phrase with the fewest samples, continuing after the last one recorded.
function pickNext(skip = false) {
  const list = phraseList();
  const need = list.filter((p) => (counts[p] || 0) < target());
  if (!need.length) { current = null; return; }
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
  $("#doneLink").classList.toggle("hidden", !!current);
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
      if (c && !c.matches) {
        // Inconsistent samples are the main cause of wrong guesses later: offer to redo it now.
        msgEl.innerHTML = `<span class="check-warn"></span>`;
        msgEl.firstChild.textContent = `Saved, but this “${said}” looked more like “${c.looks_like}”. `
          + `Click “Redo last” and mouth it the same way as before: same speed, bigger lips.`;
      } else {
        msgEl.innerHTML = `<span class="flash-ok"></span>`;
        msgEl.firstChild.textContent = `Saved “${said}” (${rec.frames.length} frames, ${rec.secs.toFixed(1)}s)`
          + (c ? " · matches your other samples ✓" : "");
      }
      pickNext(); render();
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
window.addEventListener("keydown", unlockAudio);

async function boot() {
  PH = await api("/api/phrases");
  await loadCounts();
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
