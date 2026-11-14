// Shared browser code: camera, Face Landmarker, mouth features, head pose,
// gestures, brightness check, speech, alarm and small API helpers.

const MP_VERSION = "0.10.14";
const MP_CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL_REMOTE =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// Lip landmark indices from the MediaPipe face mesh (outer ring, then inner ring).
export const OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
export const INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191];
const LIPS = OUTER_LIPS.concat(INNER_LIPS);
const EYE_R = 33, EYE_L = 263; // outer eye corners

// ------------------------------------------------------------------ settings

export const settings = {
  get user() { return store("unmute.user") || ""; },
  set user(v) { store("unmute.user", v); },
  get note() { return store("unmute.note") || ""; },
  set note(v) { store("unmute.note", v); },
  get(key, dflt) { const v = store("unmute." + key); return v === null || v === undefined ? dflt : JSON.parse(v); },
  set(key, v) { store("unmute." + key, JSON.stringify(v)); },
};
function store(k, v) {
  try {
    if (v === undefined) return localStorage.getItem(k);
    localStorage.setItem(k, v);
  } catch { return null; }
}

// ------------------------------------------------------------------ API

export async function api(path, opts = {}) {
  const init = { method: opts.method || (opts.body ? "POST" : "GET"), headers: {} };
  if (opts.body) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(opts.body); }
  const r = await fetch(path, init);
  let data = null;
  try { data = await r.json(); } catch {}
  if (!r.ok) throw new Error((data && data.detail) || `HTTP ${r.status}`);
  return data;
}

// ------------------------------------------------------------------ face tracking

async function urlExists(url) {
  try { const r = await fetch(url, { method: "HEAD" }); return r.ok; } catch { return false; }
}

export async function createLandmarker(onStatus = () => {}) {
  // Prefer local copies in web/vendor (fully offline); fall back to the CDN.
  let vision, wasmBase, modelPath;
  if (await urlExists("/vendor/face_landmarker.task") && await urlExists("/vendor/tasks-vision/vision_bundle.mjs")) {
    onStatus("Loading face tracker (local files)…");
    vision = await import("/vendor/tasks-vision/vision_bundle.mjs");
    wasmBase = "/vendor/tasks-vision/wasm";
    modelPath = "/vendor/face_landmarker.task";
  } else {
    onStatus("Loading face tracker…");
    vision = await import(`${MP_CDN}/vision_bundle.mjs`);
    wasmBase = `${MP_CDN}/wasm`;
    modelPath = MODEL_REMOTE;
  }
  const fileset = await vision.FilesetResolver.forVisionTasks(wasmBase);
  const opts = {
    baseOptions: { modelAssetPath: modelPath, delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  };
  try {
    return await vision.FaceLandmarker.createFromOptions(fileset, opts);
  } catch (e) {
    console.warn("GPU delegate failed, using CPU", e);
    opts.baseOptions.delegate = "CPU";
    return await vision.FaceLandmarker.createFromOptions(fileset, opts);
  }
}

// Explains a camera failure in plain words, with the fix.
export function cameraHelp(e) {
  if (!window.isSecureContext)
    return "The browser only allows the camera on http://localhost:8000. Open that address (not a file:// or network IP).";
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
    return "This browser window can't use cameras. Open http://localhost:8000 in Brave or Chrome.";
  const n = e && e.name;
  if (n === "NotAllowedError" || n === "SecurityError")
    return "Camera permission is blocked. 1) Use Brave or Chrome, not the Claude app's preview pane. " +
      "2) Click the camera icon in the address bar, choose Allow, then Retry. " +
      "3) On a Mac: System Settings → Privacy & Security → Camera → turn on Brave (or Chrome), then quit and reopen it.";
  if (n === "NotFoundError" || n === "OverconstrainedError")
    return "No camera found. Pick another camera from the list, or check that the webcam is connected.";
  if (n === "NotReadableError" || n === "AbortError" || n === "TrackStartError")
    return "The camera is busy or failed to start. Close FaceTime, Zoom, Photo Booth or other tabs using it, then Retry.";
  if (n === "Timeout")
    return "The browser got the camera but no picture arrives. Usually macOS is blocking it: System Settings → Privacy & Security → Camera → " +
      "turn on Brave, then quit Brave fully (Cmd+Q) and reopen. Also close Zoom/FaceTime/Photo Booth and other Unmute tabs, " +
      "or pick another camera in the list and press Retry. [" + e.message + "]";
  return `Camera error (${n || "unknown"}): ${(e && e.message) || e}`;
}

let currentStream = null;

// Starts the camera, trying gentler settings if the first attempt fails.
export async function startCamera(video, deviceId = null) {
  if (currentStream) { currentStream.getTracks().forEach((t) => t.stop()); currentStream = null; }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw Object.assign(new Error("no mediaDevices"), { name: "NoMediaDevices" });
  const dev = deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "user" };
  const attempts = [
    { ...dev, width: { ideal: 1280 }, height: { ideal: 720 } },
    { ...dev },
    deviceId ? { deviceId } : true,
  ];
  let lastErr = null, stream = null;
  for (const v of attempts) {
    try { stream = await navigator.mediaDevices.getUserMedia({ video: v, audio: false }); break; }
    catch (e) {
      lastErr = e;
      if (e.name === "NotAllowedError" || e.name === "SecurityError") throw e; // retrying won't help
    }
  }
  if (!stream) throw lastErr;
  currentStream = stream;
  video.muted = true; video.playsInline = true;
  video.srcObject = stream;
  // Don't await play(): if the camera never sends a frame, that promise never settles.
  video.play().catch(() => {});
  // Wait until real frames arrive (some Mac cameras take a moment to wake up).
  const t0 = performance.now();
  const track = stream.getVideoTracks()[0];
  while (!(video.videoWidth > 0 && video.readyState >= 2)) {
    if (performance.now() - t0 > 6000) {
      const why = `track ${track ? track.readyState : "none"}, muted=${track ? track.muted : "?"}, label="${track ? track.label : ""}"`;
      const failedId = track ? track.getSettings().deviceId : null;
      stream.getTracks().forEach((t) => t.stop()); currentStream = null;
      throw Object.assign(new Error("no frames from camera (" + why + ")"), { name: "Timeout", deviceId: failedId });
    }
    if (performance.now() - t0 > 2500) video.play().catch(() => {});
    await new Promise((r) => setTimeout(r, 150));
  }
  return { label: track.label, deviceId: track.getSettings().deviceId, width: video.videoWidth, height: video.videoHeight };
}

export async function listCameras() {
  try { return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput"); }
  catch { return []; }
}

// Built-in laptop cameras first; iPhone Continuity / Desk View cameras last
// (macOS often makes the iPhone the default even when the phone isn't set up as a webcam).
function cameraRank(c) {
  const l = (c.label || "").toLowerCase();
  if (/facetime|built-in|macbook|integrated/.test(l)) return 0;
  if (/iphone|continuity|desk view|ipad/.test(l)) return 2;
  return 1;
}
export function sortCameras(cams) { return cams.slice().sort((a, b) => cameraRank(a) - cameraRank(b)); }

// Adds a camera picker + Retry button under the video and keeps the camera running.
export class CameraControl {
  constructor({ video, after, msgEl, onReady = () => {}, onError = () => {} }) {
    this.video = video; this.msgEl = msgEl; this.onReady = onReady; this.onError = onError; this.ok = false;
    const bar = document.createElement("div");
    bar.className = "cam-bar";
    bar.innerHTML = `<select title="Camera"></select><button class="btn">Retry camera</button>`;
    after.insertAdjacentElement("afterend", bar);
    this.select = bar.querySelector("select"); this.retry = bar.querySelector("button");
    this.select.addEventListener("change", () => { settings.set("cameraId", this.select.value); this.start(); });
    this.retry.addEventListener("click", () => this.start());
    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener)
      navigator.mediaDevices.addEventListener("devicechange", () => this.fillList());
  }
  async fillList(activeId) {
    const cams = await listCameras();
    this.select.innerHTML = "";
    if (!cams.length) { this.select.innerHTML = "<option>No camera found</option>"; return; }
    cams.forEach((c, i) => {
      const o = document.createElement("option");
      o.value = c.deviceId; o.textContent = c.label || `Camera ${i + 1}`;
      if (c.deviceId && c.deviceId === activeId) o.selected = true;
      this.select.appendChild(o);
    });
  }
  async start() {
    this.ok = false;
    this.msgEl.classList.remove("hidden");
    this.msgEl.textContent = "Starting camera… (allow access if Chrome asks)";
    this.retry.disabled = true;
    const slow = setTimeout(() => {
      if (!this.ok) this.msgEl.textContent = "Still waiting for the camera… If the browser shows a permission prompt near the address bar, click Allow.";
    }, 4000);
    let wanted = settings.get("cameraId", null);
    if (!wanted) { // labels are known once permission was granted before: pick the built-in camera
      const best = sortCameras(await listCameras())[0];
      if (best && best.label && best.deviceId) wanted = best.deviceId;
    }
    try {
      let info = null, err = null;
      const tried = new Set();
      try { info = await startCamera(this.video, wanted); }
      catch (e) { err = e; tried.add(wanted); if (e.deviceId) tried.add(e.deviceId); }
      // That camera failed (busy, or an iPhone sending nothing): try every other camera in turn.
      if (!info && err.name !== "NotAllowedError" && err.name !== "SecurityError") {
        for (const c of sortCameras(await listCameras())) {
          if (!c.deviceId || tried.has(c.deviceId)) continue;
          this.msgEl.textContent = `Trying ${c.label || "another camera"}…`;
          try { info = await startCamera(this.video, c.deviceId); break; }
          catch (e) { err = e; tried.add(c.deviceId); }
        }
      }
      if (!info) throw err;
      settings.set("cameraId", info.deviceId);
      this.ok = true;
      await this.fillList(info.deviceId);
      this.msgEl.textContent = `Camera on: ${info.label || "webcam"}`;
      this.onReady(info);
    } catch (e) {
      console.warn("camera failed", e);
      await this.fillList(wanted);
      const help = cameraHelp(e);
      this.msgEl.textContent = help;
      this.onError(help, e);
    } finally { clearTimeout(slow); this.retry.disabled = false; }
    return this.ok;
  }
}

// Converts a Face Landmarker result into our per-frame feature vector.
export function extractFrame(res, w, h) {
  if (!res || !res.faceLandmarks || !res.faceLandmarks.length) return null;
  const lm = res.faceLandmarks[0];
  const P = (i) => [lm[i].x * w, lm[i].y * h];

  // Normalise lips: centre on the mouth, undo head roll, divide by eye distance.
  const e1 = P(EYE_R), e2 = P(EYE_L);
  const dx = e2[0] - e1[0], dy = e2[1] - e1[1];
  const eyeDist = Math.hypot(dx, dy) || 1;
  const ang = -Math.atan2(dy, dx);
  const c = Math.cos(ang), s = Math.sin(ang);
  const pts = LIPS.map(P);
  let cx = 0, cy = 0;
  for (const p of pts) { cx += p[0]; cy += p[1]; }
  cx /= pts.length; cy /= pts.length;
  const lips = [];
  for (const [x, y] of pts) {
    const ux = x - cx, uy = y - cy;
    lips.push((ux * c - uy * s) / eyeDist, (ux * s + uy * c) / eyeDist);
  }

  // Mouth blendshapes, in a fixed (alphabetical) order.
  let bs = [], bsNames = [], blink = 0;
  if (res.faceBlendshapes && res.faceBlendshapes.length) {
    const all = res.faceBlendshapes[0].categories;
    const sc = (n) => { const k = all.find((c) => c.categoryName === n); return k ? k.score : 0; };
    blink = Math.min(sc("eyeBlinkLeft"), sc("eyeBlinkRight")); // both eyes closed
    const cats = res.faceBlendshapes[0].categories
      .filter((k) => k.categoryName.startsWith("jaw") || k.categoryName.startsWith("mouth") || k.categoryName === "cheekPuff")
      .sort((a, b) => a.categoryName.localeCompare(b.categoryName));
    bs = cats.map((k) => k.score);
    bsNames = cats.map((k) => k.categoryName);
  }

  // Head pitch / yaw (degrees) from the 4x4 transformation matrix.
  let pitch = 0, yaw = 0;
  const m = res.facialTransformationMatrixes && res.facialTransformationMatrixes[0];
  if (m && m.data) {
    const d = m.data; // sign conventions do not matter: gestures use deviation from a baseline
    const R = (r, col) => d[col * 4 + r];
    pitch = (Math.atan2(R(2, 1), R(2, 2)) * 180) / Math.PI;
    yaw = (Math.asin(Math.max(-1, Math.min(1, -R(2, 0)))) * 180) / Math.PI;
  }

  const jawOpen = bsNames.length ? bs[bsNames.indexOf("jawOpen")] : 0;
  return {
    feat: bs.concat(lips),
    dims: { bs: bs.length, lip: lips.length },
    pitch, yaw, jawOpen, blink,
    landmarks: lm,
  };
}

export class FaceTracker {
  constructor(video, landmarker, onFrame) {
    this.video = video; this.lm = landmarker; this.onFrame = onFrame;
    this.lastTs = -1; this.running = false; this.fps = 0; this._times = [];
  }
  start() { this.running = true; const loop = () => { if (!this.running) return; this.tick(); requestAnimationFrame(loop); }; requestAnimationFrame(loop); }
  stop() { this.running = false; }
  tick() {
    const v = this.video;
    if (v.readyState < 2) return;
    let ts = performance.now();
    if (ts <= this.lastTs) ts = this.lastTs + 1;
    this.lastTs = ts;
    let res;
    try { res = this.lm.detectForVideo(v, ts); } catch (e) { console.warn(e); return; }
    this._times.push(ts); while (this._times.length && ts - this._times[0] > 1000) this._times.shift();
    this.fps = this._times.length;
    this.onFrame(extractFrame(res, v.videoWidth, v.videoHeight), ts);
  }
}

// Draws the lip outline on a canvas sized to the video.
export function drawMouth(canvas, video, frame, color = "#5eead4") {
  const ctx = canvas.getContext("2d");
  if (canvas.width !== video.videoWidth) { canvas.width = video.videoWidth; canvas.height = video.videoHeight; }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!frame) return;
  const lm = frame.landmarks, W = canvas.width, H = canvas.height;
  ctx.lineWidth = Math.max(2, W / 400);
  ctx.strokeStyle = color;
  for (const ring of [OUTER_LIPS, INNER_LIPS]) {
    ctx.beginPath();
    ring.forEach((i, k) => { const x = lm[i].x * W, y = lm[i].y * H; k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.closePath(); ctx.stroke();
  }
  ctx.fillStyle = color;
  for (const i of LIPS) { ctx.beginPath(); ctx.arc(lm[i].x * W, lm[i].y * H, ctx.lineWidth, 0, 6.283); ctx.fill(); }
}

// ------------------------------------------------------------------ push to record

// Hold Space (or the given button / a USB switch that types Space) to record; release to stop.
// In "tap" mode (settings.recordMode) one press starts and the next press stops,
// for patients who cannot hold a switch down.
export const MAX_RECORD_S = 8;
export function tapMode() { return settings.get("recordMode", "hold") === "tap"; }
export function autoMode() { return settings.get("recordMode", "hold") === "auto"; }
export class PushToRecord {
  constructor({ button, onStart, onStop, enabled = () => true }) {
    this.recording = false; this.frames = []; this.dims = null; this.t0 = 0;
    this.onStart = onStart; this.onStop = onStop; this.enabled = enabled;
    const press = () => { if (tapMode() && this.recording) this.end(); else this.begin(); };
    const release = () => { if (!tapMode()) this.end(); };
    window.addEventListener("keydown", (e) => {
      if (e.code !== "Space" || e.repeat || isTyping(e)) return;
      e.preventDefault(); press();
    });
    window.addEventListener("keyup", (e) => {
      if (e.code !== "Space" || isTyping(e)) return;
      e.preventDefault(); release();
    });
    if (button) {
      button.addEventListener("pointerdown", (e) => { e.preventDefault(); button.setPointerCapture(e.pointerId); press(); });
      button.addEventListener("pointerup", release);
      button.addEventListener("pointercancel", release);
    }
  }
  begin() {
    if (this.recording || !this.enabled()) return;
    unlockAudio();
    this.recording = true; this.frames = []; this.missed = 0; this.t0 = performance.now();
    clearTimeout(this._cap);
    this._cap = setTimeout(() => this.end(), MAX_RECORD_S * 1000); // a stuck switch never records forever
    this.onStart && this.onStart();
  }
  push(frame) {
    if (!this.recording) return;
    if (!frame) { this.missed++; return; }
    this.frames.push(frame.feat); this.dims = frame.dims;
  }
  end() {
    if (!this.recording) return;
    this.recording = false; clearTimeout(this._cap);
    const secs = (performance.now() - this.t0) / 1000;
    this.onStop && this.onStop({ frames: this.frames, dims: this.dims, secs, fps: this.frames.length / Math.max(secs, 0.001), missed: this.missed });
  }
}
function isTyping(e) {
  const t = e.target;
  return t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
}

// ------------------------------------------------------------------ gestures

// Nod = pitch leaves the baseline and comes back within maxMs, with more pitch than yaw.
// Shake = same, with more yaw than pitch.
export class GestureDetector {
  constructor(opts = {}) {
    this.pitchT = opts.pitchT ?? 8; this.yawT = opts.yawT ?? 12;
    this.maxMs = opts.maxMs ?? 1500; this.cooldownMs = opts.cooldownMs ?? 700;
    this.reset();
  }
  reset() { this.base = null; this.moving = false; this.lastEvent = performance.now(); this.warm = 0; }
  update(pitch, yaw, t) {
    if (this.base === null) { this.base = { p: pitch, y: yaw }; this.warm = 0; return null; }
    const dp = pitch - this.base.p, dy = wrap(yaw - this.base.y);
    this.last = { dp, dy };
    if (this.warm < 8) { // settle the baseline over a few frames first
      this.warm++; this.base.p += 0.3 * dp; this.base.y += 0.3 * dy; return null;
    }
    if (!this.moving) {
      if (Math.abs(dp) > this.pitchT || Math.abs(dy) > this.yawT) {
        this.moving = true; this.start = t; this.maxP = 0; this.maxY = 0;
      } else {
        this.base.p += 0.05 * dp; this.base.y += 0.05 * dy; // follow slow drift
        return null;
      }
    }
    this.maxP = Math.max(this.maxP, Math.abs(dp)); this.maxY = Math.max(this.maxY, Math.abs(dy));
    const back = Math.abs(dp) < this.pitchT * 0.5 && Math.abs(dy) < this.yawT * 0.5;
    if (back) {
      this.moving = false;
      const quick = t - this.start <= this.maxMs;
      if (!quick || t - this.lastEvent < this.cooldownMs) return null;
      this.lastEvent = t;
      return this.maxP / this.pitchT >= this.maxY / this.yawT ? "nod" : "shake";
    }
    if (t - this.start > this.maxMs * 1.5) { // head just moved somewhere new: re-baseline
      this.moving = false; this.base = { p: pitch, y: yaw };
    }
    return null;
  }
}
// Hands-free recording: start when the lips start moving, stop when they have been still for stopMs.
// Energy = mean |frame-to-frame change| over the lip coordinates, EMA-smoothed. Measured on our recordings:
// still face p95 = 0.0016, p99 = 0.0029; peak while mouthing p10 = 0.0025, median 0.0050.
export class LipActivity {
  constructor(opts = {}) {
    this.startT = opts.startT ?? 0.0025; this.stopFrac = opts.stopFrac ?? 0.45;
    this.startFrames = opts.startFrames ?? 3; this.stopMs = opts.stopMs ?? 700;
    this.minMs = opts.minMs ?? 600; this.maxMs = opts.maxMs ?? 6000;
    this.reset();
  }
  reset() { this.prev = null; this.energy = 0; this.above = 0; this.active = false; this.quietSince = null; this.startedAt = 0; }
  // Returns "start", "stop" or null. `lips` = the lip part of the feature vector.
  update(lips, t) {
    if (!lips) { this.prev = null; return this.active ? this._stop() : null; }
    if (this.prev && this.prev.length === lips.length) {
      let s = 0;
      for (let i = 0; i < lips.length; i++) s += Math.abs(lips[i] - this.prev[i]);
      this.energy = 0.6 * this.energy + 0.4 * (s / lips.length);
    }
    this.prev = lips;
    if (!this.active) {
      this.above = this.energy > this.startT ? this.above + 1 : 0;
      if (this.above >= this.startFrames) { this.active = true; this.startedAt = t; this.quietSince = null; this.above = 0; return "start"; }
      return null;
    }
    const quiet = this.energy < this.startT * this.stopFrac;
    if (quiet) { if (this.quietSince === null) this.quietSince = t; }
    else this.quietSince = null;
    const dur = t - this.startedAt;
    if ((this.quietSince !== null && t - this.quietSince >= this.stopMs && dur >= this.minMs) || dur >= this.maxMs) return this._stop();
    return null;
  }
  _stop() { this.active = false; this.quietSince = null; return "stop"; }
}

// Long blink = both eyes closed for at least holdMs. Normal blinks (~0.1-0.3 s) are ignored.
export class LongBlink {
  constructor(opts = {}) { this.holdMs = opts.holdMs ?? 800; this.thresh = opts.thresh ?? 0.5; this.reset(); }
  reset() { this.closedAt = null; this.fired = false; this.progress = 0; }
  // Returns "blink" once per long closure; this.progress is 0..1 while the eyes are shut.
  update(score, t) {
    if (score < this.thresh * 0.7) { this.reset(); return null; }   // eyes open (with hysteresis)
    if (score < this.thresh && this.closedAt === null) return null;
    if (this.closedAt === null) this.closedAt = t;
    this.progress = Math.min(1, (t - this.closedAt) / this.holdMs);
    if (!this.fired && this.progress >= 1) { this.fired = true; return "blink"; }
    return null;
  }
  get closed() { return this.closedAt !== null; }
}

function wrap(a) { while (a > 180) a -= 360; while (a < -180) a += 360; return a; }

// ------------------------------------------------------------------ brightness

export class BrightnessCheck {
  constructor(video, onChange, threshold = () => 55) {
    this.video = video; this.onChange = onChange; this.threshold = threshold;
    this.canvas = document.createElement("canvas"); this.canvas.width = 32; this.canvas.height = 24;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.dark = false; this.value = null;
    setInterval(() => this.check(), 1000);
  }
  check() {
    if (this.video.readyState < 2) return;
    this.ctx.drawImage(this.video, 0, 0, 32, 24);
    const d = this.ctx.getImageData(0, 0, 32, 24).data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) sum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    this.value = sum / (d.length / 4);
    const dark = this.value < this.threshold();
    if (dark !== this.dark) { this.dark = dark; this.onChange(dark, this.value); }
  }
}

// ------------------------------------------------------------------ speech and sound

export function speak(text) {
  if (!("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.92; u.pitch = 1; u.volume = 1;
  const voices = speechSynthesis.getVoices();
  const v = voices.find((v) => v.lang.startsWith("en") && v.localService) || voices.find((v) => v.lang.startsWith("en"));
  if (v) u.voice = v;
  speechSynthesis.speak(u);
}

let audioCtx = null;
export function unlockAudio() {
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") audioCtx.resume();
  } catch {}
}
export function beep(freq = 880, ms = 120, vol = 0.2) {
  if (!audioCtx) return;
  const o = audioCtx.createOscillator(), g = audioCtx.createGain();
  o.type = "square"; o.frequency.value = freq; g.gain.value = vol;
  o.connect(g); g.connect(audioCtx.destination);
  o.start(); o.stop(audioCtx.currentTime + ms / 1000);
}
let alarmTimer = null;
export function alarm(on) {
  if (on && !alarmTimer) {
    const play = () => { beep(988, 180, 0.25); setTimeout(() => beep(740, 180, 0.25), 220); };
    play(); alarmTimer = setInterval(play, 900);
  } else if (!on && alarmTimer) { clearInterval(alarmTimer); alarmTimer = null; }
}

// ------------------------------------------------------------------ misc

export function $(sel) { return document.querySelector(sel); }
export function localTimeString() {
  return new Date().toLocaleString([], { weekday: "long", hour: "2-digit", minute: "2-digit" });
}
