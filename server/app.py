"""Unmute backend: serves web/ and the JSON API.

Run from the repo root:  uvicorn server.app:app --reload
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import alerts, avsr, features, neural, rerank
from .classifier import Config, Recogniser
from .evaluation import evaluate_all

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ENROLL_DIR = DATA / "enrollments"
REQUESTS_FILE = DATA / "requests.jsonl"
RESULTS_FILE = DATA / "results.json"
CONFIG_FILE = DATA / "config.json"
MODEL_PATH = DATA / "model" / "lipnet"  # .npz + .json, written by scripts/train.py (gitignored)
CROPS_DIR = DATA / "crops"              # aligned 96x96 mouth crops per calibration recording (gitignored: face images)
AVSR_CONFIG = DATA / "avsr_config.json" # lip-reader weight measured by scripts/eval_avsr.py
AVSR_DEFAULT_WEIGHT = 0.5               # used until eval_avsr.py has measured one on our own recordings
_avsr_lock = threading.Lock()           # one lip-reader pass at a time on the GPU
ENROLL_DIR.mkdir(parents=True, exist_ok=True)
ENV_FILE = ROOT / ".env"


def _load_env_file():
    """Read KEY=value lines from .env (gitignored) so the Gemini key never has to be typed in a shell."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


_load_env_file()
rerank.STATE_FILE = DATA / "gemini_state.json"  # gitignored
MIN_LIP_MOTION = 0.012  # below this the lips barely moved (eye-distance units)

PHRASES = json.loads((DATA / "phrases.json").read_text())
rerank.URGENT = set(PHRASES["urgent"])
BED = os.environ.get("BED_NAME", "Bed 1")


def _load_config() -> dict:
    cfg = {}
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
        except Exception:
            cfg = {}
    if not cfg.get("ntfy_topic"):
        # Random topic: ntfy topics are public, so make it hard to guess.
        cfg["ntfy_topic"] = "unmute-" + secrets.token_hex(5)
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    return cfg


CONFIG = _load_config()
NTFY_TOPIC = os.environ.get("NTFY_TOPIC") or CONFIG["ntfy_topic"]

app = FastAPI(title="Unmute")
_lock = threading.Lock()


def safe_user(user: str) -> str:
    u = re.sub(r"[^a-zA-Z0-9_-]", "", (user or "").strip().lower())[:32]
    if not u:
        raise HTTPException(400, "user name must contain letters or digits")
    return u


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# ---------------------------------------------------------------- enrollments

def load_samples(user: str | None = None) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in sorted(ENROLL_DIR.glob("*.json")):
        try:
            s = json.loads(f.read_text())
        except Exception:
            continue
        if user and s.get("user") != user:
            continue
        s["_file"] = f.name
        out.setdefault(s["user"], []).append(s)
    return out


_rec_cache: dict[str, tuple[tuple, Recogniser]] = {}
MAX_USE_SAMPLES = 8  # per phrase per user, learned from confirmed sentences or corrections
# Building the pool is O(N^2) (hubness needs every pair): measured 0.1 s at 142 recordings, 1.2 s at 600,
# 4.4 s at 1200. Above this size a new recording is folded in by a background rebuild, and predictions keep
# using the previous model until it is ready, so nobody waits on the build.
SYNC_REBUILD_MAX = 400
_rebuild_lock = threading.Lock()
_rebuilding = False


def _fingerprint() -> tuple:
    files = list(ENROLL_DIR.glob("*.json"))
    return (len(files), max((f.stat().st_mtime for f in files), default=0))


def get_recogniser(user: str | None = None) -> Recogniser | None:
    """One recogniser over EVERY person's recordings (the shared starter model).

    Measured on our own data: a new person with no calibration gets the right sentence in the top 3
    68-94% of the time from other people's recordings, and adding other people's recordings to a
    calibrated person's own did not hurt (rishi: 75%/95% own only, 75%/98% pooled). The person's own
    samples win naturally when they are closer. So there is one pool, and `user` is only kept for
    the API shape."""
    fp = _fingerprint()
    cached = _rec_cache.get("*")
    if cached and cached[0] == fp:
        return cached[1]
    if cached and fp[0] > SYNC_REBUILD_MAX:
        _rebuild_in_background()
        return cached[1]
    return _build_pool(fp)


def _build_pool(fp: tuple) -> Recogniser | None:
    samples = [s for ss in load_samples().values() for s in ss]
    if len({s["phrase"] for s in samples}) < 2:
        _rec_cache.pop("*", None)
        return None
    rec = Recogniser(samples, Config())
    _rec_cache["*"] = (fp, rec)
    return rec


def _rebuild_in_background():
    global _rebuilding
    with _rebuild_lock:
        if _rebuilding:
            return
        _rebuilding = True

    def run():
        global _rebuilding
        try:
            _build_pool(_fingerprint())
        finally:
            _rebuilding = False

    threading.Thread(target=run, daemon=True).start()


_model_cache: dict = {"mtime": None, "net": None}


def get_model() -> neural.LipNet | None:
    """The many-people model from scripts/train.py, reloaded when the file changes. None if there is none,
    or if its leave-one-person-out test said the matcher alone is better."""
    f = MODEL_PATH.with_suffix(".json")
    try:
        mtime = f.stat().st_mtime
    except OSError:
        _model_cache.update(mtime=None, net=None)
        return None
    if _model_cache["mtime"] != mtime:
        try:
            _model_cache.update(mtime=mtime, net=neural.LipNet.load(MODEL_PATH))
        except Exception:
            _model_cache.update(mtime=mtime, net=None)
    net = _model_cache["net"]
    return net if net is not None and net.meta.get("use") else None


def model_info() -> dict:
    net = get_model()
    if net is None:
        return {"active": False}
    m = net.meta
    return {"active": True, "people": len(m.get("users", [])), "samples": m.get("samples"), "phrases": len(net.phrases),
            "best": m.get("best"), "alpha": m.get("alpha"), "for_calibrated": m.get("use_for_calibrated", False),
            "trained_at": m.get("trained_at")}


def rank_with_model(rec: Recogniser, frames, dims: dict | None, personal: bool) -> tuple[list[dict], str]:
    """Top 5 candidates. With a deployed model: the matcher's per-phrase distances fused with the model's
    probabilities, at the weight leave-one-person-out testing picked. People with their own recordings only
    get the fusion if the training script measured that it doesn't hurt them."""
    net = get_model()
    if net is None or (personal and not net.meta.get("use_for_calibrated")):
        return rec.rank(frames, top=5), "matcher"
    ranked = rec.rank(frames, top=999)
    try:
        fused = neural.fuse(net.proba(frames, dims), neural.dtw_proba(ranked, net.meta["tau"]), net.meta["alpha"])
    except ValueError:
        return ranked[:5], "matcher"
    dist = {r["phrase"]: r["distance"] for r in ranked}
    return [{"phrase": p, "distance": dist[p], "confidence": round(c, 4)} for p, c in fused[:5]], "model+matcher"


# ---------------------------------------------------------------- open-source lip reader (Auto-AVSR)

def decode_crops(c: dict | None) -> np.ndarray | None:
    """Browser crops -> (T, 96, 96) aligned mouth crops at 25 fps, or None."""
    if not c or not avsr.available():
        return None
    import base64

    import cv2

    imgs, kp, ts = c.get("imgs") or [], c.get("kp") or [], c.get("ts") or []
    n = min(len(imgs), len(kp), len(ts))
    if n < 10:
        return None
    idx = avsr.resample_fps(list(range(n)), ts[:n])
    frames, points = [], []
    for i in idx:
        try:
            raw = base64.b64decode(imgs[i].split(",", 1)[-1])
            g = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        except Exception:
            g = None
        if g is None:
            continue
        frames.append(g)
        points.append(np.asarray(kp[i], dtype=np.float64).reshape(4, 2))
    if len(frames) < 10:
        return None
    return avsr.mouth_crops(frames, np.stack(points))


def avsr_weight() -> tuple[float, bool]:
    """(weight of the lip reader in the blend, measured?)"""
    try:
        cfg = json.loads(AVSR_CONFIG.read_text())
        return float(cfg["weight"]), True
    except Exception:
        return AVSR_DEFAULT_WEIGHT, False


def blend_with_lip_reader(ranked: list[dict], lip: dict, weight: float) -> list[dict]:
    """Every phrase on the list is scored by the lip reader, including phrases nobody has recorded. The matcher's
    distances (when it has the phrase) are blended in: weight = trust in the lip reader."""
    p_lip = lip["proba"]
    p_dtw = neural.dtw_proba(ranked, 0.05) if ranked else {}
    out = []
    for ph, pl in p_lip.items():
        pd = p_dtw.get(ph, 1e-3) if p_dtw else 1.0
        out.append((ph, weight * np.log(pl + 1e-9) + (1 - weight) * np.log(pd + 1e-9)))
    v = np.array([x for _, x in out])
    w = np.exp(v - v.max()); w /= w.sum()
    dist = {r["phrase"]: r["distance"] for r in ranked}
    order = sorted(zip([p for p, _ in out], w), key=lambda kv: -kv[1])
    return [{"phrase": p, "distance": dist.get(p), "confidence": round(float(c), 4)} for p, c in order]


def calibration_info(user: str) -> dict:
    """How much of the pool is this person's own, and how much is other people's (the starter)."""
    by_user = load_samples()
    own = by_user.get(user, [])
    others = [s for u, ss in by_user.items() if u != user for s in ss]
    counts: dict[str, int] = {}
    from_use = 0
    for s in own:
        counts[s["phrase"]] = counts.get(s["phrase"], 0) + 1
        from_use += s.get("source") == "use"
    return {
        "user": user,
        "counts": counts,
        "own_samples": len(own),
        "own_from_use": from_use,
        "personal": len(counts) >= 2,
        "starter_available": len({s["phrase"] for s in others}) >= 2,
        "starter_samples": len(others),
        "starter_users": len([u for u in by_user if u != user and by_user[u]]),
    }


# ---------------------------------------------------------------- requests log

def read_requests() -> list[dict]:
    if not REQUESTS_FILE.exists():
        return []
    rows = []
    for line in REQUESTS_FILE.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def write_requests(rows: list[dict]):
    tmp = REQUESTS_FILE.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(REQUESTS_FILE)


def _ago(ts: float) -> str:
    m = int((time.time() - ts) // 60)
    if m < 1:
        return "just now"
    if m < 60:
        return f"{m} min ago"
    return f"{m // 60} h {m % 60} min ago"


# ---------------------------------------------------------------- API models

class EnrollIn(BaseModel):
    user: str
    phrase: str
    frames: list[list[float]] = Field(min_length=2)
    dims: dict = Field(default_factory=dict)
    fps: float | None = None
    crops: dict | None = None  # {imgs: [jpeg data URLs], kp: [[4 x [x, y]]], ts: [ms]} when the lip reader is installed


class PredictIn(BaseModel):
    user: str
    frames: list[list[float]] = Field(min_length=2)
    dims: dict = Field(default_factory=lambda: {"bs": 28, "lip": 80})
    crops: dict | None = None
    note: str = ""
    local_time: str = ""
    use_gemini: bool = True


class LogIn(BaseModel):
    user: str
    sentence: str
    method: str = "click"
    mode: str | None = None
    # Optional: the recording behind this sentence. With learn=true it becomes a calibration sample,
    # so a patient who never calibrated gets a personal model just by using Unmute.
    frames: list[list[float]] | None = None
    dims: dict = Field(default_factory=dict)
    fps: float | None = None
    learn: bool = True
    # confirmed=False: the best guess is urgent and the patient has not confirmed yet. The nurse is told
    # "possible"; the confirmed sentence that follows supersedes it. Never learned from.
    confirmed: bool = True
    # Where the sentence was among the 3 cards (0, 1, 2), or -1 when the patient picked it from the touch board
    # because the lip reading did not offer it. Used for the live accuracy on the Proof page.
    rank: int | None = None
    # Id of an earlier log row this one corrects (the patient confirmed the wrong card, then fixed it on the
    # touch board). That row is marked and no longer counts as a lip-reading success.
    corrects: str | None = None


class RelabelIn(BaseModel):
    user: str
    id: str
    phrase: str


class AckIn(BaseModel):
    id: str


# ---------------------------------------------------------------- routes

@app.on_event("startup")
def _warm_gemini():
    # Pick and warm up the Gemini model in the background so the first prediction isn't slow.
    threading.Thread(target=rerank.init, daemon=True).start()
    if avsr.available() and not os.environ.get("UNMUTE_NO_AVSR"):
        threading.Thread(target=_warm_avsr, daemon=True).start()


def _warm_avsr():
    """Load the lip reader and run it once on a blank clip, so the first patient doesn't wait ~5 s."""
    try:
        with _avsr_lock:
            avsr.read_lips(np.zeros((30, 96, 96), np.uint8), PHRASES["phrases"][:3])
    except Exception:
        pass


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/status")
def status():
    rerank.init()
    weight, measured = avsr_weight()
    return {"gemini": rerank.status(), "ntfy_topic": NTFY_TOPIC, "bed": BED, "model": model_info(),
            "lip_reader": {**avsr.status(), "weight": weight, "weight_measured": measured}}


@app.get("/api/model")
def model_report():
    """The trained model's leave-one-person-out report (Proof page), whether or not it was deployed."""
    f = MODEL_PATH.with_suffix(".json")
    if not f.exists():
        return {"report": None}
    info = json.loads(f.read_text())
    return {"report": info.get("meta"), "phrases": info.get("phrases")}


@app.get("/api/phrases")
def phrases():
    return PHRASES


@app.get("/api/users")
def users():
    by_user = load_samples()
    return {u: len(s) for u, s in sorted(by_user.items())}


@app.get("/api/enrollments")
def enrollments(user: str):
    return calibration_info(safe_user(user))


@app.post("/api/enroll")
def enroll(body: EnrollIn):
    user = safe_user(body.user)
    if body.phrase not in PHRASES["phrases"]:
        raise HTTPException(400, "unknown phrase")
    D = len(body.frames[0])
    if any(len(f) != D for f in body.frames):
        raise HTTPException(400, "all frames must have the same length")
    n_bs = int((body.dims or {}).get("bs", 0))
    motion = features.lip_motion(features.to_array(body.frames), n_bs)
    if motion < MIN_LIP_MOTION:
        raise HTTPException(422, "Your lips barely moved. Mouth it bigger and slower, then try again.")
    sid = f"{user}__{slug(body.phrase)}__{int(time.time() * 1000)}"
    rec = {
        "id": sid,
        "user": user,
        "phrase": body.phrase,
        "frames": body.frames,
        "dims": body.dims,
        "fps": body.fps,
        "created": time.time(),
    }
    (ENROLL_DIR / f"{sid}.json").write_text(json.dumps(rec))
    try:
        crops = decode_crops(body.crops)
        if crops is not None:
            CROPS_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(CROPS_DIR / f"{sid}.npz", crops=crops)
    except Exception:
        pass  # the lip reader's test data is a bonus; never fail a calibration over it
    return {"ok": True, "id": sid, "check": _consistency_check(user, sid, body.phrase)}


def _consistency_check(user: str, sid: str, phrase: str) -> dict | None:
    """Does the new sample look like this person's other samples of the same phrase?
    Inconsistent calibration is the main cause of wrong guesses, so tell them right away."""
    # Only this person's own recordings: cheap however big the shared pool is, and it is exactly the question
    # "does this look like your other samples of the phrase".
    own = load_samples(user).get(user, [])
    if len({s["phrase"] for s in own}) < 2:
        return None
    rec = Recogniser(own, Config())
    ids = [s.get("id") for s in rec.samples]
    own_same = sum(1 for s in rec.samples if s["phrase"] == phrase)
    # With only one other own sample of the phrase the comparison is too noisy to act on: say nothing yet.
    if sid not in ids or own_same < 3:
        return None
    i = ids.index(sid)
    ranked = rec.rank(rec.samples[i]["frames"], top=3, exclude=i)
    names = [r["phrase"] for r in ranked]
    return {"matches": bool(names and names[0] == phrase), "in_top3": phrase in names,
            "looks_like": names[0] if names else None, "others": own_same - 1}


@app.delete("/api/enrollments/last")
def delete_last(user: str):
    user = safe_user(user)
    files = sorted(ENROLL_DIR.glob(f"{user}__*.json"), key=lambda f: f.stat().st_mtime)
    if not files:
        raise HTTPException(404, "nothing to delete")
    s = json.loads(files[-1].read_text())
    files[-1].unlink()
    return {"ok": True, "deleted": s["phrase"]}


@app.post("/api/predict")
async def predict(body: PredictIn):
    user = safe_user(body.user)
    rec = get_recogniser(user)
    if rec is None:
        raise HTTPException(409, "Nobody has calibrated yet, so there is no starter model. Open Calibrate first.")
    info = calibration_info(user)
    if len(body.frames[0]) != rec.templates.shape[2] // (2 if rec.cfg.deltas else 1):
        raise HTTPException(409, "feature size does not match this user's calibration. Re-calibrate.")
    t0 = time.time()
    candidates, recogniser = await asyncio.to_thread(rank_with_model, rec, body.frames, body.dims, info["personal"])
    t_cls = time.time() - t0
    heard, lip_timing = None, None
    lips_moved = features.lip_motion(features.to_array(body.frames), int((body.dims or {}).get("bs", 28))) >= MIN_LIP_MOTION
    if body.crops and lips_moved and avsr.available():
        try:
            crops = await asyncio.to_thread(decode_crops, body.crops)
            if crops is not None:
                def run():
                    with _avsr_lock:
                        return avsr.read_lips(crops, PHRASES["phrases"])
                lip = await asyncio.to_thread(run)
                ranked_all = await asyncio.to_thread(rec.rank, body.frames, 999)
                weight, _ = avsr_weight()
                candidates = blend_with_lip_reader(ranked_all, lip, weight)[:5]
                recogniser = "lip reader+" + recogniser
                lip_timing = {**lip["timing"], "text": lip["text"], "text_conf": lip["text_conf"]}
                text = lip["text"].strip()
                if lip["text_shown"] and text.lower().rstrip(".") not in {p.lower() for p in PHRASES["phrases"]}:
                    heard = text[:120]
        except Exception as e:
            lip_timing = {"error": f"{type(e).__name__}: {e}"[:200]}
    history = [
        {"sentence": r["sentence"], "ago": _ago(r["ts"])}
        for r in reversed(read_requests())
        if r.get("user") == user
    ][:5]
    context = {"time": body.local_time or time.strftime("%A %H:%M"), "history": history, "note": body.note}
    ranked = await rerank.rerank(candidates, context, use_gemini=body.use_gemini)
    return {
        "candidates": candidates,
        "choices": ranked["choices"],
        "mode": ranked["mode"],
        "reason": ranked.get("reason"),
        "source": "personal" if info["personal"] else "starter",
        "recogniser": recogniser,
        "heard": heard,
        "lip_reader": lip_timing,
        "own_samples": info["own_samples"],
        "starter_samples": info["starter_samples"],
        "timing": {"classifier_s": round(t_cls, 3), "total_s": round(time.time() - t0, 3)},
    }


@app.post("/api/log")
async def log(body: LogIn):
    user = safe_user(body.user)
    urgent = alerts.is_urgent(body.sentence, PHRASES["urgent"])
    if not body.confirmed:
        if not urgent:
            raise HTTPException(400, "only urgent phrases can be logged as unconfirmed")
        # Same possible alert already open for this person in the last 90 s: don't push twice.
        for r in reversed(read_requests()):
            if r.get("user") == user and r.get("confirmed") is False and r["sentence"] == body.sentence \
                    and not r["acknowledged"] and time.time() - r["ts"] < 90:
                return r
    row = {
        "id": uuid.uuid4().hex[:12],
        "ts": time.time(),
        "user": user,
        "bed": BED,
        "sentence": body.sentence,
        "urgent": urgent,
        "confirmed": body.confirmed,
        "method": body.method,
        "mode": body.mode,
        "acknowledged": not urgent,
        "push": None,
        "rank": body.rank,
    }
    if urgent:
        text = body.sentence if body.confirmed else f"Possible (not yet confirmed): {body.sentence}"
        ok, detail = await alerts.push_ntfy(NTFY_TOPIC, text, BED)
        row["push"] = detail
    learned_id = _learn_from_use(user, body) if body.confirmed and body.learn and body.frames else None
    row["learned"] = bool(learned_id)
    row["learned_id"] = learned_id
    with _lock:
        rows = read_requests()
        if body.confirmed:
            # The patient has now said what they meant: close this person's open "possible" alerts.
            for r in rows:
                if r.get("user") == user and r.get("confirmed") is False and not r["acknowledged"] \
                        and time.time() - r["ts"] < 180:
                    r["acknowledged"] = True
                    r["superseded_by"] = row["id"]
        if body.corrects:
            for r in rows:
                if r["id"] == body.corrects and r.get("user") == user:
                    r["corrected_to"] = body.sentence
                    row["corrects"] = r["id"]
        rows.append(row)
        write_requests(rows)
    return row


def _learn_from_use(user: str, body: LogIn) -> str | None:
    """A confirmed sentence is a labelled recording of this person's lips: keep it as a sample.
    Only phrases from the list, only if the lips moved, at most MAX_USE_SAMPLES per phrase (oldest go).
    Returns the new sample's id, or None."""
    if body.sentence not in PHRASES["phrases"] or len(body.frames) < 8:
        return None
    try:
        arr = features.to_array(body.frames)
    except ValueError:
        return None
    if features.lip_motion(arr, int((body.dims or {}).get("bs", 0))) < MIN_LIP_MOTION:
        return None
    D = len(body.frames[0])
    if any(len(f) != D for f in body.frames):
        return None
    sid = f"{user}__{slug(body.sentence)}__{int(time.time() * 1000)}"
    rec = {"id": sid, "user": user, "phrase": body.sentence, "frames": body.frames, "dims": body.dims,
           "fps": body.fps, "created": time.time(), "source": "use", "method": body.method}
    with _lock:
        (ENROLL_DIR / f"{sid}.json").write_text(json.dumps(rec))
        _cap_use_samples(user, body.sentence)
    return sid


def _cap_use_samples(user: str, phrase: str):
    used = []
    for f in ENROLL_DIR.glob(f"{user}__{slug(phrase)}__*.json"):
        try:
            if json.loads(f.read_text()).get("source") == "use":
                used.append(f)
        except Exception:
            continue
    for f in sorted(used, key=lambda f: f.name)[:-MAX_USE_SAMPLES]:
        f.unlink()


@app.post("/api/relabel")
def relabel(body: RelabelIn):
    """The patient confirmed the wrong card, then picked the right sentence on the touch board: the recording
    that was just learned under the wrong phrase is moved to the right one, so the mistake teaches the model
    instead of poisoning it. Only samples learned from use can be relabelled, never calibration recordings."""
    user = safe_user(body.user)
    if body.phrase not in PHRASES["phrases"]:
        raise HTTPException(400, "unknown phrase")
    if not re.fullmatch(r"[a-z0-9_-]+__[a-z0-9-]+__\d+", body.id or "") or not body.id.startswith(user + "__"):
        raise HTTPException(400, "bad sample id")
    src = ENROLL_DIR / f"{body.id}.json"
    with _lock:
        if not src.exists():
            raise HTTPException(404, "sample not found")
        rec = json.loads(src.read_text())
        if rec.get("source") != "use":
            raise HTTPException(409, "only samples learned from use can be relabelled")
        old = rec["phrase"]
        sid = f"{user}__{slug(body.phrase)}__{body.id.rsplit('__', 1)[1]}"
        rec.update({"id": sid, "phrase": body.phrase, "relabelled_from": old})
        (ENROLL_DIR / f"{sid}.json").write_text(json.dumps(rec))
        if sid != body.id:
            src.unlink()
        _cap_use_samples(user, body.phrase)
    return {"ok": True, "id": sid, "from": old, "to": body.phrase}


@app.get("/api/usage")
def usage(user: str | None = None):
    """Live accuracy from real use: where the confirmed sentence was among the cards. Only rows the patient
    confirmed with a known card position count (rank 0-2 = lip reading offered it, -1 = picked on the touch board)."""
    rows = [r for r in read_requests() if r.get("confirmed", True) and r.get("rank") is not None
            and not r.get("corrected_to") and (not user or r.get("user") == user)]
    n = len(rows)
    if not n:
        return {"n": 0}
    first = sum(r["rank"] == 0 for r in rows)
    in3 = sum(0 <= r["rank"] <= 2 for r in rows)
    return {"n": n, "top1": round(first / n, 4), "top3": round(in3 / n, 4),
            "touch_board": round((n - in3) / n, 4), "corrections": n - in3}


@app.get("/api/requests")
def requests_(limit: int = 50):
    rows = read_requests()[-limit:]
    rows.reverse()
    for r in rows:
        r["ago"] = _ago(r["ts"])
    return {"requests": rows, "unacknowledged_urgent": sum(1 for r in rows if r["urgent"] and not r["acknowledged"])}


@app.get("/api/requests/{rid}")
def request_one(rid: str):
    for r in read_requests():
        if r["id"] == rid:
            return r
    raise HTTPException(404, "not found")


@app.post("/api/ack")
def ack(body: AckIn):
    with _lock:
        rows = read_requests()
        hit = False
        for r in rows:
            if body.id in (r["id"], "all") and r["urgent"] and not r["acknowledged"]:
                r["acknowledged"] = True
                r["ack_ts"] = time.time()
                hit = True
        if hit:
            write_requests(rows)
    return {"ok": True}


@app.get("/api/results")
def results():
    if not RESULTS_FILE.exists():
        return JSONResponse({"results": None})
    return {"results": json.loads(RESULTS_FILE.read_text())}


@app.post("/api/evaluate")
async def run_evaluation():
    by_user = load_samples()
    res = await asyncio.to_thread(evaluate_all, by_user, Config(), False)
    RESULTS_FILE.write_text(json.dumps(res, indent=2))
    return {"results": res}


# Static frontend last, so /api routes win.
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
