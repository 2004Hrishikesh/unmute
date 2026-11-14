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

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import alerts, features, rerank
from .classifier import Config, Recogniser
from .evaluation import evaluate_all

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ENROLL_DIR = DATA / "enrollments"
REQUESTS_FILE = DATA / "requests.jsonl"
RESULTS_FILE = DATA / "results.json"
CONFIG_FILE = DATA / "config.json"
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
MIN_LIP_MOTION = 0.012  # below this the lips barely moved (eye-distance units)

PHRASES = json.loads((DATA / "phrases.json").read_text())
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


def _fingerprint(user: str) -> tuple:
    files = sorted(ENROLL_DIR.glob(f"{user}__*.json"))
    return (len(files), max((f.stat().st_mtime for f in files), default=0))


def get_recogniser(user: str) -> Recogniser | None:
    fp = _fingerprint(user)
    cached = _rec_cache.get(user)
    if cached and cached[0] == fp:
        return cached[1]
    samples = load_samples(user).get(user, [])
    if len({s["phrase"] for s in samples}) < 2:
        return None
    rec = Recogniser(samples, Config())
    _rec_cache[user] = (fp, rec)
    return rec


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


class PredictIn(BaseModel):
    user: str
    frames: list[list[float]] = Field(min_length=2)
    note: str = ""
    local_time: str = ""
    use_gemini: bool = True


class LogIn(BaseModel):
    user: str
    sentence: str
    method: str = "click"
    mode: str | None = None


class AckIn(BaseModel):
    id: str


# ---------------------------------------------------------------- routes

@app.on_event("startup")
def _warm_gemini():
    # Pick and warm up the Gemini model in the background so the first prediction isn't slow.
    threading.Thread(target=rerank.init, daemon=True).start()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/status")
def status():
    rerank.init()
    return {"gemini": rerank.status(), "ntfy_topic": NTFY_TOPIC, "bed": BED}


@app.get("/api/phrases")
def phrases():
    return PHRASES


@app.get("/api/users")
def users():
    by_user = load_samples()
    return {u: len(s) for u, s in sorted(by_user.items())}


@app.get("/api/enrollments")
def enrollments(user: str):
    user = safe_user(user)
    counts: dict[str, int] = {}
    for s in load_samples(user).get(user, []):
        counts[s["phrase"]] = counts.get(s["phrase"], 0) + 1
    return {"user": user, "counts": counts}


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
    return {"ok": True, "id": sid, "check": _consistency_check(user, sid, body.phrase)}


def _consistency_check(user: str, sid: str, phrase: str) -> dict | None:
    """Does the new sample look like this person's other samples of the same phrase?
    Inconsistent calibration is the main cause of wrong guesses, so tell them right away."""
    rec = get_recogniser(user)
    if rec is None:
        return None
    ids = [s.get("id") for s in rec.samples]
    if sid not in ids or rec.phrases.count(phrase) < 2:
        return None
    i = ids.index(sid)
    ranked = rec.rank(rec.samples[i]["frames"], top=3, exclude=i)
    names = [r["phrase"] for r in ranked]
    return {"matches": bool(names and names[0] == phrase), "in_top3": phrase in names, "looks_like": names[0] if names else None}


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
        raise HTTPException(409, f"'{user}' has not enrolled at least 2 phrases yet. Open Calibrate first.")
    if len(body.frames[0]) != rec.templates.shape[2] // (2 if rec.cfg.deltas else 1):
        raise HTTPException(409, "feature size does not match this user's calibration. Re-calibrate.")
    t0 = time.time()
    candidates = await asyncio.to_thread(rec.rank, body.frames, 5)
    t_cls = time.time() - t0
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
        "timing": {"classifier_s": round(t_cls, 3), "total_s": round(time.time() - t0, 3)},
    }


@app.post("/api/log")
async def log(body: LogIn):
    user = safe_user(body.user)
    urgent = alerts.is_urgent(body.sentence, PHRASES["urgent"])
    row = {
        "id": uuid.uuid4().hex[:12],
        "ts": time.time(),
        "user": user,
        "bed": BED,
        "sentence": body.sentence,
        "urgent": urgent,
        "method": body.method,
        "mode": body.mode,
        "acknowledged": not urgent,
        "push": None,
    }
    if urgent:
        ok, detail = await alerts.push_ntfy(NTFY_TOPIC, body.sentence, BED)
        row["push"] = detail
    with _lock:
        with REQUESTS_FILE.open("a") as f:
            f.write(json.dumps(row) + "\n")
    return row


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
