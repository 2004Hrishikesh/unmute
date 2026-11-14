"""Gemini re-ranking with bedside context, with an offline fallback.

If GEMINI_API_KEY is missing, the network is down, the call takes longer than
TIMEOUT_S, or the reply is not valid, we fall back to the classifier's top 3
and report mode="offline".
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from pathlib import Path

TIMEOUT_S = 3.0
STATE_FILE: Path | None = None  # set by the app: remembers which models are resting, across restarts
URGENT: set[str] = set()         # set by the app from data/phrases.json

_client = None
_models: list[str] = []          # every usable Flash model, in preference order
_rest_until: dict[str, float] = {}  # model -> unix time before which we don't use it
_model_error = None
_failed_at = 0.0
_lock = threading.Lock()

_SKIP = ("image", "tts", "live", "audio", "embedding", "thinking", "exp", "8b", "native")


def _version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def _flash_models(client) -> list[str]:
    """Every Flash model the key can call (listing models costs no generate quota; no guessed names).

    Order: "lite" models first, newest first, then the full Flash models. We measured lite at
    about 0.5-1.3 s per call and the full models at 1.3-7 s (often 503 "high demand"), and our
    budget is TIMEOUT_S. GEMINI_MODEL, if set, goes first.
    """
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            continue
        n = m.name.split("/")[-1]
        if "flash" in n and not any(s in n for s in _SKIP):
            names.append(n)
    stable = [n for n in names if "preview" not in n] or names
    stable.sort(key=lambda n: ("lite" not in n, -_version(n), len(n)))
    pref = os.environ.get("GEMINI_MODEL")
    if pref:
        stable = [pref] + [n for n in stable if n != pref]
    if not stable:
        raise RuntimeError("no Gemini Flash model available to this key")
    return stable


def _load_rest():
    if STATE_FILE and STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            now = time.time()
            _rest_until.update({m: t for m, t in data.get("rest_until", {}).items() if t > now})
        except Exception:
            pass


def _save_rest():
    if STATE_FILE:
        try:
            now = time.time()
            STATE_FILE.write_text(json.dumps(
                {"rest_until": {m: t for m, t in _rest_until.items() if t > now}}, indent=2))
        except Exception:
            pass


def _rest_seconds(msg: str) -> float:
    """How long to leave a model alone after this error."""
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:  # free-tier quota (per model, often per day)
        m = re.search(r"retryDelay\W+(\d+)s", msg)
        return min(float(m.group(1)), 86400.0) if m else 60.0
    if "404" in msg or "NOT_FOUND" in msg:  # retired for this key
        return 7 * 86400.0
    if "503" in msg or "UNAVAILABLE" in msg or "500" in msg or "INTERNAL" in msg:  # overloaded
        return 60.0
    return 0.0


def _rest(model: str, msg: str):
    secs = _rest_seconds(msg)
    if secs > 0:
        with _lock:
            _rest_until[model] = time.time() + secs
            _save_rest()


def _current_model() -> str | None:
    now = time.time()
    for m in _models:
        if _rest_until.get(m, 0) <= now:
            return m
    return None


def init():
    """Create the client and list the models once. Safe to call often. Spends no generate quota."""
    global _client, _models, _model_error, _failed_at
    with _lock:
        if _client is not None:
            return
        # After a failure (network blip) try again, but at most every 20 s.
        if _model_error is not None and time.time() - _failed_at < 20:
            return
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            _model_error = "GEMINI_API_KEY not set"
            _failed_at = time.time()
            return
        try:
            from google import genai
            from google.genai import types

            # No hidden SDK retries (they can stall for many seconds on a 503). The 3 s cutoff is
            # enforced by us, not by the SDK's timeout option, which Google rejects below 10 s.
            client = genai.Client(api_key=key, http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)))
            _models = _flash_models(client)
            _load_rest()
            _client = client
            _model_error = None
        except Exception as e:
            _client = None
            _model_error = f"{type(e).__name__}: {e}"[:200]
            _failed_at = time.time()


def status() -> dict:
    model = _current_model() if _client is not None else None
    now = time.time()
    resting = {m: int(t - now) for m, t in _rest_until.items() if t > now}
    error = _model_error
    if _client is not None and model is None:
        error = "every Gemini Flash model is busy or out of free quota right now; using offline mode"
    return {"available": model is not None, "model": model, "error": error, "resting_s": resting}


def _build_prompt(candidates, context) -> str:
    cand_lines = "\n".join(
        f'- "{c["phrase"]}" (lip-reading confidence {c["confidence"]:.2f})' for c in candidates
    )
    hist = context.get("history") or []
    hist_lines = "\n".join(f'- "{h["sentence"]}" {h["ago"]}' for h in hist) or "- (none)"
    note = (context.get("note") or "").strip() or "(none)"
    return f"""You help a hospital patient who cannot speak. A lip-reading system watched them
silently mouth one sentence and produced these candidate phrases, best first:
{cand_lines}

Bedside context:
- Local time: {context.get("time") or "unknown"}
- Recent confirmed requests (most recent first):
{hist_lines}
- Care note: {note}

Choose the 3 candidates most likely to be what the patient said, ranked.
Rules:
- Lip-reading confidence is the main evidence. Only use context to break near-ties.
- A request that was just made and confirmed a few minutes ago is less likely to be repeated,
  unless it is urgent.
- Always keep the first candidate (the best lip-reading match) somewhere in your 3 choices.
- Never leave out an urgent request (breathing, pain, help, nurse, feeling sick) that is in the top 3.
- Every choice must be copied exactly from the candidate list. No new sentences.
Reply as JSON: {{"choices": ["...", "...", "..."], "reason": "one short sentence"}}"""


def _call_gemini(prompt: str, model: str) -> dict:
    from google.genai import types

    resp = _client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    return json.loads(resp.text)


def _validate(choices, candidates) -> tuple[list[str], bool]:
    """Keep only real candidates, and enforce the safety rule: Gemini may reorder, but it can never
    remove the lip-reading top choice, or an urgent phrase the lip reading put in its top 3, from
    the 3 choices. (Seen in testing: Gemini dropped "I need water" because water was "just given".)
    Returns (choices, overridden)."""
    allowed = {c["phrase"].lower(): c["phrase"] for c in candidates}
    out = []
    for ch in choices or []:
        p = allowed.get(str(ch).strip().lower())
        if p and p not in out:
            out.append(p)
    # Top up from the classifier order if Gemini returned fewer than 3 valid choices.
    for c in candidates:
        if len(out) >= 3:
            break
        if c["phrase"] not in out:
            out.append(c["phrase"])
    out = out[:3]
    must = []
    for c in [candidates[0]] + [c for c in candidates[:3] if c["phrase"] in URGENT]:
        if c["phrase"] not in must:
            must.append(c["phrase"])
    overridden = False
    for m in must:
        if m in out:
            continue
        for i in range(len(out) - 1, -1, -1):  # replace Gemini's lowest pick that isn't protected
            if out[i] not in must:
                out[i] = m
                overridden = True
                break
    return out, overridden


async def rerank(candidates: list[dict], context: dict, use_gemini: bool = True) -> dict:
    offline = {"choices": [c["phrase"] for c in candidates[:3]], "mode": "offline", "reason": None}
    if not use_gemini or len(candidates) <= 1:
        return offline
    init()
    if _client is None:
        return {**offline, "reason": _model_error}
    prompt = _build_prompt(candidates, context)
    deadline = time.monotonic() + TIMEOUT_S
    reason = "every Gemini Flash model is busy or out of free quota right now"
    # Quota / overload errors come back in ~0.1 s, so there is usually time to try the next model.
    for _ in range(3):
        model = _current_model()
        left = deadline - time.monotonic()
        if model is None or left < 0.4:
            break
        try:
            data = await asyncio.wait_for(asyncio.to_thread(_call_gemini, prompt, model), timeout=left)
            choices, overridden = _validate(data.get("choices"), candidates)
            reason = str(data.get("reason") or "")[:200]
            if overridden:
                reason += " (Unmute kept the top lip-reading match in the list.)"
            return {"choices": choices, "mode": "gemini", "reason": reason, "model": model}
        except asyncio.TimeoutError:
            reason = f"Gemini timed out after {TIMEOUT_S}s"
            break
        except Exception as e:
            msg = str(e)
            _rest(model, msg)
            code = re.search(r"\b(4\d\d|5\d\d)\b", msg)
            reason = f"Gemini error on {model}: {code.group(1) if code else type(e).__name__}"
            if _rest_seconds(msg) == 0:
                break  # not a per-model problem (e.g. bad JSON, bad key): don't burn more calls
    return {**offline, "reason": reason}
