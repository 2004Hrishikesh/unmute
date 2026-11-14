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

TIMEOUT_S = 3.0

_client = None
_model = None
_model_error = None
_lock = threading.Lock()
_failed_at = 0.0

_SKIP = ("image", "tts", "live", "audio", "embedding", "thinking", "exp", "8b", "native")


def _version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def _flash_models(client) -> list[str]:
    """Every Flash model the key can call, newest first; "lite" variants last (no guessing names)."""
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            continue
        n = m.name.split("/")[-1]
        if "flash" in n and not any(s in n for s in _SKIP):
            names.append(n)
    if not names:
        raise RuntimeError("no Gemini Flash model available to this key")
    stable = [n for n in names if "preview" not in n]
    pool = stable or names
    # Full models newest first, then lite models newest first.
    pool.sort(key=lambda n: ("lite" in n, -_version(n), len(n)))
    return pool


_PROBE_S = 2.5
_models: list[str] = []


def _probe(client, model: str) -> bool:
    """One tiny real request: is this model up and fast enough right now?"""
    from google.genai import types

    ok = threading.Event()

    def call():
        try:
            client.models.generate_content(
                model=model, contents='Return JSON {"ok": true}',
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
            )
            ok.set()
        except Exception:
            pass

    threading.Thread(target=call, daemon=True).start()
    return ok.wait(_PROBE_S)


def _pick_working_model(client) -> str:
    global _models
    _models = _flash_models(client)
    for m in _models:
        if _probe(client, m):
            return m
    raise RuntimeError("every Gemini Flash model is busy or too slow right now: " + ", ".join(_models[:4]))


def _next_model(failed: str):
    """The current model failed (overloaded / retired): move to the next one for later requests."""
    global _model
    with _lock:
        if _model != failed or failed not in _models:
            return
        i = _models.index(failed)
        if i + 1 < len(_models):
            _model = _models[i + 1]


def init():
    """Create the client and choose the model once. Safe to call repeatedly."""
    global _client, _model, _model_error, _failed_at
    with _lock:
        if _client is not None:
            return
        # After a failure (network blip, temporary 403/503) try again, but at most every 20 s.
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
            _client = genai.Client(api_key=key, http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)))
            _model = os.environ.get("GEMINI_MODEL") or _pick_working_model(_client)
            _model_error = None
        except Exception as e:
            _client = None
            _model_error = f"{type(e).__name__}: {e}"[:200]
            _failed_at = time.time()


def status() -> dict:
    return {"available": _client is not None and _model is not None, "model": _model, "error": _model_error}


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
- Never drop a candidate with clearly the highest confidence out of the 3.
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


def _validate(choices, candidates) -> list[str]:
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
    return out[:3]


async def rerank(candidates: list[dict], context: dict, use_gemini: bool = True) -> dict:
    offline = {"choices": [c["phrase"] for c in candidates[:3]], "mode": "offline", "reason": None}
    if not use_gemini or len(candidates) <= 1:
        return offline
    init()
    if _client is None:
        return {**offline, "reason": _model_error}
    model = _model
    try:
        prompt = _build_prompt(candidates, context)
        data = await asyncio.wait_for(asyncio.to_thread(_call_gemini, prompt, model), timeout=TIMEOUT_S)
        return {
            "choices": _validate(data.get("choices"), candidates),
            "mode": "gemini",
            "reason": str(data.get("reason") or "")[:200],
            "model": model,
        }
    except asyncio.TimeoutError:
        return {**offline, "reason": f"Gemini timed out after {TIMEOUT_S}s"}
    except Exception as e:
        msg = str(e)
        if any(code in msg for code in ("503", "404", "429", "UNAVAILABLE", "NOT_FOUND")):
            _next_model(model)  # overloaded or retired: use the next Flash model from now on
        return {**offline, "reason": f"Gemini error: {type(e).__name__} ({msg[:60]})"}
