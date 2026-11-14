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

TIMEOUT_S = 3.0

_client = None
_model = None
_model_error = None
_lock = threading.Lock()

_SKIP = ("lite", "image", "tts", "live", "audio", "embedding", "thinking", "exp", "8b", "native")


def _version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def _pick_flash_model(client) -> str:
    """Pick the newest Gemini Flash model the key can call (no guessing names)."""
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
    # Newest version first; prefer the plain alias (shortest name) within a version.
    pool.sort(key=lambda n: (-_version(n), len(n)))
    return pool[0]


def init():
    """Create the client and choose the model once. Safe to call repeatedly."""
    global _client, _model, _model_error
    with _lock:
        if _client is not None or _model_error is not None:
            return
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            _model_error = "GEMINI_API_KEY not set"
            return
        try:
            from google import genai

            _client = genai.Client(api_key=key)
            _model = os.environ.get("GEMINI_MODEL") or _pick_flash_model(_client)
        except Exception as e:
            _client = None
            _model_error = f"{type(e).__name__}: {e}"[:200]


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


def _call_gemini(prompt: str) -> dict:
    from google.genai import types

    resp = _client.models.generate_content(
        model=_model,
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
    try:
        prompt = _build_prompt(candidates, context)
        data = await asyncio.wait_for(asyncio.to_thread(_call_gemini, prompt), timeout=TIMEOUT_S)
        return {
            "choices": _validate(data.get("choices"), candidates),
            "mode": "gemini",
            "reason": str(data.get("reason") or "")[:200],
            "model": _model,
        }
    except asyncio.TimeoutError:
        return {**offline, "reason": f"Gemini timed out after {TIMEOUT_S}s"}
    except Exception as e:
        return {**offline, "reason": f"Gemini error: {type(e).__name__}"}
