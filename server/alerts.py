"""Urgent-phrase check and ntfy.sh push to the nurse's phone."""
from __future__ import annotations

import os

import httpx

# ntfy is open source: a hospital can run its own server on its own network and point Unmute at it.
NTFY_URL = os.environ.get("NTFY_URL", "https://ntfy.sh").rstrip("/")


def is_urgent(sentence: str, urgent: list[str]) -> bool:
    s = sentence.strip().lower()
    return any(s == u.lower() for u in urgent)


async def push_ntfy(topic: str, sentence: str, bed: str, note: str = "", title: str = "Unmute: urgent request",
                    priority: str = "urgent", tags: str = "rotating_light") -> tuple[bool, str]:
    """Send a push to the nurse's phone. Returns (ok, detail). Never raises."""
    if not topic:
        return False, "no topic configured"
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.post(
                f"{NTFY_URL}/{topic}",
                content=(f"{bed}: \"{sentence}\"" + (f" {note}" if note else "")).encode("utf-8"),
                headers={"Title": title, "Priority": priority, "Tags": tags},
            )
        return r.status_code < 300, f"ntfy HTTP {r.status_code}"
    except Exception as e:  # offline, DNS, timeout...
        return False, f"ntfy failed: {type(e).__name__}"
