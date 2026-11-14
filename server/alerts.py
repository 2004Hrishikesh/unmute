"""Urgent-phrase check and ntfy.sh push to the nurse's phone."""
from __future__ import annotations

import httpx

NTFY_URL = "https://ntfy.sh"


def is_urgent(sentence: str, urgent: list[str]) -> bool:
    s = sentence.strip().lower()
    return any(s == u.lower() for u in urgent)


async def push_ntfy(topic: str, sentence: str, bed: str) -> tuple[bool, str]:
    """Send an urgent push. Returns (ok, detail). Never raises."""
    if not topic:
        return False, "no topic configured"
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.post(
                f"{NTFY_URL}/{topic}",
                content=f"{bed}: \"{sentence}\"".encode("utf-8"),
                headers={
                    "Title": "Unmute: urgent request",
                    "Priority": "urgent",
                    "Tags": "rotating_light",
                },
            )
        return r.status_code < 300, f"ntfy HTTP {r.status_code}"
    except Exception as e:  # offline, DNS, timeout...
        return False, f"ntfy failed: {type(e).__name__}"
