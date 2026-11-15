"""Privacy, checked rather than claimed: only short text ever leaves the laptop.

The browser sends the local server (127.0.0.1, same laptop) about 108 numbers per video frame, plus a small picture
of the mouth per frame while the lip reader is switched on. Exactly two things may go to the internet, both
optional: Gemini (to reorder the candidate sentences) and ntfy (the urgent push to the nurse's phone).

This runs the whole flow (mouth a sentence with numbers AND mouth pictures, get cards, confirm an urgent sentence),
captures what is handed to Gemini and the exact HTTP request ntfy would get, and checks that neither holds a
picture, a landmark number, or the patient's name; and that a patient's mouth pictures are never written to disk
(only Calibrate recordings keep theirs, on this laptop, for measuring the lip reader).
"""
import json
import re

import numpy as np

import server.alerts as alerts
import server.app as appmod
import server.rerank as R
from synth import make_recording
from test_api import calibrate

REAL_PUSH = alerts.push_ntfy  # captured before the autouse fixture stubs it
NAME = "zelda-patient"
JPEG = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ" + "A" * 3000  # stands in for a 192x192 mouth picture


def _picture_payload(n=24):
    return {"imgs": [JPEG] * n, "kp": [[0.0] * 8] * n, "ts": [i * 40.0 for i in range(n)]}


def _assert_no_private_data(text: str, frames):
    assert "/9j/" not in text and "base64" not in text, "a picture left the laptop"
    assert NAME not in text, "the patient's name left the laptop"
    # landmark numbers: the frames are long floats; nothing outgoing has more than 3 decimals
    assert not re.search(r"\d\.\d{4,}", text), "landmark-like numbers left the laptop"
    for v in (frames[3][5], frames[10][40], frames[-1][90]):
        assert repr(v) not in text


def test_only_text_leaves_the_laptop(client, tmp_path, monkeypatch):
    calibrate(client, "ann", n=2)                       # someone else's recordings: the starter pool
    monkeypatch.setattr(appmod, "CROPS_DIR", tmp_path / "crops")

    # The lip reader path runs, with its pictures, as on the laptop (the model itself is faked: no GPU in tests).
    got = {}
    monkeypatch.setattr(appmod.avsr, "available", lambda: True)
    monkeypatch.setattr(appmod, "decode_crops", lambda c: got.setdefault("crops", c) and np.zeros((24, 96, 96), np.uint8))
    phrases = appmod.PHRASES["phrases"]

    def fake_read_lips(crops, ps, free_text=True):
        scores = {p: (-1.0 if p == "I am in pain" else -9.0) for p in ps}
        p = np.exp(np.array(list(scores.values()))); p /= p.sum()
        return {"scores": scores, "proba": dict(zip(scores, p.tolist())), "text": "where is my phone",
                "text_conf": -0.3, "text_total": -1.5, "text_shown": True, "text_first": True,
                "timing": {"score_s": 0.1, "text_s": 0.5, "frames": len(crops)}}
    monkeypatch.setattr(appmod.avsr, "read_lips", fake_read_lips)

    # Gemini "online": capture exactly what it would be sent.
    monkeypatch.setattr(R, "_client", object())
    monkeypatch.setattr(R, "_models", ["m1"])
    monkeypatch.setattr(R, "_rest_until", {})
    monkeypatch.setattr(R, "_model_error", None)
    sent_to_gemini = []

    def fake_call(prompt, model):
        sent_to_gemini.append(prompt)
        return {"choices": [], "reason": "test"}
    monkeypatch.setattr(R, "_call_gemini", fake_call)

    # ntfy: the real function, with the HTTP client replaced, so we see the exact request.
    sent_to_ntfy = []

    class FakeHTTP:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, content=b"", headers=None):
            sent_to_ntfy.append({"url": url, "body": content.decode(), "headers": dict(headers or {})})
            return type("R", (), {"status_code": 200})()
    monkeypatch.setattr(alerts, "push_ntfy", REAL_PUSH)
    monkeypatch.setattr(alerts.httpx, "AsyncClient", FakeHTTP)

    q = make_recording(1, seed=7)
    r = client.post("/api/predict", json={"user": NAME, "frames": q["frames"], "dims": q["dims"],
                                          "crops": _picture_payload(), "note": "post-op", "use_gemini": True})
    assert r.status_code == 200, r.text
    out = r.json()
    assert got.get("crops"), "the mouth pictures should have reached the LOCAL server"
    assert out["recogniser"].startswith("lip reader+") and out["mode"] == "gemini"
    r = client.post("/api/log", json={"user": NAME, "sentence": "I am in pain", "method": "nod", "mode": "gemini",
                                      "rank": 0, "frames": q["frames"], "dims": q["dims"], "learn": True})
    assert r.status_code == 200, r.text

    # What went out: text only, and small.
    assert len(sent_to_gemini) == 1 and len(sent_to_ntfy) == 1
    prompt = sent_to_gemini[0]
    _assert_no_private_data(prompt, q["frames"])
    assert len(prompt) < 2500 and "I am in pain" in prompt
    push = sent_to_ntfy[0]
    _assert_no_private_data(json.dumps(push), q["frames"])
    assert push["body"] == f'{appmod.BED}: "I am in pain"' and len(push["body"]) < 200
    assert push["url"].startswith("https://ntfy.sh/unmute-")

    # Nothing visual written to disk for a patient: no crops, and the log and the learned sample hold no picture.
    assert not (tmp_path / "crops").exists()
    log_text = appmod.REQUESTS_FILE.read_text()
    assert "/9j/" not in log_text and "crops" not in log_text
    learned = json.loads((appmod.ENROLL_DIR / f"{r.json()['learned_id']}.json").read_text())
    assert "crops" not in learned and learned["frames"] == q["frames"]  # numbers, kept on this laptop


def test_offline_nothing_leaves_at_all(client, monkeypatch):
    """Gemini off (no key / no network / switched off): predict makes no outgoing call at all."""
    calibrate(client, "ann", n=2)
    called = []
    monkeypatch.setattr(R, "_call_gemini", lambda *a: called.append(a))
    q = make_recording(1, seed=7)
    r = client.post("/api/predict", json={"user": NAME, "frames": q["frames"], "dims": q["dims"], "use_gemini": False})
    assert r.status_code == 200 and r.json()["mode"] == "offline" and not called
