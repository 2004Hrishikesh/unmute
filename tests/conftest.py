"""Test setup: no real data, no network.

Every test runs against a temporary enrollments folder, request log and results file, with Gemini
disabled (empty key) and the ntfy push replaced by a stub.
"""
from __future__ import annotations

import os

# Must happen before `server.app` is imported: app.py copies GEMINI_API_KEY from .env into the
# environment only when the variable is absent, and an empty key means "offline mode".
os.environ["GEMINI_API_KEY"] = ""
os.environ["NTFY_TOPIC"] = "unmute-test-topic"

import pytest  # noqa: E402

import server.alerts as alerts  # noqa: E402
import server.app as appmod  # noqa: E402
import server.rerank as rerank  # noqa: E402

from synth import make_recording  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Point the app at throwaway files and reset caches and Gemini state."""
    enroll = tmp_path / "enrollments"
    enroll.mkdir()
    monkeypatch.setattr(appmod, "ENROLL_DIR", enroll)
    monkeypatch.setattr(appmod, "REQUESTS_FILE", tmp_path / "requests.jsonl")
    monkeypatch.setattr(appmod, "RESULTS_FILE", tmp_path / "results.json")
    appmod._rec_cache.clear()
    monkeypatch.setattr(appmod, "MODEL_PATH", tmp_path / "model" / "lipnet")
    appmod._model_cache.update(mtime=None, net=None)

    monkeypatch.setattr(rerank, "STATE_FILE", None)
    monkeypatch.setattr(rerank, "_client", None)
    monkeypatch.setattr(rerank, "_models", [])
    monkeypatch.setattr(rerank, "_rest_until", {})
    monkeypatch.setattr(rerank, "_model_error", None)
    monkeypatch.setattr(rerank, "_failed_at", 0.0)
    monkeypatch.setenv("GEMINI_API_KEY", "")

    async def fake_push(topic, sentence, bed):
        return False, "test"

    monkeypatch.setattr(alerts, "push_ntfy", fake_push)
    yield tmp_path
    appmod._rec_cache.clear()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    # Not used as a context manager on purpose: that keeps the startup hook (Gemini warm-up thread)
    # from running.
    return TestClient(appmod.app)


@pytest.fixture
def recording():
    """recording(phrase_index, seed) -> {"frames", "dims", "fps"} with clearly moving lips."""
    return make_recording
