import asyncio
import time

import pytest

import server.rerank as R

URGENT = {"I cannot breathe", "I am in pain", "Help me", "Call the nurse", "I feel sick"}


def cands(*phrases):
    return [{"phrase": p, "distance": 1 + i / 10, "confidence": round(0.3 - i / 20, 2)} for i, p in enumerate(phrases)]


@pytest.fixture(autouse=True)
def urgent_set(monkeypatch):
    monkeypatch.setattr(R, "URGENT", set(URGENT))


# ---------------------------------------------------------------- _validate

def test_validate_reinserts_dropped_top_match():
    out, overridden = R._validate(["I am in pain", "Call my family", "Yes"],
                                  cands("I need water", "I am in pain", "Call my family", "Yes", "No"))
    assert overridden is True
    assert "I need water" in out and len(out) == 3
    assert out[0] == "I am in pain"  # Gemini's order is kept where it is allowed


def test_validate_reinserts_dropped_urgent_phrase_from_top3():
    out, overridden = R._validate(["I need water", "Yes", "No"],
                                  cands("I need water", "Yes", "I cannot breathe", "No", "Thank you"))
    assert overridden is True
    assert out[0] == "I need water" and "I cannot breathe" in out and len(out) == 3


def test_validate_leaves_a_pure_reorder_alone():
    out, overridden = R._validate(["I am in pain", "I need water", "Call my family"],
                                  cands("I need water", "I am in pain", "Call my family", "Yes", "No"))
    assert overridden is False
    assert out == ["I am in pain", "I need water", "Call my family"]


def test_validate_ignores_invented_sentences_and_tops_up():
    out, overridden = R._validate(["Please bring me coffee", "no", 42],
                                  cands("I need water", "I am in pain", "Call my family", "Yes", "No"))
    assert "Please bring me coffee" not in out
    assert out[0] == "No"  # case-insensitive match to a real candidate
    assert len(out) == 3 and len(set(out)) == 3
    assert overridden is False  # top match and the urgent #2 arrived through the top-up


def test_validate_with_fewer_than_three_candidates():
    out, _ = R._validate(None, cands("Yes", "No"))
    assert out == ["Yes", "No"]


# ---------------------------------------------------------------- resting rules

def test_rest_seconds_by_error_type():
    msg429 = ("429 RESOURCE_EXHAUSTED. {'error': {'details': [{'@type': 'RetryInfo', 'retryDelay': '41367s'}]}}")
    assert R._rest_seconds(msg429) == 41367.0
    assert R._rest_seconds("429 RESOURCE_EXHAUSTED no delay given") == 60.0
    assert R._rest_seconds("404 NOT_FOUND model retired") == 7 * 86400.0
    assert R._rest_seconds("503 UNAVAILABLE high demand") == 60.0
    assert R._rest_seconds("400 INVALID_ARGUMENT") == 0.0


def test_current_model_skips_resting_models(monkeypatch):
    monkeypatch.setattr(R, "_models", ["a", "b", "c"])
    monkeypatch.setattr(R, "_rest_until", {"a": time.time() + 100, "b": time.time() - 1})
    assert R._current_model() == "b"
    R._rest_until["b"] = time.time() + 100
    R._rest_until["c"] = time.time() + 100
    assert R._current_model() is None


# ---------------------------------------------------------------- rerank()

CTX = {"time": "03:10", "history": [], "note": ""}


def run(coro):
    return asyncio.run(coro)


def test_rerank_offline_when_disabled():
    out = run(R.rerank(cands("A", "B", "C", "D"), CTX, use_gemini=False))
    assert out == {"choices": ["A", "B", "C"], "mode": "offline", "reason": None}


def test_rerank_offline_without_key_gives_reason():
    out = run(R.rerank(cands("A", "B", "C", "D"), CTX))
    assert out["mode"] == "offline"
    assert out["choices"] == ["A", "B", "C"]
    assert "GEMINI_API_KEY" in (out["reason"] or "")
    assert R.status()["available"] is False


def _fake_online(monkeypatch, models=("m1", "m2")):
    monkeypatch.setattr(R, "_client", object())
    monkeypatch.setattr(R, "_models", list(models))
    monkeypatch.setattr(R, "_rest_until", {})
    monkeypatch.setattr(R, "_model_error", None)


def test_rerank_uses_gemini_answer_after_validation(monkeypatch):
    _fake_online(monkeypatch)
    seen = {}

    def fake_call(prompt, model):
        seen["model"] = model
        seen["prompt"] = prompt
        return {"choices": ["Call my family", "I need water", "made up"], "reason": "test"}

    monkeypatch.setattr(R, "_call_gemini", fake_call)
    out = run(R.rerank(cands("I need water", "I am in pain", "Call my family", "Yes", "No"),
                       {"time": "03:10", "history": [{"sentence": "Yes", "ago": "2 min ago"}], "note": "post-op"}))
    assert out["mode"] == "gemini" and out["model"] == "m1" == seen["model"]
    assert out["choices"][:2] == ["Call my family", "I need water"]
    assert "made up" not in out["choices"] and len(out["choices"]) == 3
    assert "I am in pain" in out["choices"]  # urgent phrase from the lip-reading top 3 was put back
    assert "post-op" in seen["prompt"] and "2 min ago" in seen["prompt"]
    assert R.status() == {"available": True, "model": "m1", "error": None, "resting_s": {}}


def test_rerank_rests_an_overloaded_model_and_moves_on(monkeypatch):
    _fake_online(monkeypatch)
    calls = []

    def fake_call(prompt, model):
        calls.append(model)
        if model == "m1":
            raise RuntimeError("503 UNAVAILABLE. high demand")
        return {"choices": ["B", "A", "C"], "reason": "ok"}

    monkeypatch.setattr(R, "_call_gemini", fake_call)
    out = run(R.rerank(cands("A", "B", "C", "D"), CTX))
    assert calls == ["m1", "m2"]
    assert out["mode"] == "gemini" and out["model"] == "m2" and out["choices"] == ["B", "A", "C"]
    assert R._rest_until["m1"] > time.time()
    assert "m1" in R.status()["resting_s"] and R.status()["model"] == "m2"
    # next call goes straight to m2
    calls.clear()
    run(R.rerank(cands("A", "B", "C", "D"), CTX))
    assert calls == ["m2"]


def test_rerank_offline_when_every_model_fails(monkeypatch):
    _fake_online(monkeypatch)

    def fake_call(prompt, model):
        raise RuntimeError("503 UNAVAILABLE")

    monkeypatch.setattr(R, "_call_gemini", fake_call)
    out = run(R.rerank(cands("A", "B", "C", "D"), CTX))
    assert out["mode"] == "offline" and out["choices"] == ["A", "B", "C"]
    assert "503" in out["reason"] or "busy" in out["reason"]
    assert R.status()["available"] is False


def test_rerank_does_not_rest_on_non_model_errors(monkeypatch):
    _fake_online(monkeypatch)
    calls = []

    def fake_call(prompt, model):
        calls.append(model)
        raise ValueError("bad json")

    monkeypatch.setattr(R, "_call_gemini", fake_call)
    out = run(R.rerank(cands("A", "B", "C", "D"), CTX))
    assert out["mode"] == "offline" and calls == ["m1"]  # no second model burned
    assert R._rest_until == {}


def test_rerank_times_out_and_falls_back(monkeypatch):
    _fake_online(monkeypatch)
    monkeypatch.setattr(R, "TIMEOUT_S", 0.3)

    def slow_call(prompt, model):
        time.sleep(1.0)
        return {"choices": ["A"]}

    monkeypatch.setattr(R, "_call_gemini", slow_call)
    async def timed():
        t = time.monotonic()
        out = await R.rerank(cands("A", "B", "C", "D"), CTX)
        return out, time.monotonic() - t

    # Timed inside the event loop: asyncio.run() itself then waits for the abandoned worker thread on
    # shutdown, which a long-running server never does.
    out, took = run(timed())
    assert out["mode"] == "offline" and "timed out" in out["reason"]
    assert out["choices"] == ["A", "B", "C"]
    assert took < 0.8


def test_init_retries_after_a_failure_window(monkeypatch):
    monkeypatch.setattr(R, "_model_error", "boom")
    monkeypatch.setattr(R, "_failed_at", time.time())
    R.init()  # inside the 20 s window: nothing changes
    assert R._model_error == "boom"
    monkeypatch.setattr(R, "_failed_at", time.time() - 100)
    R.init()  # window passed: tries again, and with no key records that
    assert "GEMINI_API_KEY" in R._model_error
