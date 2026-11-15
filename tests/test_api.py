import json
import time

import server.app as appmod
from synth import DIMS, make_recording, still_frames

A_PHRASES = ["Yes", "No", "I need water"]  # all in the built-in list


def enroll(client, user, phrase, idx, seed):
    r = make_recording(idx, seed=seed)
    resp = client.post("/api/enroll", json={"user": user, "phrase": phrase, **r})
    time.sleep(0.002)  # ids are millisecond timestamps
    return resp


def calibrate(client, user="ann", n=3, seed0=0):
    last = None
    for pi, p in enumerate(A_PHRASES):
        for k in range(n):
            last = enroll(client, user, p, pi, seed0 + k)
            assert last.status_code == 200, last.text
    return last


def test_health_and_phrases(client):
    assert client.get("/health").json() == {"ok": True}
    ph = client.get("/api/phrases").json()
    assert set(ph) >= {"phrases", "urgent", "demo_set", "quick_set"}
    assert set(ph["urgent"]) <= set(ph["phrases"])
    assert set(ph["demo_set"]) <= set(ph["phrases"]) and set(ph["quick_set"]) <= set(ph["phrases"])
    assert "I cannot breathe" in ph["quick_set"] and "Yes" in ph["quick_set"]


def test_status_reports_offline_gemini(client):
    s = client.get("/api/status").json()
    assert s["gemini"]["available"] is False
    assert s["ntfy_topic"] == "unmute-test-topic"


def test_enroll_validation(client):
    good = make_recording(0, seed=1)
    assert client.post("/api/enroll", json={"user": "ann", "phrase": "Not a phrase", **good}).status_code == 400
    ragged = {**good, "frames": good["frames"][:5] + [good["frames"][5][:-1]]}
    assert client.post("/api/enroll", json={"user": "ann", "phrase": "Yes", **ragged}).status_code == 400
    still = {"frames": still_frames(30), "dims": DIMS, "fps": 50}
    r = client.post("/api/enroll", json={"user": "ann", "phrase": "Yes", **still})
    assert r.status_code == 422 and "barely moved" in r.json()["detail"]
    assert client.post("/api/enroll", json={"user": "!!!", "phrase": "Yes", **good}).status_code == 400
    r = client.post("/api/enroll", json={"user": "Ann Marie", "phrase": "Yes", **good})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["id"].startswith("annmarie__yes__") and body["check"] is None
    assert len(list(appmod.ENROLL_DIR.glob("annmarie__*.json"))) == 1


def test_enrollments_info_and_consistency_check(client):
    info = client.get("/api/enrollments?user=ann").json()
    assert info["counts"] == {} and info["own_samples"] == 0 and info["personal"] is False
    assert info["starter_available"] is False and info["starter_samples"] == 0
    last = calibrate(client, "ann", n=3)
    check = last.json()["check"]  # third sample of the third phrase: enough own samples to compare
    assert check is not None and set(check) >= {"matches", "in_top3", "looks_like", "others"}
    assert check["others"] == 2 and check["matches"] is True
    info = client.get("/api/enrollments?user=ann").json()
    assert info["counts"] == {p: 3 for p in A_PHRASES}
    assert info["own_samples"] == 9 and info["personal"] is True and info["own_from_use"] == 0
    # another person sees ann's recordings as a starter model
    other = client.get("/api/enrollments?user=bob").json()
    assert other["starter_available"] is True and other["starter_samples"] == 9 and other["starter_users"] == 1
    assert client.get("/api/users").status_code == 200


def test_predict_personal_starter_and_no_data(client):
    r = client.post("/api/predict", json={"user": "ann", "frames": make_recording(0)["frames"], "use_gemini": False})
    assert r.status_code == 409
    calibrate(client, "ann", n=3)
    q = make_recording(2, seed=77)  # "I need water" pattern
    r = client.post("/api/predict", json={"user": "ann", "frames": q["frames"], "note": "", "local_time": "10:00", "use_gemini": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "personal" and body["own_samples"] == 9 and body["starter_samples"] == 0
    assert body["mode"] == "offline" and len(body["choices"]) == 3
    assert body["choices"][0] == "I need water" and body["candidates"][0]["phrase"] == "I need water"
    assert set(body["candidates"][0]) == {"phrase", "distance", "confidence"}
    assert body["timing"]["total_s"] >= 0
    # a brand-new person is matched against ann's recordings
    r = client.post("/api/predict", json={"user": "bob", "frames": q["frames"], "use_gemini": False})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "starter" and body["own_samples"] == 0 and body["starter_samples"] == 9
    assert body["choices"][0] == "I need water"
    # wrong feature width is refused, not crashed
    r = client.post("/api/predict", json={"user": "ann", "frames": [[0.1] * 50] * 20, "use_gemini": False})
    assert r.status_code == 409


def test_log_learns_from_use_with_a_cap(client):
    calibrate(client, "ann", n=2)
    q = make_recording(1, seed=5)  # "No"
    r = client.post("/api/log", json={"user": "bob", "sentence": "No", "method": "nod", "mode": "offline",
                                      "learn": True, "frames": q["frames"], "dims": q["dims"], "fps": q["fps"]})
    assert r.status_code == 200
    row = r.json()
    assert row["learned"] is True and row["urgent"] is False and row["acknowledged"] is True
    files = list(appmod.ENROLL_DIR.glob("bob__no__*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["source"] == "use"
    info = client.get("/api/enrollments?user=bob").json()
    assert info["own_samples"] == 1 and info["own_from_use"] == 1 and info["personal"] is False
    for _ in range(10):
        client.post("/api/log", json={"user": "bob", "sentence": "No", "method": "nod", "learn": True,
                                      "frames": q["frames"], "dims": q["dims"], "fps": q["fps"]})
        time.sleep(0.002)
    assert len(list(appmod.ENROLL_DIR.glob("bob__no__*.json"))) == appmod.MAX_USE_SAMPLES
    # not learned: unknown sentence, still lips, learn off, no frames
    bad = [
        {"sentence": "Not a phrase", "frames": q["frames"], "dims": q["dims"]},
        {"sentence": "Yes", "frames": still_frames(30), "dims": DIMS},
        {"sentence": "Yes", "frames": q["frames"], "dims": q["dims"], "learn": False},
        {"sentence": "Yes"},
    ]
    for b in bad:
        r = client.post("/api/log", json={"user": "bob", "method": "nod", **b})
        assert r.status_code == 200 and r.json()["learned"] is False, b.get("sentence")
    assert not list(appmod.ENROLL_DIR.glob("bob__yes__*.json"))


def test_urgent_log_requests_and_ack(client):
    r = client.post("/api/log", json={"user": "ann", "sentence": "I cannot breathe", "method": "blink", "mode": "offline"})
    row = r.json()
    assert row["urgent"] is True and row["acknowledged"] is False and row["push"] == "test"
    assert row["learned"] is False
    rid = row["id"]
    reqs = client.get("/api/requests").json()
    assert reqs["unacknowledged_urgent"] == 1
    assert reqs["requests"][0]["id"] == rid and "ago" in reqs["requests"][0]
    assert client.get(f"/api/requests/{rid}").json()["acknowledged"] is False
    assert client.get("/api/requests/nope").status_code == 404
    assert client.post("/api/ack", json={"id": rid}).json() == {"ok": True}
    assert client.get("/api/requests").json()["unacknowledged_urgent"] == 0
    assert client.get(f"/api/requests/{rid}").json()["acknowledged"] is True
    # "all" acknowledges everything open
    client.post("/api/log", json={"user": "ann", "sentence": "Help me", "method": "nod"})
    client.post("/api/log", json={"user": "ann", "sentence": "I am in pain", "method": "nod"})
    assert client.get("/api/requests").json()["unacknowledged_urgent"] == 2
    client.post("/api/ack", json={"id": "all"})
    assert client.get("/api/requests").json()["unacknowledged_urgent"] == 0


def test_history_reaches_predict_context(client, monkeypatch):
    calibrate(client, "ann", n=2)
    client.post("/api/log", json={"user": "ann", "sentence": "Yes", "method": "nod"})
    seen = {}

    async def fake_rerank(candidates, context, use_gemini=True):
        seen.update(context)
        return {"choices": [c["phrase"] for c in candidates[:3]], "mode": "offline", "reason": "stub"}

    monkeypatch.setattr(appmod.rerank, "rerank", fake_rerank)
    r = client.post("/api/predict", json={"user": "ann", "frames": make_recording(0)["frames"], "note": "post-op",
                                          "local_time": "Friday 03:10", "use_gemini": True})
    assert r.status_code == 200 and r.json()["reason"] == "stub"
    assert seen["note"] == "post-op" and seen["time"] == "Friday 03:10"
    assert seen["history"][0]["sentence"] == "Yes" and seen["history"][0]["ago"] == "just now"


def test_delete_last_enrollment(client):
    assert client.delete("/api/enrollments/last?user=ann").status_code == 404
    enroll(client, "ann", "Yes", 1, 0)
    enroll(client, "ann", "No", 2, 0)
    r = client.delete("/api/enrollments/last?user=ann")
    assert r.status_code == 200 and r.json()["deleted"] == "No"
    assert client.get("/api/enrollments?user=ann").json()["counts"] == {"Yes": 1}
    client.delete("/api/enrollments/last?user=ann")
    assert client.delete("/api/enrollments/last?user=ann").status_code == 404


def test_evaluate_and_results(client):
    assert client.get("/api/results").json() == {"results": None}
    calibrate(client, "ann", n=2)
    calibrate(client, "bob", n=2, seed0=10)
    res = client.post("/api/evaluate").json()["results"]
    assert res["overall"]["samples"] == 12 and set(res["users"]) == {"ann", "bob"}
    assert set(res["starter"]["users"]) == {"ann", "bob"} and set(res["pooled"]["users"]) == {"ann", "bob"}
    assert appmod.RESULTS_FILE.exists()
    assert client.get("/api/results").json()["results"]["overall"] == res["overall"]


def test_static_pages_are_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<title>Unmute</title>" in r.text
    assert client.get("/enroll.html").status_code == 200
    assert client.get("/nurse.html").status_code == 200
    assert client.get("/proof.html").status_code == 200
    assert client.get("/common.js").status_code == 200
