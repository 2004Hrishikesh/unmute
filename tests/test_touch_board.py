"""The touch board (assistive touch): picking a sentence the cards did not offer, fixing a wrong confirmation,
and the live accuracy that comes out of the request log."""
import json
import time

import server.app as appmod
from synth import make_recording
from test_api import calibrate


def log(client, **body):
    r = client.post("/api/log", json={"user": "bob", "method": "nod", "mode": "offline", **body})
    assert r.status_code == 200, r.text
    time.sleep(0.002)
    return r.json()


def test_touch_pick_learns_the_recording_under_the_picked_sentence(client):
    calibrate(client, "ann", n=2)
    q = make_recording(1, seed=9)
    row = log(client, sentence="I need water", method="touch", rank=-1, learn=True, **q)
    assert row["learned"] is True and row["learned_id"].startswith("bob__i-need-water__")
    stored = json.loads((appmod.ENROLL_DIR / f"{row['learned_id']}.json").read_text())
    assert stored["phrase"] == "I need water" and stored["source"] == "use" and stored["method"] == "touch"


def test_relabel_moves_a_wrongly_learned_sample(client):
    calibrate(client, "ann", n=2)
    q = make_recording(1, seed=9)
    wrong = log(client, sentence="Yes", rank=0, learn=True, **q)
    sid = wrong["learned_id"]
    r = client.post("/api/relabel", json={"user": "bob", "id": sid, "phrase": "No"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["from"] == "Yes" and out["to"] == "No" and out["id"].startswith("bob__no__")
    assert not (appmod.ENROLL_DIR / f"{sid}.json").exists()
    moved = json.loads((appmod.ENROLL_DIR / f"{out['id']}.json").read_text())
    assert moved["phrase"] == "No" and moved["relabelled_from"] == "Yes" and moved["frames"] == q["frames"]
    # the pool picks the change up
    info = client.get("/api/enrollments?user=bob").json()
    assert info["counts"] == {"No": 1}


def test_relabel_refuses_calibration_samples_other_users_and_bad_input(client):
    calibrate(client, "ann", n=2)
    cal = next(appmod.ENROLL_DIR.glob("ann__yes__*.json")).stem
    assert client.post("/api/relabel", json={"user": "ann", "id": cal, "phrase": "No"}).status_code == 409
    q = make_recording(1, seed=9)
    sid = log(client, sentence="Yes", learn=True, **q)["learned_id"]
    assert client.post("/api/relabel", json={"user": "ann", "id": sid, "phrase": "No"}).status_code == 400  # not ann's
    assert client.post("/api/relabel", json={"user": "bob", "id": sid, "phrase": "Not a phrase"}).status_code == 400
    assert client.post("/api/relabel", json={"user": "bob", "id": "../../etc/passwd", "phrase": "No"}).status_code == 400
    assert client.post("/api/relabel", json={"user": "bob", "id": "bob__no__123", "phrase": "No"}).status_code == 404


def test_usage_counts_cards_touch_picks_and_corrections(client):
    assert client.get("/api/usage").json() == {"n": 0}
    log(client, sentence="Yes", rank=0)
    log(client, sentence="No", rank=2)
    log(client, sentence="Thank you", rank=-1, method="touch")  # not on the cards
    wrong = log(client, sentence="Yes", rank=0)                # confirmed the wrong card...
    fix = log(client, sentence="No", rank=-1, method="touch-fix", corrects=wrong["id"])  # ...then fixed it
    assert fix["corrects"] == wrong["id"]
    log(client, sentence="Hello")                              # no rank (old client / plain touch say): ignored
    log(client, sentence="I am in pain", rank=0, confirmed=False, learn=False)  # possible alert: ignored
    u = client.get("/api/usage").json()
    # counted: Yes@0, No@2, Thank you@-1, fix No@-1 (the corrected Yes@0 no longer counts as a success)
    assert u["n"] == 4 and u["top1"] == 0.25 and u["top3"] == 0.5 and u["corrections"] == 2
    rows = {r["id"]: r for r in client.get("/api/requests").json()["requests"]}
    assert rows[wrong["id"]]["corrected_to"] == "No"
    assert client.get("/api/usage?user=nobody").json() == {"n": 0}


def test_big_pool_rebuilds_in_the_background_and_keeps_answering(client, monkeypatch):
    monkeypatch.setattr(appmod, "SYNC_REBUILD_MAX", 3)
    calibrate(client, "ann", n=2)  # 6 recordings > 3
    first = appmod.get_recogniser()
    assert first is not None
    q = make_recording(1, seed=9)
    log(client, sentence="No", learn=True, **q)
    # stale model returned at once, while a rebuild runs
    assert appmod.get_recogniser() is first
    for _ in range(200):
        if not appmod._rebuilding and appmod._rec_cache["*"][1] is not first:
            break
        time.sleep(0.01)
    new = appmod.get_recogniser()
    assert new is not first and len(new.samples) == 7
    r = client.post("/api/predict", json={"user": "bob", "use_gemini": False, **make_recording(1, seed=12)})
    assert r.status_code == 200 and r.json()["choices"][0] == "No"
