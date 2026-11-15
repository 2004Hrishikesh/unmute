"""Urgent sentences that raise the alarm by themselves (no nod), and the patient's "Not what I said" cancel.

The browser decides when to raise it (the lip reader is sure of an urgent sentence); the server's job is to tell
the nurse honestly: an automatic alarm says it is unconfirmed, is never learned from, never counts as a card the
patient picked, and a cancel closes it and tells the nurse it was a misread.
"""
import server.alerts as alerts
import server.app as appmod


def _capture(monkeypatch):
    pushes = []

    async def fake_push(topic, sentence, bed, **kw):
        pushes.append({"sentence": sentence, **kw})
        return True, "ntfy HTTP 200"
    monkeypatch.setattr(alerts, "push_ntfy", fake_push)
    return pushes


def _log(client, **body):
    r = client.post("/api/log", json={"user": "bob", "mode": "offline", **body})
    assert r.status_code == 200, r.text
    return r.json()


def test_automatic_alarm_tells_the_nurse_it_is_unconfirmed_and_is_not_learned(client, monkeypatch, recording):
    pushes = _capture(monkeypatch)
    q = recording(1, seed=3)
    row = _log(client, sentence="I am in pain", method="auto-urgent", rank=None, learn=False, **q)
    assert row["urgent"] and not row["acknowledged"] and row["method"] == "auto-urgent"
    assert row["learned"] is False and row["learned_id"] is None
    assert pushes[-1]["sentence"] == "I am in pain" and "automatically" in pushes[-1]["note"]
    # not a card the patient picked: it does not count in live accuracy
    assert client.get("/api/usage").json() == {"n": 0}


def test_a_confirmed_urgent_sentence_has_no_automatic_note(client, monkeypatch):
    pushes = _capture(monkeypatch)
    _log(client, sentence="Help me", method="nod", rank=0, learn=False)
    assert pushes[-1]["sentence"] == "Help me" and not pushes[-1].get("note")


def test_patient_cancels_an_automatic_alarm_and_the_nurse_is_told(client, monkeypatch):
    pushes = _capture(monkeypatch)
    row = _log(client, sentence="Help me", method="auto-urgent", rank=None, learn=False)
    r = client.post("/api/ack", json={"id": row["id"], "cancelled": True})
    assert r.status_code == 200 and r.json()["push"] == "ntfy HTTP 200"
    back = client.get(f"/api/requests/{row['id']}").json()
    assert back["acknowledged"] is True and back["cancelled"] is True
    last = pushes[-1]
    assert last["sentence"] == "Help me" and "misread" in last["note"]
    assert last["title"] == "Unmute: alarm cancelled" and last["priority"] == "default"


def test_nurse_clearing_an_alarm_is_not_a_cancel(client, monkeypatch):
    pushes = _capture(monkeypatch)
    row = _log(client, sentence="I am in pain", method="auto-urgent", rank=None, learn=False)
    n = len(pushes)
    r = client.post("/api/ack", json={"id": row["id"]})
    assert r.status_code == 200 and "push" not in r.json()
    back = client.get(f"/api/requests/{row['id']}").json()
    assert back["acknowledged"] is True and not back.get("cancelled")
    assert len(pushes) == n  # no extra push


def test_cancel_all_does_not_mark_anything_cancelled(client, monkeypatch):
    _capture(monkeypatch)
    row = _log(client, sentence="I am in pain", method="auto-urgent", rank=None, learn=False)
    client.post("/api/ack", json={"id": "all", "cancelled": True})
    back = client.get(f"/api/requests/{row['id']}").json()
    assert back["acknowledged"] is True and not back.get("cancelled")


def test_push_body_keeps_the_note_outside_the_quotes(monkeypatch):
    import asyncio

    sent = []

    class FakeHTTP:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, content=b"", headers=None):
            sent.append((content.decode(), headers)); return type("R", (), {"status_code": 200})()
    monkeypatch.setattr(alerts.httpx, "AsyncClient", FakeHTTP)
    from test_privacy import REAL_PUSH
    ok, _ = asyncio.run(REAL_PUSH("t", "I am in pain", "Bed 7", note="(raised automatically; the patient can cancel it)"))
    assert ok and sent[-1][0] == 'Bed 7: "I am in pain" (raised automatically; the patient can cancel it)'
    assert sent[-1][1]["Priority"] == "urgent"
    asyncio.run(REAL_PUSH("t", "Help me", "Bed 7", note="was a misread", title="Unmute: alarm cancelled", priority="default"))
    assert sent[-1][0] == 'Bed 7: "Help me" was a misread' and sent[-1][1]["Title"] == "Unmute: alarm cancelled"
