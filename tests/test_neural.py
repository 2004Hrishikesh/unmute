"""The many-people model: features, augmentation, training, save/load, fusion, the fair test, and the API."""
import numpy as np
import pytest

import server.app as appmod
from server.model_eval import best_setting, lopo
from server.neural import LIPS, MIRROR_IDX, LipNet, NetConfig, augment, dtw_proba, featurise, fuse, lips_of
from synth import make_recording

PH = ["I need water", "Yes", "No", "Call the nurse"]
FAST = NetConfig(epochs=25, ensemble=2, hidden=64)


def samples(user, n, seed0, phrases=PH):
    return [{"id": f"{user}-{p}-{k}", "user": user, "phrase": p, **make_recording(i, seed=seed0 + k)}
            for i, p in enumerate(phrases) for k in range(n)]


def test_mirror_map_is_a_left_right_involution():
    assert (MIRROR_IDX[MIRROR_IDX] == np.arange(40)).all()
    assert LIPS[MIRROR_IDX[LIPS.index(61)]] == 291   # mouth corners swap
    assert LIPS[MIRROR_IDX[LIPS.index(0)]] == 0      # top-centre stays


def test_featurise_ignores_resting_shape_and_mouth_size():
    lips = lips_of(samples("a", 1, 0)[0])
    f = featurise(lips, NetConfig())
    assert f.shape == (24 * 160,)
    assert np.allclose(featurise(lips * 2.5 + 0.3, NetConfig()), f, rtol=1e-3, atol=1e-4)  # bigger mouth, shifted: same features
    assert not np.allclose(featurise(lips * 2.5, NetConfig(scale_norm=False)), f)


def test_augment_changes_the_sample_but_keeps_its_size():
    lips = lips_of(samples("a", 1, 0)[0])
    rng = np.random.default_rng(0)
    a, b = augment(lips, rng, NetConfig()), augment(lips, rng, NetConfig())
    assert a.shape == b.shape == featurise(lips, NetConfig()).shape and not np.allclose(a, b)


def test_train_generalises_to_a_new_person_and_survives_save_load(tmp_path):
    train = samples("ann", 4, 0) + samples("bob", 4, 50)
    net = LipNet.train(train, FAST)
    assert net.phrases == sorted(PH) and net.meta["samples"] == 32 and net.meta["users"] == ["ann", "bob"]
    new_person = samples("cat", 2, 900)
    right = sum(max(net.proba(s["frames"], s["dims"]).items(), key=lambda kv: kv[1])[0] == s["phrase"] for s in new_person)
    assert right / len(new_person) >= 0.9
    p = net.proba(new_person[0]["frames"], new_person[0]["dims"])
    assert set(p) == set(PH) and abs(sum(p.values()) - 1) < 1e-6
    net.save(tmp_path / "m" / "lipnet")
    back = LipNet.load(tmp_path / "m" / "lipnet")
    q = back.proba(new_person[0]["frames"], new_person[0]["dims"])
    assert all(abs(q[k] - p[k]) < 1e-9 for k in p)


def test_train_needs_two_phrases():
    with pytest.raises(ValueError):
        LipNet.train(samples("ann", 3, 0, phrases=["Yes"]), FAST)


def test_fuse_weights():
    ranked = [{"phrase": "A", "distance": 0.1}, {"phrase": "B", "distance": 0.2}, {"phrase": "C", "distance": 0.4}]
    pd = dtw_proba(ranked, 0.1)
    assert abs(sum(pd.values()) - 1) < 1e-9 and pd["A"] > pd["B"] > pd["C"]
    pn = {"A": 0.05, "B": 0.05, "C": 0.9, "Z": 0.0}
    assert [p for p, _ in fuse(pn, pd, 0.0)] == ["A", "B", "C"]   # matcher only
    assert [p for p, _ in fuse(pn, pd, 1.0)][0] == "C"            # model only
    assert "Z" not in dict(fuse(pn, pd, 0.5))                     # only phrases the matcher can answer


def test_lopo_reports_new_and_calibrated_patients():
    data = {"ann": samples("ann", 3, 0), "bob": samples("bob", 3, 50)}
    res = lopo(data, FAST)
    assert res["people"] == 2 and set(res["users"]) == {"ann", "bob"}
    for key in ("matcher", "model"):
        assert res["overall"][key]["n"] == 24
    assert res["calibrated"]["matcher"]["n"] == 24
    name, score = best_setting(res)
    assert name in res["overall"] and 0 <= score["top1"] <= score["top3"] <= 1
    assert lopo({"ann": data["ann"]}, FAST)["users"] == {}


def test_predict_uses_the_model_only_for_new_patients_when_deployed(client):
    from test_api import calibrate

    calibrate(client, "ann", n=3)
    q = make_recording(1, seed=33)  # "No"
    body = {"user": "newbie", "use_gemini": False, **q}
    assert client.post("/api/predict", json=body).json()["recogniser"] == "matcher"  # no model yet
    net = LipNet.train([s for ss in appmod.load_samples().values() for s in ss], FAST)
    net.meta.update({"use": True, "alpha": 0.25, "tau": 0.05, "use_for_calibrated": False, "best": "fused a=0.25 tau=0.05"})
    net.save(appmod.MODEL_PATH)
    r = client.post("/api/predict", json=body).json()
    assert r["recogniser"] == "model+matcher" and r["choices"][0] == "No" and len(r["candidates"]) <= 5
    assert client.get("/api/status").json()["model"]["active"] is True
    # ann has her own recordings: the matcher alone, as the fair test decided
    assert client.post("/api/predict", json={**body, "user": "ann"}).json()["recogniser"] == "matcher"
    # a model whose test said "no" is never used
    net.meta["use"] = False
    net.save(appmod.MODEL_PATH)
    import os, time
    os.utime(appmod.MODEL_PATH.with_suffix(".json"), (time.time() + 5, time.time() + 5))
    assert client.post("/api/predict", json=body).json()["recogniser"] == "matcher"
    assert client.get("/api/model").json()["report"]["use"] is False
