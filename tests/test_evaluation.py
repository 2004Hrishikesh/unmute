from server.classifier import Config
from server.evaluation import evaluate_all, evaluate_pooled, evaluate_starter, evaluate_user
from synth import make_recording

PH = ["I need water", "Yes", "No"]


def user_samples(user, n_per_phrase, seed0):
    out = []
    for pi, p in enumerate(PH):
        for k in range(n_per_phrase):
            out.append({"id": f"{user}-{p}-{k}", "user": user, "phrase": p, **make_recording(pi, seed=seed0 + k)})
    return out


def test_evaluate_user_returns_none_with_too_little_data():
    assert evaluate_user(user_samples("a", 1, 0), Config()) is None
    one_phrase = [s for s in user_samples("a", 3, 0) if s["phrase"] == "Yes"]
    assert evaluate_user(one_phrase, Config()) is None


def test_evaluate_all_shape_and_ranges():
    data = {"ann": user_samples("ann", 3, 0), "bob": user_samples("bob", 2, 10), "solo": user_samples("solo", 1, 20)}
    res = evaluate_all(data, Config())
    assert set(res) >= {"generated_at", "method", "config", "with_gemini", "overall", "users", "pooled", "starter"}
    assert res["with_gemini"] is False
    assert set(res["users"]) == {"ann", "bob"}  # solo has one sample per phrase: skipped
    assert res["overall"]["samples"] == 9 + 6
    for u in res["users"].values():
        assert 0 <= u["top1"] <= u["top3"] <= 1
        assert u["phrases"] == 3 and u["chance_top1"] == round(1 / 3, 4) and u["chance_top3"] == 1.0
        assert set(u["per_phrase"]) == set(PH)
        assert isinstance(u["confusions"], list)
        for v in u["per_phrase"].values():
            assert v["n"] >= 2 and 0 <= v["top1"] <= v["top3"] <= 1
    # clean synthetic data: the recogniser should get them all
    assert res["users"]["ann"]["top1"] == 1.0
    assert res["overall"]["top3"] == 1.0


def test_pooled_and_starter_sections():
    data = {"ann": user_samples("ann", 3, 0), "bob": user_samples("bob", 2, 10)}
    pooled = evaluate_pooled(data, Config())
    assert set(pooled["users"]) == {"ann", "bob"}
    assert pooled["users"]["ann"]["samples"] == 9
    starter = evaluate_starter(data, Config())
    assert set(starter["users"]) == {"ann", "bob"}
    s = starter["users"]["bob"]
    assert s["samples"] == 6 and s["phrases"] == 3 and s["chance_top1"] == round(1 / 3, 4)
    assert s["top3"] == 1.0  # other person's synthetic recordings are the same patterns
    # only one user: no starter possible, pooled still works
    assert evaluate_starter({"ann": data["ann"]}, Config())["users"] == {}
    assert "ann" in evaluate_pooled({"ann": data["ann"]}, Config())["users"]
    assert evaluate_pooled({}, Config())["users"] == {}


def test_learning_curve_replays_in_order_and_buckets_by_own_samples():
    from server.evaluation import evaluate_learning

    ann = user_samples("ann", 3, 0)
    bob = user_samples("bob", 4, 10)
    for i, s in enumerate(bob):
        s["created"] = float(i)
    res = evaluate_learning({"ann": ann, "bob": bob}, Config(), max_bucket=2)
    b = res["users"]["bob"]
    assert b["new_phrase"] == 0  # ann recorded every phrase, so even bob's first try is matchable
    buckets = b["by_own_samples"]
    assert set(buckets) == {"0", "1", "2"}
    assert buckets["0"]["n"] == 3 and buckets["1"]["n"] == 3 and buckets["2"]["n"] == 6  # 2 = "2 or more"
    for v in res["overall"].values():
        assert 0 <= v["top1"] <= v["top3"] <= 1
    # one user alone: the very first sample of each phrase has nothing to match
    solo = evaluate_learning({"bob": bob}, Config())
    assert solo["users"]["bob"]["new_phrase"] >= 1
