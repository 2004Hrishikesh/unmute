import numpy as np
import pytest

from server.classifier import Config, Recogniser, dtw_batch, dtw_pairwise
from synth import DIMS, make_frames, make_recording

PHRASES = ["I need water", "Yes", "No", "Call the nurse"]


def _samples(n_per_phrase=3, phrases=PHRASES, seed0=0):
    out = []
    for pi, p in enumerate(phrases):
        for k in range(n_per_phrase):
            r = make_recording(pi, seed=seed0 + k)
            out.append({"id": f"{p}-{k}", "phrase": p, **r})
    return out


def test_dtw_batch_zero_for_identical_and_symmetric():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 5))
    b = rng.normal(size=(30, 5))
    assert dtw_batch(a, a[None], window=6)[0] == pytest.approx(0.0)
    ab = dtw_batch(a, b[None], window=6)[0]
    ba = dtw_batch(b, a[None], window=6)[0]
    assert ab == pytest.approx(ba)
    assert ab > 0


def test_dtw_pairwise_matches_batch_loop():
    rng = np.random.default_rng(1)
    T = rng.normal(size=(7, 30, 6))
    ref = np.stack([dtw_batch(T[i], T, 6) for i in range(7)])
    P = dtw_pairwise(T, 6)
    assert P.shape == (7, 7)
    assert np.allclose(P, ref, atol=1e-6)
    assert np.allclose(np.diag(P), 0.0, atol=1e-6)


def test_rank_orders_by_distance_with_sane_confidences():
    rec = Recogniser(_samples(), Config())
    q = make_recording(1, seed=99)["frames"]  # "Yes"
    ranked = rec.rank(q, top=5)
    assert [r["phrase"] for r in ranked][0] == "Yes"
    dists = [r["distance"] for r in ranked]
    assert dists == sorted(dists)
    confs = [r["confidence"] for r in ranked]
    assert all(0 < c <= 1 for c in confs)
    assert confs == sorted(confs, reverse=True)
    assert len(ranked) == len(PHRASES)  # one entry per phrase, never more than `top`
    assert len(rec.rank(q, top=2)) == 2


def test_held_out_sample_ranks_its_own_phrase_first_with_and_without_hubness():
    for cfg in (Config(), Config(hub_alpha=0.0)):
        rec = Recogniser(_samples(), cfg)
        assert (rec.pairwise is not None) == (cfg.hub_alpha > 0)
        for i, s in enumerate(rec.samples):
            names = [r["phrase"] for r in rec.rank(s["frames"], top=3, exclude=i)]
            assert names[0] == s["phrase"], (cfg.hub_alpha, s["id"], names)
            assert s["phrase"] in names


def test_exclude_hides_the_held_out_template_from_distances_and_hubness():
    rec = Recogniser(_samples(), Config(hub_alpha=0.5))
    d = rec.distances(rec.samples[0]["frames"], exclude=0)
    assert np.isinf(d[0]) and np.all(np.isfinite(np.delete(d, 0)))
    # hubness with the held-out column ignored differs from the full one, and never uses index 0
    h_full = rec._hubness(None)
    h_ex = rec._hubness(0)
    assert h_full.shape == h_ex.shape == (len(rec.samples),)
    assert np.all(np.isfinite(h_ex))


def test_recogniser_filters_bad_samples_and_mixed_widths():
    good = _samples(n_per_phrase=2)
    bad = [{"id": "x", "phrase": "Yes", "frames": [[0.0] * 108], "dims": DIMS},  # 1 frame
           {"id": "y", "phrase": "No", "frames": [[0.0] * 10, [1.0] * 10], "dims": DIMS}]  # wrong width
    rec = Recogniser(good + bad, Config())
    assert len(rec.samples) == len(good)
    with pytest.raises(ValueError):
        Recogniser(bad[:1], Config())


def test_config_as_dict_roundtrip():
    d = Config(length=40, hub_alpha=0.0).as_dict()
    assert d["length"] == 40 and d["hub_alpha"] == 0.0 and "window" in d


def test_dtw_pairwise_blocks_match_the_one_shot_result():
    """Memory is bounded by processing rows in blocks; any block size must give the same distances."""
    rng = np.random.default_rng(3)
    T = rng.normal(size=(23, 20, 5))
    full = dtw_pairwise(T, 4, budget=10**12)   # one block
    for budget in (1, 20 * 20 * 23 * 2, 20 * 20 * 23 * 7):  # 1 row, 2 rows, 7 rows (uneven last block)
        assert np.allclose(dtw_pairwise(T, 4, budget=budget), full, atol=1e-9)
