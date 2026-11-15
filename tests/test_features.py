import numpy as np
import pytest

from server import features as F
from synth import N_BS, make_frames


def test_resample_length_and_endpoints():
    seq = np.linspace(0, 1, 50)[:, None] * np.array([[1.0, 2.0, -3.0]])
    out = F.resample(seq, 40)
    assert out.shape == (40, 3)
    assert np.allclose(out[0], seq[0])
    assert np.allclose(out[-1], seq[-1])
    # linear data stays linear
    assert np.allclose(out[:, 0], np.linspace(0, 1, 40))


def test_to_array_rejects_bad_shapes_and_cleans_nans():
    with pytest.raises(ValueError):
        F.to_array([[1.0, 2.0]])  # one frame only
    with pytest.raises(ValueError):
        F.to_array([1.0, 2.0, 3.0])  # not 2D
    arr = F.to_array([[1.0, float("nan")], [2.0, 3.0]])
    assert np.all(np.isfinite(arr))


def test_fit_stats_std_floor():
    seqs = [np.array([[1.0, 5.0], [1.0, 7.0]]), np.array([[1.0, 9.0], [1.0, 11.0]])]
    mean, std = F.fit_stats(seqs)
    assert mean[0] == 1.0 and std[0] == 1.0  # constant column: std floored to 1, no divide-by-zero
    assert std[1] > 1.0


def test_group_weights_give_each_group_equal_total_influence():
    w = F.group_weights({"bs": N_BS}, 108, bs_weight=2.0, lip_weight=1.0)
    assert w.shape == (108,)
    assert np.isclose((w[:N_BS] ** 2).sum(), 4.0)
    assert np.isclose((w[N_BS:] ** 2).sum(), 1.0)
    w0 = F.group_weights({"bs": 0}, 10)
    assert np.allclose(w0, 1.0 / np.sqrt(10))


def test_trim_still_cuts_leading_and_trailing_still_frames():
    moving = make_frames(0, seed=1, n_frames=40, noise=0.0)
    still = np.repeat(moving[:1], 30, axis=0)
    arr = np.concatenate([still, moving, still], axis=0)
    out = F.trim_still(arr, N_BS, pad=3)
    # moving part + padding + the 5-frame smoothing window bleeding up to 2 frames past each edge
    assert 40 <= len(out) <= 40 + 2 * 3 + 2 + 2 * 2
    # what is kept is the moving part (plus padding), not a still edge
    assert np.ptp(out[:, N_BS:], axis=0).mean() > 0.01


def test_trim_still_leaves_short_or_still_recordings_alone():
    short = make_frames(0, seed=2, n_frames=8)
    assert F.trim_still(short, N_BS) is short
    still = np.full((40, 108), 0.2)
    assert F.trim_still(still, N_BS) is still


def test_lip_motion_zero_when_still_and_positive_when_moving():
    assert F.lip_motion(np.full((40, 108), 0.2), N_BS) == 0.0
    assert F.lip_motion(make_frames(1, seed=3), N_BS) > 0.012


def test_add_deltas_doubles_width():
    arr = make_frames(2, seed=4, n_frames=20)
    out = F.add_deltas(arr)
    assert out.shape == (20, 216)
    assert np.allclose(out[:, :108], arr)
