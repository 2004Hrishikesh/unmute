"""Feature helpers shared by the recogniser and the evaluation script.

A recorded sample is a list of frames. Each frame is a flat list of floats laid
out as  [mouth blendshapes (dims["bs"] values) | lip landmark x,y (dims["lip"] values)].
The browser already centres the lip points on the mouth, removes head roll and
divides by the eye-corner distance, so the server only has to resample and
z-normalise.
"""
from __future__ import annotations

import numpy as np

DEFAULT_LEN = 40


def to_array(frames) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError("frames must be a 2D list with at least 2 frames")
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr)
    return arr


def resample(seq: np.ndarray, length: int = DEFAULT_LEN) -> np.ndarray:
    """Linearly resample a (T, D) sequence to (length, D)."""
    t_old = np.linspace(0.0, 1.0, seq.shape[0])
    t_new = np.linspace(0.0, 1.0, length)
    out = np.empty((length, seq.shape[1]))
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(t_new, t_old, seq[:, d])
    return out


def fit_stats(seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and std over every frame of one user's enrollments."""
    allf = np.concatenate(seqs, axis=0)
    mean = allf.mean(axis=0)
    std = allf.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def group_weights(dims: dict, D: int, bs_weight: float = 1.0, lip_weight: float = 1.0) -> np.ndarray:
    """Give the blendshape group and the lip-point group equal total influence,
    so 80 lip coordinates do not drown out 28 blendshapes."""
    n_bs = int(dims.get("bs", 0))
    n_lip = D - n_bs
    w = np.ones(D)
    if n_bs > 0:
        w[:n_bs] = bs_weight / np.sqrt(n_bs)
    if n_lip > 0:
        w[n_bs:] = lip_weight / np.sqrt(n_lip)
    return w


def add_deltas(seq: np.ndarray) -> np.ndarray:
    d = np.gradient(seq, axis=0)
    return np.concatenate([seq, d], axis=1)


def lip_motion(arr: np.ndarray, n_bs: int) -> float:
    """Average range of the lip coordinates over a recording (eye-distance units)."""
    lips = arr[:, n_bs:]
    if lips.shape[1] == 0:
        lips = arr
    return float(np.ptp(lips, axis=0).mean())


def trim_still(arr: np.ndarray, n_bs: int, thr: float = 0.15, pad: int = 3) -> np.ndarray:
    """Cut the still frames before the lips start and after they stop moving
    (people press Space a little early and let go a little late)."""
    lips = arr[:, n_bs:] if arr.shape[1] > n_bs else arr
    if len(arr) < 10:
        return arr
    v = np.linalg.norm(np.diff(lips, axis=0), axis=1)
    v = np.convolve(v, np.ones(5) / 5, mode="same")
    if v.max() <= 0:
        return arr
    idx = np.where(v > thr * v.max())[0]
    if len(idx) < 5:
        return arr
    lo, hi = max(0, idx[0] - pad), min(len(arr), idx[-1] + pad + 2)
    return arr[lo:hi] if hi - lo >= 8 else arr
