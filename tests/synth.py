"""Synthetic lip recordings for the tests (no real calibration data is ever read).

A frame is a flat list of 108 floats laid out like the browser sends them:
[28 mouth blendshapes | 80 lip coordinates (x, y pairs)], dims = {"bs": 28, "lip": 80}.

Each phrase index is a distinct smooth lip trajectory: a sinusoid with a phrase-specific
frequency, phase and amplitude (0.05-0.1) spread over the lip coordinates with a phrase-specific
spatial pattern, plus a little gaussian noise (0.003) so repeated samples of one phrase are similar
but never identical. Every generator is seeded, so the suite is deterministic.
"""
from __future__ import annotations

import numpy as np

N_BS = 28
N_LIP = 80
D = N_BS + N_LIP
DIMS = {"bs": N_BS, "lip": N_LIP}
FPS = 30.0
NOISE = 0.003
N_PATTERNS = 8  # phrase indices 0..7 are guaranteed distinct

_CYCLES = [1.0, 2.0, 0.5, 3.0, 1.5, 2.5, 0.75, 3.5]


def _params(p: int) -> tuple[float, float, float, np.ndarray]:
    """(cycles, phase, amplitude, spatial pattern over the 80 lip coordinates) for phrase index p."""
    p = p % N_PATTERNS
    cycles = _CYCLES[p]
    phase = (p * 0.9) % (2 * np.pi)
    amp = 0.05 + 0.05 * ((p * 0.37) % 1.0)  # in [0.05, 0.1)
    j = np.arange(N_LIP)
    spatial = np.cos(j * (0.15 + 0.11 * p) + 0.6 * p)  # in [-1, 1], different per phrase
    return cycles, phase, amp, spatial


def make_frames(phrase_index: int, seed: int = 0, n_frames: int | None = None,
                noise: float = NOISE) -> np.ndarray:
    """(T, 108) array; T is about 60 (54-66, seeded) unless n_frames is given."""
    rng = np.random.default_rng(100_003 * seed + phrase_index)
    T = int(n_frames) if n_frames else int(60 + rng.integers(-6, 7))
    cycles, phase, amp, spatial = _params(phrase_index)
    t = np.linspace(0.0, 1.0, T)
    wave = np.sin(2 * np.pi * cycles * t + phase)[:, None] * amp * spatial[None, :]
    lips = wave + rng.normal(0.0, noise, size=(T, N_LIP))
    bs = 0.1 + rng.normal(0.0, noise, size=(T, N_BS))
    return np.concatenate([bs, lips], axis=1)


def make_recording(phrase_index: int, seed: int = 0, **kw) -> dict:
    """{"frames": list of 108-float lists, "dims": DIMS, "fps": FPS} ready for the JSON API."""
    return {"frames": make_frames(phrase_index, seed, **kw).tolist(), "dims": dict(DIMS), "fps": FPS}


def make_sample(user: str, phrase: str, phrase_index: int, seed: int = 0, sid: str | None = None,
                **kw) -> dict:
    """A stored enrollment record, the shape server.app writes to data/enrollments/*.json."""
    rec = make_recording(phrase_index, seed, **kw)
    return {"id": sid or f"{user}__{phrase_index}__{seed}", "user": user, "phrase": phrase, **rec,
            "created": 0.0}


def still_frames(n: int = 30, value: float = 0.2) -> list[list[float]]:
    """n identical frames: the lips did not move at all."""
    return [[value] * D for _ in range(n)]
