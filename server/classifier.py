"""DTW nearest-neighbour recogniser over per-frame mouth features.

No training: every enrolled sample is a template. A query is compared with each
template of the same user by dynamic time warping, and each phrase is scored by
the mean of its best `k` distances.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import features as F


@dataclass
class Config:
    # Defaults chosen by leave-one-out on the team's own recordings (scripts/evaluate.py --sweep).
    length: int = 30              # frames after resampling
    window: int = 6               # Sakoe-Chiba band half-width (frames)
    k: int = 1                    # best-k distances per phrase
    deltas: bool = True           # append frame-to-frame velocity
    delta_scale: float = 3.0
    center: bool = True           # subtract each recording's own mean (removes resting-lip drift)
    trim: bool = True             # drop still frames at the start and end
    bs_weight: float = 0.0        # blendshapes were noisier than lip points in our data
    lip_weight: float = 1.0

    def as_dict(self):
        return dict(self.__dict__)


def dtw_batch(q: np.ndarray, T: np.ndarray, window: int) -> np.ndarray:
    """DTW distance between one query (L, D) and N templates (N, L, D).

    Vectorised over templates; loops only over the band cells.
    Returns (N,) distances normalised by path length bound (2L).
    """
    N, L, _ = T.shape
    cost = np.sqrt(((q[None, :, None, :] - T[:, None, :, :]) ** 2).sum(-1))  # (N, L, L)
    acc = np.full((N, L + 1, L + 1), np.inf)
    acc[:, 0, 0] = 0.0
    for i in range(1, L + 1):
        lo, hi = max(1, i - window), min(L, i + window)
        for j in range(lo, hi + 1):
            best = np.minimum(np.minimum(acc[:, i - 1, j], acc[:, i, j - 1]), acc[:, i - 1, j - 1])
            acc[:, i, j] = cost[:, i - 1, j - 1] + best
    return acc[:, L, L] / (2 * L)


@dataclass
class Recogniser:
    samples: list  # list of dicts with phrase, frames, dims
    cfg: Config = field(default_factory=Config)

    def __post_init__(self):
        self.samples = [s for s in self.samples if len(s["frames"]) >= 2]
        if not self.samples:
            raise ValueError("no samples")
        D = len(self.samples[0]["frames"][0])
        self.samples = [s for s in self.samples if len(s["frames"][0]) == D]
        self.dims = self.samples[0].get("dims", {"bs": 0})
        raw = [self._clean(F.to_array(s["frames"])) for s in self.samples]
        self.mean, self.std = F.fit_stats(raw)
        self.weights = F.group_weights(self.dims, D, self.cfg.bs_weight, self.cfg.lip_weight)
        self.phrases = [s["phrase"] for s in self.samples]
        self.templates = np.stack([self._prep(r) for r in raw])

    def _clean(self, arr: np.ndarray) -> np.ndarray:
        n_bs = int(self.dims.get("bs", 0))
        if self.cfg.trim:
            arr = F.trim_still(arr, n_bs)
        if self.cfg.center:
            arr = arr - arr.mean(axis=0)
        return arr

    def _prep(self, arr: np.ndarray) -> np.ndarray:
        x = (arr - self.mean) / self.std * self.weights
        x = F.resample(x, self.cfg.length)
        if self.cfg.deltas:
            x = np.concatenate([x, np.gradient(x, axis=0) * self.cfg.delta_scale], axis=1)
        return x

    def distances(self, frames, exclude: int | None = None) -> np.ndarray:
        q = self._prep(self._clean(F.to_array(frames)))
        d = dtw_batch(q, self.templates, self.cfg.window)
        if exclude is not None:
            d[exclude] = np.inf
        return d

    def rank(self, frames, top: int = 5, exclude: int | None = None) -> list[dict]:
        d = self.distances(frames, exclude)
        by_phrase: dict[str, list[float]] = {}
        for p, dist in zip(self.phrases, d):
            if np.isfinite(dist):
                by_phrase.setdefault(p, []).append(float(dist))
        scores = {p: float(np.mean(sorted(v)[: self.cfg.k])) for p, v in by_phrase.items()}
        ordered = sorted(scores.items(), key=lambda kv: kv[1])
        # Relative confidence: softmax over standardised negative scores.
        vals = np.array([s for _, s in ordered])
        z = (vals - vals.mean()) / (vals.std() + 1e-9) if len(vals) > 1 else np.zeros(1)
        e = np.exp(-z)
        conf = e / e.sum()
        return [
            {"phrase": p, "distance": round(s, 4), "confidence": round(float(c), 4)}
            for (p, s), c in zip(ordered[:top], conf[:top])
        ]
