"""A lip-reading model trained on MANY people's recordings, for patients who can't record anything themselves.

The DTW matcher (classifier.py) compares a new recording with stored examples, which works best when the
examples are the same person's. This model instead learns what each phrase looks like across mouths, so
it can serve a brand-new patient from the first minute. It is trained on everyone's calibration recordings
and is only switched on when leave-one-person-out testing (train on everyone else, test on the person left
out = exactly a brand-new patient) says it beats the matcher. See scripts/train.py.

Pure numpy: no GPU, no PyTorch, nothing new to install, runs offline in a few milliseconds per prediction.

How it generalises across people from little data:
- Speaker normalisation: each recording is trimmed to the moving part, centred on its own mean lip shape
  (removes the person's resting mouth shape) and scaled by its own movement size (big and small mouthers
  look alike). Only the 40 lip points are used (the blendshapes were noisier in our data).
- Augmentation: every epoch sees a different random variant of every recording: time-warped, cropped,
  stretched in x and y, slightly rotated, mirrored left-right, with noise. That is what stands in for
  thousands of training videos.
- An ensemble of small MLPs with dropout and weight decay, averaged.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import features as F

# web/common.js: LIPS = OUTER_LIPS.concat(INNER_LIPS), each point as x, y (centred, de-rolled, / eye distance).
OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LIPS = OUTER_LIPS + INNER_LIPS
# Left-right symmetric pairs of MediaPipe face-mesh lip landmarks (points on the midline map to themselves).
_PAIRS = [(61, 291), (146, 375), (91, 321), (181, 405), (84, 314), (37, 267), (39, 269), (40, 270), (185, 409),
          (78, 308), (95, 324), (88, 318), (178, 402), (87, 317), (82, 312), (81, 311), (80, 310), (191, 415)]
_MIRROR = {a: b for a, b in _PAIRS} | {b: a for a, b in _PAIRS}
MIRROR_IDX = np.array([LIPS.index(_MIRROR.get(i, i)) for i in LIPS])


@dataclass
class NetConfig:
    length: int = 24          # frames after resampling
    hidden: int = 256
    dropout: float = 0.3
    weight_decay: float = 3e-4
    lr: float = 2e-3
    epochs: int = 120
    batch: int = 32
    ensemble: int = 5
    seed: int = 0
    augment: bool = True
    mirror: bool = True
    scale_norm: bool = True   # divide each recording by its own movement size

    def as_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- features

def lips_of(sample: dict) -> np.ndarray:
    """(T, 40, 2) lip points of a recording, trimmed to where the lips move."""
    arr = F.to_array(sample["frames"])
    n_bs = int((sample.get("dims") or {}).get("bs", 0))
    arr = F.trim_still(arr, n_bs)
    lips = arr[:, n_bs:]
    if lips.shape[1] != 80:
        raise ValueError(f"expected 80 lip values per frame, got {lips.shape[1]}")
    return lips.reshape(len(lips), 40, 2)


def _resample_idx(T: int, length: int, t: np.ndarray | None = None):
    pos = (np.linspace(0.0, 1.0, length) if t is None else t) * (T - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, T - 1)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (pos - i0)[:, None, None]
    return i0, i1, w


def featurise(lips: np.ndarray, cfg: NetConfig, t: np.ndarray | None = None) -> np.ndarray:
    """(T, 40, 2) -> flat vector: resampled, centred, scale-normalised positions + velocities."""
    x = lips - lips.mean(axis=0, keepdims=True)
    if cfg.scale_norm:
        x = x / (np.sqrt((x ** 2).mean()) + 1e-6)
    i0, i1, w = _resample_idx(len(x), cfg.length, t)
    r = x[i0] * (1 - w) + x[i1] * w                      # (L, 40, 2)
    v = np.gradient(r, axis=0) * 3.0
    return np.concatenate([r.reshape(cfg.length, -1), v.reshape(cfg.length, -1)], axis=1).ravel()


def augment(lips: np.ndarray, rng: np.random.Generator, cfg: NetConfig) -> np.ndarray:
    """One random variant of a recording, plus the time grid to resample it on."""
    T = len(lips)
    x = lips
    if cfg.mirror and rng.random() < 0.5:
        x = x[:, MIRROR_IDX] * np.array([-1.0, 1.0])
    a = np.deg2rad(rng.uniform(-8, 8))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    x = (x @ R.T) * rng.uniform(0.8, 1.25, size=2)
    x = x + rng.normal(0, 0.03 * (x.std() + 1e-6), size=x.shape)
    # crop up to 10% at each end, then a smooth monotonic time warp
    lo, hi = rng.uniform(0, 0.1), 1 - rng.uniform(0, 0.1)
    u = np.linspace(0, 1, cfg.length)
    warp = u + rng.uniform(-0.12, 0.12) * np.sin(np.pi * u)
    warp = np.maximum.accumulate(np.clip(warp, 0, 1))
    t = lo + (hi - lo) * warp
    return featurise(x, cfg, t) if T >= 2 else featurise(x, cfg)


# ---------------------------------------------------------------- the network (numpy MLP, Adam)

class _MLP:
    def __init__(self, d_in, d_hid, d_out, rng):
        self.p = {
            "W1": rng.normal(0, np.sqrt(2 / d_in), (d_in, d_hid)), "b1": np.zeros(d_hid),
            "W2": rng.normal(0, np.sqrt(1 / d_hid), (d_hid, d_out)), "b2": np.zeros(d_out),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def logits(self, X, drop=0.0, rng=None):
        h = np.maximum(X @ self.p["W1"] + self.p["b1"], 0)
        mask = None
        if drop > 0:
            mask = (rng.random(h.shape) >= drop) / (1 - drop)
            h = h * mask
        return h @ self.p["W2"] + self.p["b2"], h, mask

    def step(self, X, y, cfg: NetConfig, rng):
        z, h, mask = self.logits(X, cfg.dropout, rng)
        z = z - z.max(1, keepdims=True)
        P = np.exp(z); P /= P.sum(1, keepdims=True)
        n = len(X)
        dz = P.copy(); dz[np.arange(n), y] -= 1; dz /= n
        g = {"W2": h.T @ dz, "b2": dz.sum(0)}
        dh = (dz @ self.p["W2"].T) * (h > 0)
        if mask is not None:
            dh = dh * mask
        g["W1"] = X.T @ dh
        g["b1"] = dh.sum(0)
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in self.p:
            if k.startswith("W"):
                g[k] = g[k] + cfg.weight_decay * self.p[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= cfg.lr * mh / (np.sqrt(vh) + eps)
        return float(-np.log(P[np.arange(n), y] + 1e-12).mean())


@dataclass
class LipNet:
    phrases: list
    cfg: NetConfig
    mu: np.ndarray
    sd: np.ndarray
    nets: list = field(default_factory=list)  # list of param dicts
    meta: dict = field(default_factory=dict)

    # -------------------------------------------------------- training
    @classmethod
    def train(cls, samples: list[dict], cfg: NetConfig | None = None, log=None) -> "LipNet":
        cfg = cfg or NetConfig()
        data = []
        for s in samples:
            try:
                data.append((lips_of(s), s["phrase"]))
            except ValueError:
                continue
        phrases = sorted({p for _, p in data})
        if len(phrases) < 2:
            raise ValueError("need recordings of at least 2 phrases")
        idx = {p: i for i, p in enumerate(phrases)}
        y = np.array([idx[p] for _, p in data])
        X0 = np.stack([featurise(l, cfg) for l, _ in data])
        mu, sd = X0.mean(0), X0.std(0) + 1e-6
        # Balance phrases: each epoch draws every phrase equally often.
        by_phrase = [np.where(y == k)[0] for k in range(len(phrases))]
        per_phrase = max(len(b) for b in by_phrase)
        nets = []
        t0 = time.time()
        for e in range(cfg.ensemble):
            rng = np.random.default_rng(cfg.seed * 1000 + e)
            net = _MLP(X0.shape[1], cfg.hidden, len(phrases), rng)
            for ep in range(cfg.epochs):
                order = np.concatenate([rng.choice(b, per_phrase, replace=len(b) < per_phrase) for b in by_phrase])
                rng.shuffle(order)
                Xa = np.stack([augment(data[i][0], rng, cfg) for i in order]) if cfg.augment else X0[order]
                Xa = (Xa - mu) / sd
                loss = 0.0
                for b0 in range(0, len(order), cfg.batch):
                    sl = slice(b0, b0 + cfg.batch)
                    loss = net.step(Xa[sl], y[order][sl], cfg, rng)
                if log and (ep + 1) % 40 == 0:
                    log(f"  net {e + 1}/{cfg.ensemble} epoch {ep + 1}/{cfg.epochs} loss {loss:.3f}")
            nets.append(net.p)
        meta = {"samples": len(data), "phrases": len(phrases), "train_s": round(time.time() - t0, 1),
                "users": sorted({s.get("user", "?") for s in samples}), "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        return cls(phrases, cfg, mu, sd, nets, meta)

    # -------------------------------------------------------- inference
    def proba(self, frames, dims: dict | None = None) -> dict[str, float]:
        lips = lips_of({"frames": frames, "dims": dims or {"bs": 28}})
        x = (featurise(lips, self.cfg) - self.mu) / self.sd
        P = np.zeros(len(self.phrases))
        for p in self.nets:
            z = np.maximum(x @ p["W1"] + p["b1"], 0) @ p["W2"] + p["b2"]
            z = np.exp(z - z.max())
            P += z / z.sum()
        P /= len(self.nets)
        return {ph: float(v) for ph, v in zip(self.phrases, P)}

    # -------------------------------------------------------- save / load (npz + json, no pickle)
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"mu": self.mu, "sd": self.sd}
        for i, p in enumerate(self.nets):
            for k, v in p.items():
                arrays[f"n{i}_{k}"] = v
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        path.with_suffix(".json").write_text(json.dumps(
            {"phrases": self.phrases, "cfg": self.cfg.as_dict(), "n_nets": len(self.nets), "meta": self.meta}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "LipNet":
        info = json.loads(path.with_suffix(".json").read_text())
        z = np.load(path.with_suffix(".npz"), allow_pickle=False)
        nets = [{k: z[f"n{i}_{k}"] for k in ("W1", "b1", "W2", "b2")} for i in range(info["n_nets"])]
        return cls(info["phrases"], NetConfig(**info["cfg"]), z["mu"], z["sd"], nets, info.get("meta", {}))


# ---------------------------------------------------------------- fusing with the DTW matcher

def dtw_proba(ranked_all: list[dict], tau: float) -> dict[str, float]:
    """Turn the matcher's per-phrase distances into probabilities (softmax of -distance / tau)."""
    if not ranked_all:
        return {}
    d = np.array([r["distance"] for r in ranked_all])
    z = np.exp(-(d - d.min()) / tau)
    z /= z.sum()
    return {r["phrase"]: float(v) for r, v in zip(ranked_all, z)}


def fuse(p_net: dict[str, float], p_dtw: dict[str, float], alpha: float) -> list[tuple[str, float]]:
    """Weighted geometric mean: alpha = trust in the trained model, 1 - alpha = trust in the matcher.
    Only phrases the matcher can answer (it has recordings of them) are ranked."""
    out = []
    for ph, pd in p_dtw.items():
        pn = p_net.get(ph, 1e-4)
        out.append((ph, alpha * np.log(pn + 1e-6) + (1 - alpha) * np.log(pd + 1e-6)))
    s = np.array([v for _, v in out])
    w = np.exp(s - s.max()); w /= w.sum()
    return sorted(((ph, float(v)) for (ph, _), v in zip(out, w)), key=lambda kv: -kv[1])
