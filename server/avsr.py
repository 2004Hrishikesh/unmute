"""Open-source visual speech recognition: Auto-AVSR (Ma et al., ICASSP 2023, Imperial College London, Apache 2.0).

A 250M-parameter model trained on 3,291 hours of video of people talking (LRS2, LRS3, VoxCeleb2, AVSpeech);
20.3% word error rate on the LRS3 lip-reading benchmark. It reads the mouth region of the video, so it
works for a brand-new person with no calibration and no training on our side, and runs on this laptop
(Apple GPU via PyTorch MPS, or CPU). Nothing goes to the internet.

We use it two ways:
- score_phrases: how likely each of our phrases is, given the lips (the decoder's probability of exactly that
  sentence, plus CTC, the same 0.9 / 0.1 mix the authors decode with). Much more accurate than free text
  when the answer is one of a known list.
- transcribe: free text (beam search), for anything that is not on the list.

Setup (once, about 1 GB):  bash scripts/fetch_avsr.sh
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "third_party" / "auto_avsr"
WEIGHTS = ROOT / "third_party" / "weights" / "vsr_trlrs2lrs3vox2avsp_base.pth"
FPS = 25.0                 # the model was trained on 25 fps video
CTC_WEIGHT = 0.1           # the authors' decoding setting
# Free text (the lip reader's own reading, for sentences not on the list). Measured on 240 real silent-mouthing
# recordings of list phrases (2 people, 14 Nov 2026; scripts/eval_avsr.py). The reading is usually a near miss of
# the list phrase ("all my family", "called my family", "i'm in bed"), and a beam-searched sentence nearly always
# scores higher than any list phrase, so "more likely than the list" put free text on card 1 wrongly 124/240 times.
# Now free text is only shown when it is NOT a variant of a list phrase, and only first when, in addition, it is
# very confident and no list phrase fits the lips. Measured: wrong card 1 on list phrases 4/240; with the true
# phrase removed from the list (a stand-in for "said something off the list") free text was a right card 1 in 45/240
# and a wrong one in 8/240. Thresholds were picked on those same recordings, so this is optimistic.
# On garbage input (noise, a grey picture) the model "reads" TED-talk filler at -1.6 to -1.9 per word piece.
HEARD_MIN_LOGPROB = -0.8    # shown at all (as an extra card): average log-probability per word piece
FIRST_MIN_LOGPROB = -0.4    # shown FIRST: very confident
LIST_SIMILAR = 0.6          # character similarity to a list phrase at or above which it counts as that phrase
LIST_FIT_PER_WORD = -2.5    # no list phrase fits: best list score per word (+ end) below this


def _norm(t: str) -> str:
    import re

    t = t.lower().replace("i'm", "i am").replace("can't", "cannot").replace("cant", "cannot")
    return re.sub(r"[^a-z ]+", "", t).strip()


def _similar(a: str, b: str) -> float:
    import difflib

    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def place_free_text(text: str, text_conf: float | None, text_total: float | None, phrase_scores: dict) -> dict:
    """Should the free-text reading be shown, and first? Hidden when the model is not sure of it or when it is
    just a (mis)reading of a list phrase: that phrase's card is already there and is cleaner. First only when it
    is very confident AND no list phrase fits the lips well."""
    text = (text or "").strip()
    if not text or text_conf is None or text_conf <= HEARD_MIN_LOGPROB:
        return {"show": False, "first": False}
    if phrase_scores and max(_similar(text, p) for p in phrase_scores) >= LIST_SIMILAR:
        return {"show": False, "first": False}
    best_fit = max((s / (len(_norm(p).split()) + 1) for p, s in phrase_scores.items()), default=-1e9)
    return {"show": True, "first": text_conf > FIRST_MIN_LOGPROB and best_fit < LIST_FIT_PER_WORD}

_lock = threading.Lock()
_state: dict = {"model": None, "error": None, "device": None, "spm": None, "units": None, "token_list": None}


def available() -> bool:
    return WEIGHTS.exists() and (REPO / "espnet").exists()


def status() -> dict:
    return {"installed": available(), "loaded": _state["model"] is not None, "device": _state["device"],
            "error": _state["error"]}


def load():
    """Load once (about 5 s); safe to call from many threads."""
    with _lock:
        if _state["model"] is not None or _state["error"]:
            return _state["model"]
        try:
            import sentencepiece
            import torch

            if str(REPO) not in sys.path:
                sys.path.insert(0, str(REPO))
            from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E

            units = [ln.split()[0] for ln in (REPO / "spm/unigram/unigram5000_units.txt").read_text().splitlines()]
            token_list = ["<blank>"] + units + ["<eos>"]
            model = E2E(len(token_list), "video", ctc_weight=CTC_WEIGHT)
            model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
            model.eval()
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model.to(device)
            _state.update(model=model, device=device, token_list=token_list,
                          spm=sentencepiece.SentencePieceProcessor(model_file=str(REPO / "spm/unigram/unigram5000.model")),
                          units={u: i + 1 for i, u in enumerate(units)})
        except Exception as e:  # missing files, missing packages: the app keeps working without it
            _state["error"] = f"{type(e).__name__}: {e}"[:300]
        return _state["model"]


# ---------------------------------------------------------------- mouth crops (the authors' preprocessing)

def _stable_reference() -> np.ndarray:
    ref = np.load(REPO / "preparation/detectors/mediapipe/20words_mean_face.npy")
    # right eye, left eye, nose, mouth centre (68-point layout), in a 256 x 256 frame
    return np.vstack([ref[36:42].mean(0), ref[42:48].mean(0), ref[31:36].mean(0), ref[48:68].mean(0)])


def mouth_crops(frames: list[np.ndarray], keypoints: np.ndarray) -> np.ndarray:
    """frames: T gray images (H, W) uint8; keypoints: (T, 4, 2) right eye, left eye, nose tip, mouth centre in
    each frame's pixels. Returns (T, 96, 96) uint8 mouth crops, aligned to the mean face like the authors do:
    landmarks smoothed over 12 frames, similarity transform to the reference, 96 x 96 around the mouth."""
    import cv2

    ref = _stable_reference()
    T = len(frames)
    out = []
    for i in range(T):
        m = min(6, i, T - 1 - i)
        sm = keypoints[i - m:i + m + 1].mean(0)
        sm = sm + keypoints[i].mean(0) - sm.mean(0)
        M = cv2.estimateAffinePartial2D(sm.astype(np.float32), ref.astype(np.float32), method=cv2.LMEDS)[0]
        if M is None:
            M = cv2.estimateAffinePartial2D(sm.astype(np.float32), ref.astype(np.float32))[0]
        warped = cv2.warpAffine(frames[i], M, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mouth = (sm[3:4] @ M[:, :2].T + M[:, 2])[0]
        cx, cy = int(round(mouth[0])), int(round(mouth[1]))
        y0, x0 = np.clip(cy - 48, 0, 256 - 96), np.clip(cx - 48, 0, 256 - 96)
        out.append(warped[y0:y0 + 96, x0:x0 + 96])
    return np.stack(out)


def resample_fps(items: list, ts_ms: list[float], fps: float = FPS) -> list[int]:
    """Indices of the frames closest to a steady `fps` clock (the browser runs at 30-60 fps, unevenly)."""
    t = np.asarray(ts_ms, dtype=float)
    t = t - t[0]
    grid = np.arange(0, t[-1] + 1e-6, 1000.0 / fps)
    return [int(np.abs(t - g).argmin()) for g in grid] or [0]


def _tensor(crops: np.ndarray):
    import torch

    x = torch.from_numpy(crops).float() / 255.0           # (T, 96, 96)
    x = x[:, 4:92, 4:92]                                  # centre crop 88
    x = (x - 0.421) / 0.165
    return x.unsqueeze(1)                                 # (T, 1, 88, 88)


def _encode(crops: np.ndarray):
    import torch

    model = load()
    if model is None:
        raise RuntimeError(_state["error"] or "lip reader not installed")
    x = _tensor(crops).to(_state["device"])
    with torch.no_grad():
        f = model.frontend(x.unsqueeze(0))
        f = model.proj_encoder(f)
        enc, _ = model.encoder(f, None)
    return enc                                            # (1, T, 768)


def _ids(text: str) -> list[int]:
    pieces = _state["spm"].EncodeAsPieces(text.upper())
    unk = _state["units"]["<unk>"]
    return [_state["units"].get(p, unk) for p in pieces]


def score_phrases(crops: np.ndarray, phrases: list[str]) -> dict[str, float]:
    """log P(phrase | lips) for each phrase: attention decoder (with <eos>) and CTC, mixed 0.9 / 0.1."""
    import torch
    import torch.nn.functional as Fn

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask

    enc = _encode(crops)
    model, dev = _state["model"], _state["device"]
    eos = model.eos
    seqs = [_ids(p) for p in phrases]
    B, L = len(seqs), max(len(s) for s in seqs) + 1
    ys_in = torch.full((B, L), eos, dtype=torch.long)
    ys_out = torch.full((B, L), -1, dtype=torch.long)
    for b, s in enumerate(seqs):
        ys_in[b, 1:len(s) + 1] = torch.tensor(s)
        ys_out[b, :len(s)] = torch.tensor(s)
        ys_out[b, len(s)] = eos
    ys_in, ys_out = ys_in.to(dev), ys_out.to(dev)
    tgt_mask = subsequent_mask(L, device=dev).unsqueeze(0)
    mem = enc.expand(B, -1, -1)
    with torch.no_grad():
        logits, _ = model.decoder(ys_in, tgt_mask, mem, None)
        lp = logits.log_softmax(-1)
        keep = ys_out >= 0
        att = (lp.gather(-1, ys_out.clamp(min=0).unsqueeze(-1)).squeeze(-1) * keep).sum(1)
        ctc_lp = model.ctc.log_softmax(enc).transpose(0, 1).float().cpu()   # (T, 1, V)
        T = ctc_lp.shape[0]
        ctc = []
        for s in seqs:
            if len(s) > T:
                ctc.append(-1e4)
                continue
            ctc.append(-float(Fn.ctc_loss(ctc_lp, torch.tensor([s]), torch.tensor([T]), torch.tensor([len(s)]),
                                          blank=0, reduction="sum", zero_infinity=True)))
    att = att.float().cpu().numpy()
    return {p: float((1 - CTC_WEIGHT) * a + CTC_WEIGHT * c) for p, a, c in zip(phrases, att, ctc)}


def transcribe(crops: np.ndarray, beam: int = 8) -> str:
    """Free text, beam search with the authors' scorers (decoder 0.9 + CTC 0.1)."""
    import torch

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.scorers.length_bonus import LengthBonus

    enc = _encode(crops)
    model, tl = _state["model"], _state["token_list"]
    scorers = model.scorers()
    scorers["length_bonus"] = LengthBonus(len(tl))
    bs = BatchBeamSearch(beam_size=beam, vocab_size=len(tl),
                         weights={"decoder": 1 - CTC_WEIGHT, "ctc": CTC_WEIGHT, "length_bonus": 0.0},
                         scorers=scorers, sos=model.sos, eos=model.eos, token_list=tl, pre_beam_score_key="decoder")
    bs.to(device=_state["device"])
    with torch.no_grad():
        hyps = bs(enc.squeeze(0))
    if not hyps:
        return ""
    ids = [int(i) for i in hyps[0].asdict()["yseq"][1:] if int(i) != model.eos]
    return "".join(tl[i] for i in ids).replace("▁", " ").strip().lower()


def read_lips(crops: np.ndarray, phrases: list[str], free_text: bool = True) -> dict:
    """Everything /api/predict needs from the lip reader, with timings."""
    t0 = time.time()
    scores = score_phrases(crops, phrases)
    t1 = time.time()
    text = transcribe(crops) if free_text else ""
    text_conf = text_total = None
    if text:
        text_total = score_phrases(crops, [text])[text]
        text_conf = text_total / (len(_ids(text)) + 1)  # per word piece, incl. end
    vals = np.array(list(scores.values()))
    p = np.exp(vals - vals.max()); p /= p.sum()
    place = place_free_text(text, text_conf, text_total, scores)
    return {"scores": scores, "proba": dict(zip(scores, p.tolist())), "text": text,
            "text_conf": None if text_conf is None else round(float(text_conf), 3),
            "text_total": None if text_total is None else round(float(text_total), 3),
            "text_shown": place["show"], "text_first": place["first"],
            "timing": {"score_s": round(t1 - t0, 3), "text_s": round(time.time() - t1, 3), "frames": len(crops)}}
