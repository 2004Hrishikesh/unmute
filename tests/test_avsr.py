"""Open-source lip reader glue: crop alignment, frame-rate resampling, blending. The model itself is only
smoke-tested when its weights are installed (third_party/, not in git)."""
import base64

import numpy as np
import pytest

import server.app as appmod
from server import avsr

needs_repo = pytest.mark.skipif(not (avsr.REPO / "preparation").exists(), reason="Auto-AVSR not installed")


def test_resample_fps_picks_a_steady_25fps_clock():
    ts = list(np.cumsum([0] + [16.7] * 119))  # 2 s at 60 fps
    idx = avsr.resample_fps(list(range(120)), ts)
    assert 49 <= len(idx) <= 51 and idx == sorted(idx) and idx[0] == 0


@needs_repo
def test_mouth_crops_align_the_mouth_into_the_centre():
    ref = avsr._stable_reference()
    # A synthetic "face": a bright square at the mouth position, everything else dark; drawn at half scale and shifted
    frames, kps = [], []
    for t in range(12):
        img = np.zeros((192, 192), np.uint8)
        kp = ref * 0.6 + np.array([10 + t, 5])
        mx, my = kp[3].astype(int)
        img[my - 4:my + 5, mx - 4:mx + 5] = 255
        frames.append(img); kps.append(kp)
    crops = avsr.mouth_crops(frames, np.stack(kps))
    assert crops.shape == (12, 96, 96) and crops.dtype == np.uint8
    ys, xs = np.nonzero(crops[6] > 128)
    assert abs(ys.mean() - 48) < 6 and abs(xs.mean() - 48) < 6   # the mouth lands in the middle of the crop


def test_blend_scores_every_phrase_and_respects_the_weight():
    ranked = [{"phrase": "Yes", "distance": 0.1}, {"phrase": "No", "distance": 0.5}]
    lip = {"proba": {"Yes": 0.1, "No": 0.2, "I need water": 0.7}}
    lip_only = appmod.blend_with_lip_reader(ranked, lip, 1.0)
    assert [c["phrase"] for c in lip_only] == ["I need water", "No", "Yes"]     # can pick a phrase nobody recorded
    matcher_only = appmod.blend_with_lip_reader(ranked, lip, 0.0)
    assert matcher_only[0]["phrase"] == "Yes"
    assert abs(sum(c["confidence"] for c in lip_only) - 1) < 1e-3
    assert lip_only[0]["distance"] is None and matcher_only[0]["distance"] == 0.1


@needs_repo
def test_decode_crops_from_browser_payload(monkeypatch):
    import cv2

    monkeypatch.setattr(avsr, "available", lambda: True)
    ref = avsr._stable_reference() * 0.75
    ok, jpg = cv2.imencode(".jpg", np.full((192, 192), 90, np.uint8))
    url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
    payload = {"imgs": [url] * 40, "kp": [ref.tolist()] * 40, "ts": [i * 33.3 for i in range(40)]}
    crops = appmod.decode_crops(payload)
    assert crops.shape[1:] == (96, 96) and 30 <= len(crops) <= 35  # 1.3 s at 25 fps
    assert appmod.decode_crops({"imgs": [url] * 3, "kp": [ref.tolist()] * 3, "ts": [0, 1, 2]}) is None
    assert appmod.decode_crops(None) is None


@pytest.mark.skipif(not avsr.WEIGHTS.exists(), reason="lip reader weights not installed")
def test_real_model_scores_every_phrase():
    crops = (np.random.default_rng(0).random((40, 96, 96)) * 255).astype(np.uint8)
    out = avsr.read_lips(crops, ["I need water", "Yes", "Call the nurse"], free_text=False)
    assert set(out["proba"]) == {"I need water", "Yes", "Call the nurse"}
    assert abs(sum(out["proba"].values()) - 1) < 1e-6 and all(np.isfinite(list(out["scores"].values())))
