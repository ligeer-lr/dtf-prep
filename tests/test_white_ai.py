"""اختبارات «الأبيض إلى شفاف» و«قوة التكبير الذكي»."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import pipeline
from app.main import app
from app.pipeline import Params, UserError, process_image, validate_options, white_to_alpha
from tests.test_preview import decode, new_session, preview


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def swatch_image() -> np.ndarray:
    """أزرق + أبيض + كريمي + بيج + رمادي فاتح جداً + بقعة بيضاء صغيرة داخل الأزرق."""
    a = np.full((60, 120, 3), (20, 80, 190), np.uint8)
    a[:, 0:20] = (255, 255, 255)  # أبيض
    a[:, 20:40] = (255, 253, 208)  # كريمي
    a[:, 40:60] = (245, 245, 220)  # بيج
    a[:, 60:80] = (250, 250, 250)  # أبيض مائل للرمادي
    a[25:35, 100:110] = (255, 255, 255)  # بقعة داخل الأزرق
    return a


def test_white_becomes_transparent_cream_and_beige_stay():
    rgb = swatch_image()
    alpha = np.full(rgb.shape[:2], 255, np.uint8)
    new_rgb, new_a = white_to_alpha(rgb, alpha, 30)
    assert new_a[30, 5] == 0  # أبيض
    assert new_a[30, 70] == 0  # 250,250,250 يُعد أبيض
    assert new_a[30, 30] == 255  # كريمي
    assert new_a[30, 50] == 255  # بيج
    assert tuple(new_rgb[30, 30]) == (255, 253, 208) and tuple(new_rgb[30, 50]) == (245, 245, 220)
    # الأزرق لا يتغير إطلاقاً، والبقعة البيضاء داخل التصميم صارت شفافة
    assert new_a[5, 90] == 255 and tuple(new_rgb[5, 90]) == (20, 80, 190)
    assert (new_a[27:33, 102:108] == 0).all()


def test_soft_transition_and_unmixed_color():
    # تدرج من الأبيض إلى الأزرق: ألفا يتدرج، واللون يُفكّ مزجه عن الأبيض (بلا هالة بيضاء)
    t = np.linspace(0, 1, 256, dtype=np.float32)[None, :, None]
    blue = np.array([20, 80, 190], np.float32)
    rgb = np.repeat((255 * (1 - t) + blue * t), 8, axis=0).astype(np.uint8)
    alpha = np.full(rgb.shape[:2], 255, np.uint8)
    _, a = white_to_alpha(rgb, alpha, 30)
    row = a[0]
    assert row[0] == 0 and row[-1] == 255
    mid = row[(row > 0) & (row < 255)]
    d = np.diff(row.astype(int))
    assert len(mid) >= 10 and d.max() <= 80 and d.min() >= -20  # تدرج ناعم (مع تفاوت تكميم 8 بت)
    new_rgb, a2 = white_to_alpha(rgb, alpha, 30)
    k = int(np.argmax((a2[0] > 100) & (a2[0] < 200)))
    assert new_rgb[0, k, 0] < rgb[0, k, 0]  # أقرب للأزرق من الأصلي الممزوج بالأبيض


def test_threshold_widens_selection_and_existing_alpha_kept():
    rgb = np.full((10, 10, 3), (235, 235, 235), np.uint8)  # رمادي فاتح
    alpha = np.full((10, 10), 255, np.uint8)
    alpha[:, :5] = 0
    _, low = white_to_alpha(rgb, alpha, 0)
    _, high = white_to_alpha(rgb, alpha, 100)
    assert low[0, 8] > high[0, 8]
    assert (low[:, :5] == 0).all() and (high[:, :5] == 0).all()


def test_validation_messages():
    ok = Params(10, 10)
    validate_options(ok)
    for kw, msg in (({"white_threshold": 101}, "عتبة الأبيض"), ({"ai_strength": -1}, "قوة التكبير الذكي"),
                    ({"ai_strength": 101}, "قوة التكبير الذكي")):
        with pytest.raises(UserError, match=msg):
            validate_options(Params(10, 10, **kw))
    assert ok.white_to_alpha is False and ok.ai_strength == 60


def test_preview_white_to_alpha_off_by_default_and_on(client):
    im = Image.fromarray(np.dstack([swatch_image()]), "RGB")
    sid = new_session(client, im)["session_id"]
    base = dict(remove_bg="false", width_cm=0, height_cm=0)
    off = decode(preview(client, sid, **base))
    assert (off[..., 3] == 255).all()  # الافتراضي: لا شيء يتغير
    on = decode(preview(client, sid, white_to_alpha="true", white_threshold=30, **base))
    h, w = on.shape[:2]
    x = lambda px: int(px / 120 * w)  # noqa: E731
    assert on[h // 2, x(10), 3] == 0 and on[h // 2, x(30), 3] == 255 and on[h // 2, x(50), 3] == 255
    assert on[int(h * 0.1), x(90), 3] == 255
    bad = preview(client, sid, white_to_alpha="true", white_threshold=500, **base)
    assert bad.status_code == 400


class FakeUp:
    """بديل Real-ESRGAN: يعيد صورة سوداء ×4 (فرق واضح عن Lanczos لقياس المزج)."""

    def __init__(self):
        self.calls = 0

    def upscale4x(self, rgb, alpha=None, progress=None):
        self.calls += 1
        if progress:
            progress(1.0)
        return np.zeros((rgb.shape[0] * 4, rgb.shape[1] * 4, 3), np.uint8)


def run_up(tmp_path, monkeypatch, strength):
    fake = FakeUp()
    monkeypatch.setattr(pipeline, "upscaler", fake)
    src = tmp_path / f"in{strength}.png"
    Image.new("RGB", (100, 100), (200, 200, 200)).save(src)
    res = process_image(
        src, tmp_path / f"o{strength}", Params(5, 5, remove_bg=False, alpha_mode="keep", ai_strength=strength),
        lambda f, s: None,
    )
    out = np.asarray(Image.open(res.result_path).convert("RGBA"))
    return fake, res, out


def test_ai_strength_zero_skips_esrgan(tmp_path, monkeypatch):
    fake, res, out = run_up(tmp_path, monkeypatch, 0)
    assert fake.calls == 0 and res.info["ai_upscale"] is False
    assert abs(int(out[300, 300, 0]) - 200) <= 1


def test_ai_strength_blends_with_lanczos(tmp_path, monkeypatch):
    f100, r100, o100 = run_up(tmp_path, monkeypatch, 100)
    f60, r60, o60 = run_up(tmp_path, monkeypatch, 60)
    assert f100.calls >= 1 and f60.calls >= 1 and r60.info["ai_upscale"] is True
    v100, v60 = int(o100[300, 300, 0]), int(o60[300, 300, 0])
    assert v100 <= 2  # ناتج الذكاء كاملاً (أسود في البديل)
    assert abs(v60 - 200 * 0.4) <= 3  # 60% ذكاء + 40% Lanczos
    assert o60.shape == (591, 591, 4) and Image.open(r60.result_path).info["dpi"][0] == pytest.approx(300, abs=0.5)


def test_process_endpoint_accepts_new_fields(client):
    b = io.BytesIO()
    Image.fromarray(swatch_image(), "RGB").save(b, "PNG")
    from tests.test_pipeline import run_job
    jid, j = run_job(client, b.getvalue(), remove_bg="false", white_to_alpha="true", white_threshold=30, ai_strength=0)
    assert j["status"] == "done", j
    res = client.get(f"/api/jobs/{jid}/result")
    a = np.asarray(Image.open(io.BytesIO(res.content)).convert("RGBA"))
    assert a[..., 3].min() == 0 and a[..., 3].max() == 255
    r = client.post("/api/process", files={"file": ("a.png", b.getvalue())},
                    data={"width_cm": "10", "height_cm": "10", "ai_strength": "150"})
    assert r.status_code == 400
