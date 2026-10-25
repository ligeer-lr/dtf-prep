"""اختبارات وضع «تصميم بخلفية سادة» (flood fill من الحواف)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from app.main import app
from app.pipeline import UserError, detect_bg_color, remove_solid_background
from tests.test_pipeline import png_bytes, run_job


def design_with_white_text(bg=(255, 255, 255), size=(600, 400)) -> Image.Image:
    """شكل أزرق فيه نص أبيض ونجمة بيضاء وثقب بلون الخلفية، على خلفية سادة."""
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([80, 60, 520, 340], radius=40, fill=(20, 80, 190))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 90)
    except OSError:
        font = ImageFont.load_default()
    d.text((300, 200), "DTF", fill=(255, 255, 255), font=font, anchor="mm")
    d.ellipse([110, 90, 160, 140], fill=bg)  # ثقب بلون الخلفية لكنه داخل الشكل
    return im


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_detect_bg_color_from_borders():
    arr = np.asarray(design_with_white_text(bg=(250, 240, 230)))
    assert tuple(detect_bg_color(arr)) == (250, 240, 230)


def test_flood_fill_keeps_inner_white_text_and_holes():
    arr = np.asarray(design_with_white_text())
    alpha, bg, manual = remove_solid_background(arr, "", 25)
    assert not manual and tuple(bg) == (255, 255, 255)
    assert alpha[2, 2] == 0 and alpha[-3, -3] == 0  # الزوايا شُفِّفت
    assert alpha[200, 100] == 255  # جسم الشكل
    # النص الأبيض (داخل الشكل) باقٍ معتم
    text = (arr[130:270, 200:400] > 245).all(axis=2)
    assert text.sum() > 500
    assert (alpha[130:270, 200:400][text] == 255).all()
    # الثقب بلون الخلفية لكنه محاط بالشكل فيبقى
    assert alpha[115, 135] == 255


def test_manual_color_and_tolerance_for_jpeg_noise():
    img = design_with_white_text(bg=(200, 200, 200))
    b = io.BytesIO()
    img.save(b, "JPEG", quality=60)
    arr = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
    alpha, bg, manual = remove_solid_background(arr, "#c8c8c8", 30)
    assert manual and tuple(bg) == (200, 200, 200)
    assert (alpha[:20, :20] == 0).mean() > 0.99


def test_no_solid_background_gives_arabic_error():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    with pytest.raises(UserError, match="خلفية سادة"):
        remove_solid_background(noise, "", 25)


def test_end_to_end_default_mode_is_solid_and_text_survives(client):
    buf = io.BytesIO()
    design_with_white_text().save(buf, "PNG")
    jid, j = run_job(client, buf.getvalue(), width_cm=10, height_cm=6.67)  # الوضع الافتراضي = خلفية سادة
    assert j["status"] == "done", j
    assert any("لون الخلفية" in n and "#ffffff" in n for n in j["notes"])
    im = Image.open(io.BytesIO(client.get(f"/api/jobs/{jid}/result").content))
    assert im.size == (1181, 788) and im.mode == "RGBA"
    rgba = np.asarray(im)
    a = rgba[..., 3]
    assert a[3, 3] == 0 and a[-4, -4] == 0
    # مكان النص الأبيض (المركز تقريباً) يجب أن يحتوي بكسلات بيضاء معتمة
    cy, cx = im.height // 2, im.width // 2
    region = rgba[cy - 90: cy + 90, cx - 160: cx + 160]
    white_opaque = ((region[..., :3] > 235).all(axis=2)) & (region[..., 3] == 255)
    assert white_opaque.sum() > 2000, "النص الأبيض حُذف مع الخلفية"
    # لا يوجد بكسل أبيض معتم قرب الحافة الخارجية (لا هالة)
    outer = rgba[:40, :]
    assert (outer[..., 3] == 255).sum() == 0


def test_manual_color_via_api(client):
    buf = io.BytesIO()
    design_with_white_text(bg=(200, 200, 200)).save(buf, "PNG")
    jid, j = run_job(client, buf.getvalue(), width_cm=8, height_cm=5, bg_color="#c8c8c8", bg_tolerance=20)
    assert j["status"] == "done", j
    assert any("المحدد يدوياً" in n for n in j["notes"])


def test_invalid_mode_and_color_rejected(client):
    buf = png_bytes(Image.new("RGB", (20, 20), "white"))
    r = client.post("/api/process", files={"file": ("a.png", buf)},
                    data={"width_cm": "5", "height_cm": "5", "bg_mode": "x"})
    assert r.status_code == 400 and "وضع" in r.json()["error"]
    r = client.post("/api/process", files={"file": ("a.png", buf)},
                    data={"width_cm": "5", "height_cm": "5", "bg_color": "red"})
    assert r.status_code == 400 and "لون الخلفية" in r.json()["error"]
