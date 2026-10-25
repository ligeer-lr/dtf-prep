"""اختبارات: مقاس الإخراج، الشفافية، الـ DPI، والأخطاء العربية.

تحتاج أوزان Real-ESRGAN (تُنزَّل تلقائياً). اختبار إزالة الخلفية يحتاج نموذج rembg.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config
from app.main import app
from app.pipeline import Params, UserError, cm_to_px, validate_params
from tests.make_sample import make_logo


def transparent_logo_with_halo(w=420, h=300) -> Image.Image:
    """شعار على خلفية شفافة وحوافه ملوّثة بالأبيض (هالة)."""
    rgb = np.asarray(make_logo(w, h, background=(255, 255, 255)).convert("RGB")).astype(np.float32)
    white = (rgb > 245).all(axis=2)
    alpha = np.where(white, 0, 255).astype(np.uint8)
    # حواف ناعمة + هالة بيضاء على البكسلات شبه الشفافة
    import cv2

    soft = cv2.GaussianBlur(alpha, (0, 0), 1.2)
    edge = (soft > 0) & (soft < 255)
    rgb[edge] = rgb[edge] * 0.35 + 255 * 0.65
    return Image.fromarray(np.dstack([rgb.astype(np.uint8), soft]), "RGBA")


def png_bytes(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def run_job(client, data: bytes, filename="a.png", **form):
    form = {"width_cm": "10", "height_cm": "10", **{k: str(v) for k, v in form.items()}}
    r = client.post("/api/process", files={"file": (filename, data)}, data=form)
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    for _ in range(600):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            return jid, j
        time.sleep(0.5)
    raise AssertionError("انتهت مهلة المعالجة")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_cm_to_px():
    assert cm_to_px(2.54) == 300
    assert cm_to_px(60) == 7087
    assert cm_to_px(40) == 4724


def test_validate_params_messages_are_arabic():
    with pytest.raises(UserError, match="كبير"):
        validate_params(Params(width_cm=500, height_cm=10))
    with pytest.raises(UserError, match="صغير"):
        validate_params(Params(width_cm=0.2, height_cm=10))
    with pytest.raises(UserError, match="ميجابكسل"):
        validate_params(Params(width_cm=100, height_cm=100))


def test_transparent_input_exact_size_alpha_dpi(client):
    jid, j = run_job(client, png_bytes(transparent_logo_with_halo()), width_cm=12, height_cm=8)
    assert j["status"] == "done", j
    r = client.get(f"/api/jobs/{jid}/result")
    assert r.status_code == 200
    im = Image.open(io.BytesIO(r.content))
    assert im.size == (cm_to_px(12), cm_to_px(8)) == (1417, 945)
    assert im.mode == "RGBA"
    assert round(im.info["dpi"][0]) == 300 and round(im.info["dpi"][1]) == 300
    a = np.asarray(im)[..., 3]
    assert set(np.unique(a)) <= {0, 255}  # الوضع الافتراضي: قطع حاد
    assert (a == 0).mean() > 0.2 and (a == 255).mean() > 0.1
    assert any("شفافة أصلاً" in n for n in j["notes"])
    # لا هالة بيضاء: لون أقرب بكسلات الحافة المعتمة ليس أبيض
    rgb = np.asarray(im)[..., :3].astype(int)
    import cv2

    edge = (a == 255) & (cv2.erode((a == 255).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)
    assert (rgb[edge].min(axis=1) > 235).mean() < 0.02
    # المعاينة والمعاينة السابقة متاحتان ثم تُحذف الملفات
    assert client.get(f"/api/jobs/{jid}/preview").status_code == 200
    assert client.get(f"/api/jobs/{jid}/before").status_code == 200
    assert client.delete(f"/api/jobs/{jid}").status_code == 200
    assert client.get(f"/api/jobs/{jid}").status_code == 404


def test_keep_mode_preserves_partial_alpha(client):
    jid, j = run_job(client, png_bytes(transparent_logo_with_halo()), width_cm=10, height_cm=7, alpha_mode="keep")
    assert j["status"] == "done", j
    im = Image.open(io.BytesIO(client.get(f"/api/jobs/{jid}/result").content))
    a = np.asarray(im)[..., 3]
    assert ((a > 0) & (a < 255)).sum() > 100


def test_large_input_is_downscaled_to_print_size(client):
    big = Image.new("RGBA", (3000, 2000), (0, 0, 0, 0))
    big.paste(Image.new("RGBA", (1500, 1000), (30, 90, 200, 255)), (750, 500))
    jid, j = run_job(client, png_bytes(big), width_cm=5, height_cm=5)
    assert j["status"] == "done"
    im = Image.open(io.BytesIO(client.get(f"/api/jobs/{jid}/result").content))
    assert im.size == (591, 591)
    assert not j["info"]["ai_upscale"]


def test_errors_are_arabic(client):
    r = client.post("/api/process", files={"file": ("x.txt", b"hello")}, data={"width_cm": "10", "height_cm": "10"})
    assert r.status_code == 400 and "صورة" in r.json()["error"]
    r = client.post("/api/process", files={"file": ("a.png", b"x")}, data={"width_cm": "abc", "height_cm": "10"})
    assert r.status_code == 422 and "المقاس" in r.json()["error"]
    r = client.post("/api/process", data={"width_cm": "10", "height_cm": "10"})
    assert r.status_code == 422 and "صورة" in r.json()["error"]
    big = b"\0" * (config.MAX_UPLOAD_BYTES + 10)
    r = client.post("/api/process", files={"file": ("a.png", big)}, data={"width_cm": "10", "height_cm": "10"})
    assert r.status_code == 413 and "ميجابايت" in r.json()["error"]
    r = client.post(
        "/api/process", files={"file": ("a.png", png_bytes(Image.new("RGB", (10, 10))))},
        data={"width_cm": "300", "height_cm": "10"},
    )
    assert r.status_code == 400 and "كبير" in r.json()["error"]


def test_low_resolution_warning(client):
    tiny = transparent_logo_with_halo(120, 90)
    jid, j = run_job(client, png_bytes(tiny), width_cm=30, height_cm=22)
    assert j["status"] == "done"
    assert any("ضعيفة" in w for w in j["warnings"])
    im = Image.open(io.BytesIO(client.get(f"/api/jobs/{jid}/result").content))
    assert im.size == (cm_to_px(30), cm_to_px(22))


def test_background_removal_on_opaque_jpeg(client):
    img = Image.new("RGB", (500, 500), (240, 240, 240))
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    d.ellipse([100, 100, 400, 400], fill=(200, 30, 30))
    b = io.BytesIO()
    img.save(b, "JPEG", quality=85)
    jid, j = run_job(client, b.getvalue(), "a.jpg", width_cm=8, height_cm=8, remove_bg="true", bg_mode="ai")
    if j["status"] == "error":
        pytest.skip(f"نموذج إزالة الخلفية غير متاح: {j['error']}")
    im = Image.open(io.BytesIO(client.get(f"/api/jobs/{jid}/result").content))
    a = np.asarray(im)[..., 3]
    assert im.size == (945, 945)
    assert a[5, 5] == 0 and a[472, 472] == 255
