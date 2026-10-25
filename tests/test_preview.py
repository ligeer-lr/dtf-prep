"""اختبارات الجلسات والمعاينة المباشرة (/api/sessions)."""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app import config
from app.main import app
from app.pipeline import Params, erode_alpha
from tests.test_pipeline import png_bytes, run_job
from tests.test_solid_bg import design_with_white_text


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def new_session(client, im: Image.Image, fmt="PNG"):
    b = io.BytesIO()
    im.save(b, fmt, **({"quality": 85} if fmt == "JPEG" else {}))
    r = client.post("/api/sessions", files={"file": ("a." + fmt.lower(), b.getvalue())})
    assert r.status_code == 200, r.text
    return r.json()


def preview(client, sid, **form):
    data = {k: str(v) for k, v in form.items()}
    return client.post(f"/api/sessions/{sid}/preview", data=data)


def decode(r) -> np.ndarray:
    assert r.status_code == 200, r.text
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGBA"))


def test_fractional_erode_is_between_integer_steps():
    a = np.zeros((40, 40), np.uint8)
    a[10:30, 10:30] = 255
    e0, e1, eh = erode_alpha(a, 0), erode_alpha(a, 1), erode_alpha(a, 0.5)
    assert int(e0.sum(dtype=np.int64)) > int(eh.sum(dtype=np.int64)) > int(e1.sum(dtype=np.int64))
    assert (erode_alpha(a, 2) == erode_alpha(a, 2.0)).all()


def test_session_info_and_preview_size(client):
    big = design_with_white_text(size=(2400, 1600)).resize((2400, 1600))
    info = new_session(client, big)
    assert (info["width"], info["height"]) == (2400, 1600)
    assert max(info["preview_width"], info["preview_height"]) == config.LIVE_PREVIEW_SIDE
    assert info["transparent"] is False and info["auto_bg"] == "#ffffff"
    r = preview(client, info["session_id"])
    a = decode(r)
    assert a.shape[:2] == (info["preview_height"], info["preview_width"])
    assert r.headers["x-bg-color"] == "#ffffff" and r.headers["x-bg-manual"] == "0"
    assert a[2, 2, 3] == 0


def test_preview_keeps_inner_white_text(client):
    info = new_session(client, design_with_white_text())
    a = decode(preview(client, info["session_id"]))
    h, w = a.shape[:2]
    region = a[h // 2 - 70: h // 2 + 70, w // 2 - 130: w // 2 + 130]
    white_opaque = ((region[..., :3] > 235).all(axis=2)) & (region[..., 3] == 255)
    assert white_opaque.sum() > 1000, "النص الأبيض حُذف في المعاينة"


def test_preview_options_change_output(client):
    info = new_session(client, design_with_white_text())
    sid = info["session_id"]
    base = decode(preview(client, sid, erode_px=0))
    eroded = decode(preview(client, sid, erode_px=3))
    assert (eroded[..., 3] > 0).sum() < (base[..., 3] > 0).sum()
    keep = decode(preview(client, sid, alpha_mode="keep"))
    thr = decode(preview(client, sid, alpha_mode="threshold"))
    assert set(np.unique(thr[..., 3])) <= {0, 255}
    assert ((keep[..., 3] > 0) & (keep[..., 3] < 255)).sum() > 0
    # بدون إزالة خلفية: كل البكسلات معتمة
    none = decode(preview(client, sid, remove_bg="false"))
    assert (none[..., 3] == 255).all()


def test_manual_color_and_no_solid_bg_error(client):
    img = design_with_white_text(bg=(200, 200, 200))
    info = new_session(client, img)
    r = preview(client, info["session_id"], bg_color="#c8c8c8")
    assert r.status_code == 200 and r.headers["x-bg-manual"] == "1"
    noise = Image.fromarray(np.random.default_rng(1).integers(0, 255, (200, 200, 3), dtype=np.uint8))
    info2 = new_session(client, noise)
    r = preview(client, info2["session_id"])
    assert r.status_code == 400 and "خلفية سادة" in r.json()["error"]


def test_transparent_image_skips_removal(client):
    im = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([50, 30, 250, 170], fill=(20, 90, 200, 255))
    info = new_session(client, im)
    assert info["transparent"] is True and info["auto_bg"] == ""
    a = decode(preview(client, info["session_id"]))
    assert a[0, 0, 3] == 0 and a[100, 150, 3] == 255


def test_bad_inputs_and_missing_session(client):
    assert preview(client, "nope").status_code == 404
    info = new_session(client, design_with_white_text())
    r = preview(client, info["session_id"], bg_mode="x")
    assert r.status_code == 400 and "وضع" in r.json()["error"]
    r = client.post("/api/sessions", files={"file": ("a.txt", b"hello")})
    assert r.status_code == 400
    r = client.post("/api/sessions", files={"file": ("a.png", b"\0" * (config.MAX_UPLOAD_BYTES + 10))})
    assert r.status_code == 413
    assert client.delete(f"/api/sessions/{info['session_id']}").status_code == 200
    assert preview(client, info["session_id"]).status_code == 404


def test_process_from_session_without_reupload(client):
    info = new_session(client, design_with_white_text())
    sid = info["session_id"]
    r = client.post(
        "/api/process",
        data={"session_id": sid, "width_cm": "10", "height_cm": "6.67"},
    )
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    for _ in range(300):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.4)
    assert j["status"] == "done", j
    im = Image.open(io.BytesIO(client.get(f"/api/jobs/{jid}/result").content))
    assert im.size == (1181, 788) and round(im.info["dpi"][0]) == 300
    # الجلسة باقية لإعادة التصدير بمقاس آخر
    assert preview(client, sid).status_code == 200
    r = client.post("/api/process", data={"session_id": "gone", "width_cm": "10", "height_cm": "10"})
    assert r.status_code == 404 and "جلسة" in r.json()["error"]


def test_preview_latency_budget(client):
    """المعاينة على نسخة 1000px: التعديلات الخفيفة (حواف/شفافية) وتغيير الحساسية تحت ثانية على CPU متواضع."""
    im = design_with_white_text(size=(3000, 2000))
    info = new_session(client, im, "JPEG")
    sid = info["session_id"]
    preview(client, sid)  # تسخين
    light, heavy = [], []
    for e in (0, 1, 2, 3, 1):
        t = time.perf_counter()
        r = preview(client, sid, erode_px=e)
        assert r.status_code == 200
        light.append(time.perf_counter() - t)
    for tol in (10, 30, 50, 70):
        t = time.perf_counter()
        r = preview(client, sid, bg_tolerance=tol)
        assert r.status_code == 200
        heavy.append(time.perf_counter() - t)
    print(f"\nlight(median)={sorted(light)[2]*1000:.0f}ms heavy(median)={sorted(heavy)[2]*1000:.0f}ms")
    assert sorted(light)[2] < 0.5
    assert sorted(heavy)[2] < 1.5


def test_ai_preview_mask_is_cached(client):
    img = Image.new("RGB", (500, 500), (240, 240, 240))
    ImageDraw.Draw(img).ellipse([100, 100, 400, 400], fill=(200, 30, 30))
    info = new_session(client, img, "JPEG")
    sid = info["session_id"]
    t = time.perf_counter()
    r = preview(client, sid, bg_mode="ai")
    first = time.perf_counter() - t
    if r.status_code != 200:
        pytest.skip(f"نموذج إزالة الخلفية غير متاح: {r.text}")
    a = decode(r)
    assert a[5, 5, 3] == 0 and a[250, 250, 3] == 255
    assert r.headers["x-bg-color"] == ""  # لا يوجد لون خلفية في وضع الذكاء الاصطناعي
    t = time.perf_counter()
    assert preview(client, sid, bg_mode="ai", erode_px=2).status_code == 200
    second = time.perf_counter() - t
    print(f"\nai first={first*1000:.0f}ms then={second*1000:.0f}ms")
    assert second < first
