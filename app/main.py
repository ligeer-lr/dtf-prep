"""واجهة FastAPI لموقع DTF Prep."""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__, config
from .jobs import manager
from .pipeline import Params, UserError, check_upload_header, validate_options, validate_params
from .sessions import store as sessions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("dtf")

STATIC_DIR = config.BASE_DIR / "static"
NO_STORE = {"Cache-Control": "no-store"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("DTF_PRELOAD", "1") == "1":
        threading.Thread(target=_warm_up, name="dtf-warmup", daemon=True).start()
    yield


app = FastAPI(
    title="DTF Prep", version=__version__, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
)


# ---------------------------------------------------------------------------
# معالجة الأخطاء برسائل عربية موحّدة {"error": "..."}
# ---------------------------------------------------------------------------
def _err(msg: str, code: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=code, headers=NO_STORE)


@app.exception_handler(UserError)
async def _user_error(_: Request, exc: UserError):
    return _err(str(exc), 400)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError):
    fields = {str(e.get("loc", ["", ""])[-1]) for e in exc.errors()}
    if "file" in fields:
        return _err("لم يتم رفع أي صورة. اختر صورة أولاً.", 422)
    if fields & {"width_cm", "height_cm"}:
        return _err("المقاس غير صحيح. أدخل العرض والطول بالسنتيمتر كأرقام.", 422)
    return _err("بيانات الطلب غير صحيحة.", 422)


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception):
    log.exception("خطأ غير متوقع: %s", exc)
    return _err("حدث خطأ غير متوقع في الخادم. حاول مرة أخرى.", 500)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """نرفض الطلبات الضخمة مبكراً قبل قراءة جسمها."""
    if request.method == "POST":
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > config.MAX_UPLOAD_BYTES + 1024 * 1024:
            return _err(f"حجم الملف كبير جداً. الحد الأقصى {config.MAX_UPLOAD_MB} ميجابايت.", 413)
    return await call_next(request)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/config")
async def get_config():
    return JSONResponse(
        {
            "max_upload_mb": config.MAX_UPLOAD_MB,
            "dpi": config.PRINT_DPI,
            "min_side_cm": config.MIN_SIDE_CM,
            "max_side_cm": config.MAX_SIDE_CM,
            "max_output_mp": config.MAX_OUTPUT_PIXELS // 1_000_000,
            "low_dpi_warning": config.LOW_DPI_WARNING,
            "preview_side": config.LIVE_PREVIEW_SIDE,
            "version": __version__,
        },
        headers=NO_STORE,
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


async def _save_upload(file: UploadFile, dest: Path) -> None:
    """يحفظ الملف المرفوع مع التحقق من الحجم والصيغة (يرمي UserError برسالة عربية)."""
    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    raise UserError(f"حجم الملف كبير جداً. الحد الأقصى {config.MAX_UPLOAD_MB} ميجابايت.")
                out.write(chunk)
    finally:
        await file.close()
    if size == 0:
        raise UserError("الملف المرفوع فارغ.")
    check_upload_header(dest)


def _upload_error(e: UserError) -> JSONResponse:
    code = 413 if "كبير جداً" in str(e) and "ميجابايت" in str(e) else 400
    return _err(str(e), code)


def _make_params(
    width_cm, height_cm, remove_bg, bg_mode, bg_tolerance, bg_color, erode_px, alpha_mode, alpha_threshold
) -> Params:
    return Params(
        width_cm=width_cm,
        height_cm=height_cm,
        remove_bg=remove_bg,
        bg_mode=bg_mode,
        bg_tolerance=bg_tolerance,
        bg_color=bg_color.strip(),
        erode_px=erode_px,
        alpha_mode=alpha_mode,
        alpha_threshold=alpha_threshold,
    )


SESSION_GONE = "انتهت صلاحية جلسة الصورة. ارفع الصورة من جديد."


@app.post("/api/process")
async def start_process(
    file: UploadFile | None = File(None),
    session_id: str = Form(""),
    width_cm: float = Form(...),
    height_cm: float = Form(...),
    remove_bg: bool = Form(True),
    bg_mode: str = Form("solid"),
    bg_tolerance: int = Form(25),
    bg_color: str = Form(""),
    erode_px: int = Form(1),
    alpha_mode: str = Form("threshold"),
    alpha_threshold: int = Form(50),
):
    """المعالجة الكاملة بدقة الطباعة. تقبل ملفاً مرفوعاً، أو session_id لجلسة المعاينة (بدون إعادة رفع)."""
    params = _make_params(
        width_cm, height_cm, remove_bg, bg_mode, bg_tolerance, bg_color, erode_px, alpha_mode, alpha_threshold
    )
    validate_params(params)  # نفشل بسرعة قبل حفظ أي شيء

    sess = None
    if session_id:
        sess = sessions.get(session_id)
        if sess is None:
            return _err(SESSION_GONE, 404)
    elif file is None:
        return _err("لم يتم رفع أي صورة. اختر صورة أولاً.", 422)

    jid, job_dir = manager.new_job_dir()
    upload = job_dir / "upload.bin"
    try:
        if sess is not None:
            await run_in_threadpool(shutil.copyfile, sess.orig, upload)
        else:
            await _save_upload(file, upload)
    except UserError as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        return _upload_error(e)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    manager.submit(jid, job_dir, upload, params)
    return JSONResponse({"job_id": jid}, headers=NO_STORE)


# ---------------------------------------------------------------------------
# جلسات المعاينة المباشرة
# ---------------------------------------------------------------------------
@app.post("/api/sessions")
async def create_session(file: UploadFile = File(...)):
    sid, sdir = sessions.new_dir()
    try:
        orig = sdir / "orig.bin"
        await _save_upload(file, orig)
        sess = await run_in_threadpool(sessions.create, sid, sdir, orig)
    except UserError as e:
        shutil.rmtree(sdir, ignore_errors=True)
        return _upload_error(e)
    except Exception:
        shutil.rmtree(sdir, ignore_errors=True)
        raise
    return JSONResponse(
        {
            "session_id": sess.id,
            "width": sess.width,
            "height": sess.height,
            "preview_width": int(sess.small_rgb.shape[1]),
            "preview_height": int(sess.small_rgb.shape[0]),
            "transparent": sess.transparent,
            "auto_bg": sess.auto_bg,
        },
        headers=NO_STORE,
    )


@app.post("/api/sessions/{sid}/preview")
def session_preview(
    sid: str,
    width_cm: float = Form(0.0),
    height_cm: float = Form(0.0),
    remove_bg: bool = Form(True),
    bg_mode: str = Form("solid"),
    bg_tolerance: int = Form(25),
    bg_color: str = Form(""),
    erode_px: int = Form(1),
    alpha_mode: str = Form("threshold"),
    alpha_threshold: int = Form(50),
):
    """معاينة سريعة على نسخة مصغّرة: PNG شفاف بنفس مراحل المعالجة (بدون تكبير)."""
    sess = sessions.get(sid)
    if sess is None:
        return _err(SESSION_GONE, 404)
    params = _make_params(
        width_cm, height_cm, remove_bg, bg_mode, bg_tolerance, bg_color, erode_px, alpha_mode, alpha_threshold
    )
    validate_options(params)
    res = sessions.render_preview(sess, params, width_cm, height_cm)
    return Response(
        content=res.png,
        media_type="image/png",
        headers={
            **NO_STORE,
            "X-Bg-Color": res.bg_color,
            "X-Bg-Manual": "1" if res.bg_manual else "0",
            "X-Render-Ms": str(res.ms),
        },
    )


@app.delete("/api/sessions/{sid}")
async def session_delete(sid: str):
    sessions.delete(sid)
    return JSONResponse({"ok": True}, headers=NO_STORE)


def _job_or_404(jid: str):
    job = manager.get(jid)
    if job is None:
        return None, _err("انتهت صلاحية هذه المعالجة أو لم تعد موجودة. ارفع الصورة من جديد.", 404)
    return job, None


@app.get("/api/jobs/{jid}")
async def job_status(jid: str):
    job, err = _job_or_404(jid)
    if err:
        return err
    return JSONResponse(job.public(manager.queue_position(jid)), headers=NO_STORE)


def _file_response(jid: str, which: str, download_name: str | None = None):
    job, err = _job_or_404(jid)
    if err:
        return err
    if job.status != "done" or not job.result:
        return _err("النتيجة غير جاهزة بعد.", 409)
    path: Path = {
        "before": job.result.before_path,
        "preview": job.result.preview_path,
        "result": job.result.result_path,
    }[which]
    if not path.exists():
        return _err("الملف لم يعد متاحاً. أعد المعالجة.", 404)
    media = "image/jpeg" if path.suffix == ".jpg" else "image/png"
    return FileResponse(path, media_type=media, filename=download_name, headers=NO_STORE)


@app.get("/api/jobs/{jid}/before")
async def job_before(jid: str):
    return _file_response(jid, "before")


@app.get("/api/jobs/{jid}/preview")
async def job_preview(jid: str):
    return _file_response(jid, "preview")


@app.get("/api/jobs/{jid}/result")
async def job_result(jid: str):
    job = manager.get(jid)
    name = "dtf-design.png"
    if job and job.result:
        w, h = job.result.info["output_cm"]
        name = f"dtf-{w:g}x{h:g}cm-300dpi.png"
    return _file_response(jid, "result", download_name=name)


@app.delete("/api/jobs/{jid}")
async def job_delete(jid: str):
    manager.delete(jid)
    return JSONResponse({"ok": True}, headers=NO_STORE)


# ---------------------------------------------------------------------------
# تسخين النماذج في الخلفية لتسريع أول معالجة
# ---------------------------------------------------------------------------
def _warm_up() -> None:
    try:
        from .esrgan import ensure_onnx_model
        from .pipeline import _get_rembg_session

        ensure_onnx_model()
        _get_rembg_session()
        log.info("النماذج جاهزة")
    except Exception:  # noqa: BLE001
        log.exception("تعذّر تسخين النماذج (ستُحمَّل عند أول طلب)")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
