"""جلسات المعاينة المباشرة.

عند رفع الصورة تُنشأ جلسة تحتفظ بالأصل (للمعالجة النهائية) وبنسخة مصغّرة (أطول ضلع 1000 بكسل) في الذاكرة.
كل تحريك لسلايدر يطلب معاينة تُحسب على النسخة المصغّرة فقط، مع كاش لأثقل مرحلتين:
- قناع الذكاء الاصطناعي (يُحسب مرة واحدة لكل صورة).
- نتيجة إزالة الخلفية + إزالة الهالة (تُعاد فقط عند تغيير الوضع أو اللون أو الحساسية).
فتغيير «تصغير الحواف» أو «حدة الشفافية» خفيف جداً.

المعاينة تقريبية: تُحسب على نسخة مصغّرة، فقد يختلف الناتج النهائي بفروق طفيفة عند الحواف.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config
from .pipeline import (
    Params,
    cleanup_alpha,
    white_to_alpha,
    cm_to_px,
    compute_bg_alpha,
    defringe,
    detect_bg_color,
    load_image,
    remove_background,
    rgb_to_hex,
)

log = logging.getLogger("dtf.sessions")

_CACHE_ENTRIES = 3


@dataclass
class Session:
    id: str
    dir: Path
    orig: Path
    small_rgb: np.ndarray
    small_alpha: np.ndarray
    transparent: bool
    width: int
    height: int
    auto_bg: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    ai_mask: Optional[np.ndarray] = None
    cache: dict = field(default_factory=dict)
    touched: float = field(default_factory=time.time)


@dataclass
class PreviewResult:
    png: bytes
    bg_color: str  # لون الخلفية المستخدم في الوضع السادة، وإلا فارغ
    bg_manual: bool
    ms: int


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        config.TMP_ROOT.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._janitor, name="dtf-sessions-janitor", daemon=True).start()

    # ---- إنشاء ----
    def new_dir(self) -> tuple[str, Path]:
        sid = uuid.uuid4().hex[:16]
        d = config.TMP_ROOT / f"s-{sid}"
        d.mkdir(parents=True, exist_ok=True)
        return sid, d

    def create(self, sid: str, sdir: Path, orig: Path) -> Session:
        """يقرأ الأصل (قد يرمي UserError) ويبني النسخة المصغّرة."""
        rgb, alpha, meta = load_image(orig)
        h, w = alpha.shape
        s = min(1.0, config.LIVE_PREVIEW_SIDE / max(w, h))
        if s < 1.0:
            size = (max(1, round(w * s)), max(1, round(h * s)))
            small_rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            small_alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
        else:
            small_rgb, small_alpha = rgb, alpha
        del rgb, alpha
        transparent = bool(meta["transparent"])
        auto_bg = "" if transparent else rgb_to_hex(detect_bg_color(small_rgb))
        sess = Session(
            id=sid, dir=sdir, orig=orig, small_rgb=np.ascontiguousarray(small_rgb),
            small_alpha=np.ascontiguousarray(small_alpha), transparent=transparent,
            width=w, height=h, auto_bg=auto_bg,
        )
        evict: list[Session] = []
        with self._lock:
            self._sessions[sid] = sess
            while len(self._sessions) > config.MAX_SESSIONS:
                oldest = min(self._sessions.values(), key=lambda x: x.touched)
                if oldest.id == sid:
                    break
                self._sessions.pop(oldest.id, None)
                evict.append(oldest)
        for e in evict:
            shutil.rmtree(e.dir, ignore_errors=True)
        return sess

    def get(self, sid: str) -> Optional[Session]:
        with self._lock:
            sess = self._sessions.get(sid)
            if sess:
                sess.touched = time.time()
            return sess

    def delete(self, sid: str) -> None:
        with self._lock:
            sess = self._sessions.pop(sid, None)
        if sess:
            shutil.rmtree(sess.dir, ignore_errors=True)

    # ---- المعاينة ----
    @staticmethod
    def _erode_scale(sess: Session, width_cm: float, height_cm: float) -> float:
        """نسبة تحويل «تصغير الحواف» من دقة الصورة وقت المعالجة الكاملة إلى دقة المعاينة المصغّرة."""
        w0, h0 = sess.width, sess.height
        w_work = w0
        try:
            if width_cm > 0 and height_cm > 0:
                tw, th = cm_to_px(width_cm), cm_to_px(height_cm)
                cw = tw if tw / w0 <= th / h0 else max(1, round(w0 * th / h0))
                if cw < w0:  # المعالجة الكاملة تصغّر الصورة لدقة الطباعة أولاً
                    w_work = cw
        except Exception:  # noqa: BLE001
            pass
        return sess.small_rgb.shape[1] / max(1, w_work)

    def render_preview(
        self, sess: Session, params: Params, width_cm: float = 0.0, height_cm: float = 0.0
    ) -> PreviewResult:
        t0 = time.perf_counter()
        with sess.lock:
            sess.touched = time.time()
            rgb, alpha0 = sess.small_rgb, sess.small_alpha
            key = (sess.transparent, params.remove_bg, params.bg_mode, params.bg_color, params.bg_tolerance)
            hit = sess.cache.get(key)
            if hit is None:
                bg_rgb: Optional[np.ndarray] = None
                manual = False
                if sess.transparent or not params.remove_bg:
                    alpha_raw = alpha0
                else:
                    ai_mask = None
                    if params.bg_mode == "ai":
                        if sess.ai_mask is None:
                            sess.ai_mask = remove_background(rgb)
                        ai_mask = sess.ai_mask
                    alpha_raw, bg_rgb, manual = compute_bg_alpha(rgb, params, ai_mask)
                rgb_def = defringe(rgb, alpha_raw)
                hit = (alpha_raw, rgb_def, bg_rgb, manual)
                while len(sess.cache) >= _CACHE_ENTRIES:
                    sess.cache.pop(next(iter(sess.cache)))
                sess.cache[key] = hit
            alpha_raw, rgb_def, bg_rgb, manual = hit
            if params.white_to_alpha:  # خفيف: لا يدخل في مفتاح التخزين المؤقت
                rgb_def, alpha_raw = white_to_alpha(rgb_def, alpha_raw, params.white_threshold)

            hard = params.alpha_mode == "threshold"
            erode = params.erode_px * self._erode_scale(sess, width_cm, height_cm)
            alpha = cleanup_alpha(alpha_raw, erode, params.alpha_threshold if hard else None)
            if hard:
                alpha = np.where(alpha >= 128, 255, 0).astype(np.uint8)
            out = np.dstack([rgb_def, alpha])
            out[alpha == 0, :3] = 0
            ok, buf = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not ok:
                raise RuntimeError("png encode failed")
        return PreviewResult(
            png=buf.tobytes(),
            bg_color=rgb_to_hex(bg_rgb) if bg_rgb is not None else "",
            bg_manual=manual,
            ms=int((time.perf_counter() - t0) * 1000),
        )

    # ---- تنظيف ----
    def cleanup_expired(self, now: Optional[float] = None) -> int:
        now = now or time.time()
        with self._lock:
            dead = [s for s in self._sessions.values() if now - s.touched > config.SESSION_TTL_SECONDS]
            for s in dead:
                self._sessions.pop(s.id, None)
        for s in dead:
            shutil.rmtree(s.dir, ignore_errors=True)
        # مجلدات جلسات يتيمة (مثلاً فشل الإنشاء بعد الرفع)
        with self._lock:
            known = {f"s-{sid}" for sid in self._sessions}
        for d in config.TMP_ROOT.glob("s-*"):
            if d.is_dir() and d.name not in known and now - d.stat().st_mtime > config.SESSION_TTL_SECONDS:
                shutil.rmtree(d, ignore_errors=True)
        return len(dead)

    def _janitor(self) -> None:
        while True:
            time.sleep(60)
            try:
                self.cleanup_expired()
            except Exception:  # noqa: BLE001
                log.exception("فشل تنظيف الجلسات")


store = SessionStore()
