"""خط معالجة تصميم DTF: فحص ← إزالة خلفية ← تنظيف حواف ← شفافية جزئية ← تكبير ← تصدير."""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import config
from .esrgan import upscaler

log = logging.getLogger("dtf.pipeline")

Image.MAX_IMAGE_PIXELS = None  # نفحص الحجم بأنفسنا قبل فك الترميز

ProgressCB = Callable[[float, str], None]
_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class UserError(Exception):
    """خطأ يُعرض للمستخدم كما هو (رسالة عربية واضحة)."""


@dataclass
class Params:
    width_cm: float
    height_cm: float
    remove_bg: bool = True
    bg_mode: str = "solid"  # "solid" (خلفية سادة) | "ai" (صور واقعية)
    bg_tolerance: int = 25  # 0..100 — حساسية اللون في وضع الخلفية السادة
    bg_color: str = ""  # "#rrggbb" يدوي، أو فارغ للاكتشاف التلقائي
    erode_px: int = 1  # 0..3
    alpha_mode: str = "threshold"  # "threshold" | "keep" (بدون تغيير)
    alpha_threshold: int = 50  # نسبة مئوية 5..95
    white_to_alpha: bool = False  # الأبيض إلى شفاف (للقمصان الغامقة)
    white_threshold: int = 30  # 0..100 — مدى اعتبار اللون «أبيض»
    ai_strength: int = 60  # 0..100 — نسبة Real-ESRGAN في مزجه مع Lanczos


@dataclass
class Result:
    result_path: Path
    preview_path: Path
    before_path: Path
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# أدوات مساعدة
# ---------------------------------------------------------------------------
def cm_to_px(cm: float) -> int:
    return max(1, int(round(cm / 2.54 * config.PRINT_DPI)))


def validate_params(p: Params) -> tuple[int, int]:
    """يتحقق من المقاس المطلوب ويرجع (العرض، الطول) بالبكسل."""
    for v in (p.width_cm, p.height_cm):
        if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
            raise UserError("المقاس غير صحيح. أدخل العرض والطول بالسنتيمتر كأرقام.")
    if p.width_cm < config.MIN_SIDE_CM or p.height_cm < config.MIN_SIDE_CM:
        raise UserError(f"المقاس صغير جداً. أقل مقاس مسموح {config.MIN_SIDE_CM:g} سم لكل جهة.")
    if p.width_cm > config.MAX_SIDE_CM or p.height_cm > config.MAX_SIDE_CM:
        raise UserError(f"المقاس كبير جداً. أقصى مقاس مسموح {config.MAX_SIDE_CM:g} سم لكل جهة.")
    tw, th = cm_to_px(p.width_cm), cm_to_px(p.height_cm)
    if tw * th > config.MAX_OUTPUT_PIXELS:
        mp = config.MAX_OUTPUT_PIXELS / 1e6
        # أقصى ضلع مربع تقريبي
        side_cm = (config.MAX_OUTPUT_PIXELS ** 0.5) / config.PRINT_DPI * 2.54
        raise UserError(
            f"مقاس الطباعة كبير على الخادم: الملف الناتج سيكون {tw}×{th} بكسل (≈{tw * th / 1e6:.0f} ميجابكسل) "
            f"والحد الأقصى {mp:.0f} ميجابكسل (يعادل تقريباً {side_cm:.0f}×{side_cm:.0f} سم). صغّر المقاس وحاول مجدداً."
        )
    validate_options(p)
    return tw, th


def validate_options(p: Params) -> None:
    """يتحقق من خيارات المعالجة (بدون المقاس). تستخدمها المعاينة المباشرة أيضاً."""
    if p.alpha_mode not in ("threshold", "keep"):
        raise UserError("خيار الشفافية غير صحيح.")
    if p.bg_mode not in ("solid", "ai"):
        raise UserError("وضع إزالة الخلفية غير صحيح.")
    if not 0 <= p.bg_tolerance <= 100:
        raise UserError("قيمة حساسية اللون يجب أن تكون بين 0 و 100.")
    if p.bg_color and not _HEX_RE.match(p.bg_color):
        raise UserError("لون الخلفية غير صحيح. اختر اللون بالضغط على الصورة.")
    if not 0 <= p.erode_px <= 3:
        raise UserError("قيمة تصغير الحواف يجب أن تكون بين 0 و 3 بكسل.")
    if not 5 <= p.alpha_threshold <= 95:
        raise UserError("قيمة حدة الشفافية يجب أن تكون بين 5 و 95.")
    if not 0 <= p.white_threshold <= 100:
        raise UserError("قيمة عتبة الأبيض يجب أن تكون بين 0 و 100.")
    if not 0 <= p.ai_strength <= 100:
        raise UserError("قوة التكبير الذكي يجب أن تكون بين 0 و 100.")


def check_upload_header(path: Path) -> None:
    """فحص سريع للملف المرفوع (الصيغة والمقاس) بدون فك الترميز الكامل."""
    try:
        with Image.open(path) as im:
            fmt = (im.format or "").upper()
            size = im.size
    except Exception as e:  # noqa: BLE001
        raise UserError("الملف ليس صورة صالحة. ارفع صورة PNG أو JPG أو WebP.") from e
    if fmt not in config.ALLOWED_FORMATS:
        raise UserError("صيغة الملف غير مدعومة. الصيغ المسموحة: PNG وJPG وWebP.")
    if size[0] * size[1] > config.MAX_INPUT_PIXELS:
        raise UserError(
            f"الصورة كبيرة جداً ({size[0]}×{size[1]} بكسل). "
            f"الحد الأقصى {config.MAX_INPUT_PIXELS // 1_000_000} ميجابكسل. صغّر الصورة وحاول مجدداً."
        )


def load_image(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """يقرأ الصورة ويرجع (rgb uint8, alpha uint8, معلومات)."""
    try:
        im = Image.open(path)
    except Exception as e:  # noqa: BLE001
        raise UserError("تعذّر قراءة الصورة. تأكد أن الملف PNG أو JPG أو WebP وغير تالف.") from e
    fmt = (im.format or "").upper()
    if fmt not in config.ALLOWED_FORMATS:
        raise UserError("صيغة الملف غير مدعومة. الصيغ المسموحة: PNG وJPG وWebP.")
    if im.width * im.height > config.MAX_INPUT_PIXELS:
        raise UserError(
            f"الصورة كبيرة جداً ({im.width}×{im.height} بكسل). "
            f"الحد الأقصى {config.MAX_INPUT_PIXELS // 1_000_000} ميجابكسل. صغّر الصورة وحاول مجدداً."
        )
    try:
        im = ImageOps.exif_transpose(im)
        had_alpha_channel = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
        im = im.convert("RGBA")
    except Exception as e:  # noqa: BLE001
        raise UserError("تعذّر فك ترميز الصورة. جرّب حفظها من جديد بصيغة PNG.") from e
    arr = np.asarray(im)
    rgb = np.ascontiguousarray(arr[..., :3])
    alpha = np.ascontiguousarray(arr[..., 3])
    # "شفافة فعلاً" = نسبة معتبرة من البكسلات غير معتمة (وليس مجرد قناة ألفا كاملة المعتمة)
    transparent = bool(had_alpha_channel and (alpha < 250).mean() > 0.0005)
    info = {"format": fmt, "width": im.width, "height": im.height, "transparent": transparent}
    return rgb, alpha, info


def resize_rgba(rgba: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """تغيير مقاس RGBA مع ضرب مسبق بالألفا لتفادي الحواف الداكنة/البيضاء."""
    w, h = size
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    premul = np.concatenate([rgba[..., :3].astype(np.float32) * a, a * 255.0], axis=2)
    interp = cv2.INTER_AREA if (w < rgba.shape[1] or h < rgba.shape[0]) else cv2.INTER_CUBIC
    premul = cv2.resize(premul, (w, h), interpolation=interp)
    a2 = premul[..., 3:4] / 255.0
    rgb = np.where(a2 > 1e-6, premul[..., :3] / np.maximum(a2, 1e-6), 0)
    out = np.concatenate([rgb, premul[..., 3:4]], axis=2)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def save_preview(rgba: np.ndarray, path: Path, as_jpeg: bool = False) -> None:
    h, w = rgba.shape[:2]
    m = max(h, w)
    if m > config.PREVIEW_MAX_SIDE:
        s = config.PREVIEW_MAX_SIDE / m
        rgba = resize_rgba(rgba, (max(1, round(w * s)), max(1, round(h * s))))
    if as_jpeg:
        Image.fromarray(rgba[..., :3], "RGB").save(path, "JPEG", quality=88)
    else:
        Image.fromarray(rgba, "RGBA").save(path, "PNG", compress_level=3)


# ---------------------------------------------------------------------------
# 2) إزالة الخلفية
# ---------------------------------------------------------------------------
_rembg_session = None
_rembg_lock = threading.Lock()


def _get_rembg_session():
    global _rembg_session
    with _rembg_lock:
        if _rembg_session is None:
            from rembg import new_session

            log.info("تحميل نموذج إزالة الخلفية: %s", config.REMBG_MODEL)
            _rembg_session = new_session(config.REMBG_MODEL, providers=["CPUExecutionProvider"])
        return _rembg_session


def remove_background(rgb: np.ndarray) -> np.ndarray:
    """يرجع قناع الألفا (uint8). نأخذ القناع فقط ونبقي ألوان الصورة الأصلية كما هي."""
    from rembg import remove

    session = _get_rembg_session()
    mask = remove(Image.fromarray(rgb, "RGB"), session=session, only_mask=True, post_process_mask=False)
    mask = np.asarray(mask.convert("L") if isinstance(mask, Image.Image) else mask, dtype=np.uint8)
    return np.ascontiguousarray(mask)


# ---------------------------------------------------------------------------
# 2-ب) إزالة الخلفية السادة (flood fill من الحواف)
# ---------------------------------------------------------------------------
def hex_to_rgb(c: str) -> np.ndarray:
    return np.array([int(c[i: i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)


def rgb_to_hex(c) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(v) for v in c))


def _lab_distance(rgb: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """المسافة اللونية (ΔE في فضاء Lab) بين كل بكسل ولون الخلفية، على شرائح لتوفير الذاكرة."""
    bg_lab = cv2.cvtColor(bg_rgb.reshape(1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab).reshape(3)
    h = rgb.shape[0]
    out = np.empty(rgb.shape[:2], dtype=np.float32)
    step = 1024
    for y0 in range(0, h, step):
        lab = cv2.cvtColor(rgb[y0: y0 + step].astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
        out[y0: y0 + step] = np.sqrt(((lab - bg_lab) ** 2).sum(axis=2))
    return out


def _border_pixels(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    t = max(1, int(round(min(h, w) * 0.02)))
    return np.concatenate(
        [rgb[:t].reshape(-1, 3), rgb[-t:].reshape(-1, 3), rgb[:, :t].reshape(-1, 3), rgb[:, -t:].reshape(-1, 3)]
    )


def detect_bg_color(rgb: np.ndarray) -> np.ndarray:
    """أكثر لون شيوعاً على حواف الصورة وزواياها."""
    border = _border_pixels(rgb)
    q = (border >> 4).astype(np.int32)
    key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    best = np.bincount(key).argmax()
    return np.median(border[key == best], axis=0).astype(np.uint8)


def remove_solid_background(
    rgb: np.ndarray, bg_color: str, tolerance: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    """يزيل خلفية سادة بـ flood fill يبدأ من حواف الصورة فقط.

    أي منطقة بنفس لون الخلفية لكنها محاطة بالتصميم (نجمة بيضاء، نص، ثقب حرف O…) لا تُمسّ لأنها
    غير متصلة بالحافة. بعدها تُحسب ألفا الحافة بفصل لون الخلفية عن لون التصميم (بدل قطع حاد)
    حتى لا تؤكل التفاصيل الدقيقة.

    يرجع (ألفا، لون الخلفية المستخدم، هل اللون يدوي).
    """
    manual = bool(bg_color)
    bg = hex_to_rgb(bg_color) if manual else detect_bg_color(rgb)
    thr = max(1.0, tolerance * 0.6)  # ΔE: 0..60
    dist = _lab_distance(rgb, bg)
    cand = (dist <= thr).astype(np.uint8)

    border = np.concatenate([cand[0], cand[-1], cand[:, 0], cand[:, -1]])
    if border.mean() < 0.30:
        raise UserError(
            "لم نجد خلفية سادة واضحة على حواف الصورة. اضغط على الصورة لاختيار لون الخلفية بنفسك، "
            "أو ارفع «حساسية اللون»، أو اختر وضع «صورة واقعية (ذكاء اصطناعي)»."
        )

    # flood fill من الحواف = المكوّنات المتصلة (اتصال رباعي) التي تلمس حدود الصورة
    n, labels = cv2.connectedComponents(cand, connectivity=4)
    edge_labels = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    lut = np.zeros(n, dtype=bool)
    lut[edge_labels] = True
    lut[0] = False  # 0 = ليست خلفية مرشحة
    bg_mask = lut[labels]
    del labels

    # ألفا أساسية: التصميم معتم، والخلفية شفافة
    alpha = np.where(bg_mask, 0, 255).astype(np.uint8)

    # حافة التصميم: نحسب نسبة المزج بين لون الخلفية ولون التصميم القريب (بدل قطع حاد)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    near_bg = cv2.dilate(bg_mask.astype(np.uint8), k, iterations=2) > 0
    band = near_bg & ~bg_mask
    if band.any():
        core = (~near_bg).astype(np.uint8) * 255
        fg_est = defringe(rgb, core) if core.any() else rgb
        ys, xs = np.nonzero(band)
        p = rgb[ys, xs].astype(np.float32)
        f = fg_est[ys, xs].astype(np.float32)
        b = bg.astype(np.float32)
        d = f - b
        denom = (d * d).sum(axis=1)
        a = np.where(denom > 36.0, ((p - b) * d).sum(axis=1) / np.maximum(denom, 1e-6), 1.0)
        a = np.clip(a, 0.0, 1.0)
        a = np.where(a < 0.04, 0.0, a)  # ضجيج JPEG بلون الخلفية
        a = np.where(a > 0.97, 1.0, a)
        alpha[ys, xs] = np.round(a * 255).astype(np.uint8)
    return alpha, bg, manual


# ---------------------------------------------------------------------------
# 3) تنظيف الحواف
# ---------------------------------------------------------------------------
def _push_pull(rgb_f: np.ndarray, w: np.ndarray) -> np.ndarray:
    """ملء الألوان المفقودة (w=0) من أقرب بكسلات معروفة (w=1) بهرم تنعيم مُطبَّع."""
    num = rgb_f * w[..., None]
    den = w.copy()
    levels = [(num, den)]
    while min(num.shape[:2]) > 2 and len(levels) < 16:
        nh, nw = max(1, num.shape[0] // 2), max(1, num.shape[1] // 2)
        num = cv2.resize(num, (nw, nh), interpolation=cv2.INTER_AREA)
        den = cv2.resize(den, (nw, nh), interpolation=cv2.INTER_AREA)
        levels.append((num, den))
    n, d = levels[-1]
    color = n / np.maximum(d, 1e-8)[..., None]
    for n, d in reversed(levels[:-1]):
        up = cv2.resize(color, (n.shape[1], n.shape[0]), interpolation=cv2.INTER_LINEAR)
        has = (d > 1e-6)[..., None]
        color = np.where(has, n / np.maximum(d, 1e-8)[..., None], up)
    return color


def defringe(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """إزالة الهالة (color decontamination).

    البكسلات شبه الشفافة على حافة التصميم تحمل غالباً لون الخلفية القديمة (هالة بيضاء).
    نستبدل لونها بلون أقرب بكسلات معتمة من داخل التصميم. كذلك نملأ البكسلات الشفافة
    تماماً بألوان الحواف حتى لا تتسرب ألوان عشوائية للحواف أثناء التكبير.
    """
    h, w = alpha.shape
    solid = alpha >= 250
    if not solid.any():
        solid = alpha >= 128
        if not solid.any():
            return rgb
    band_r = int(np.clip(round(max(h, w) / 600), 2, 6))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_r + 1, 2 * band_r + 1))
    near = cv2.dilate(solid.astype(np.uint8), k) > 0
    replace = (~solid) & ((alpha == 0) | ((alpha < 250) & near))
    if not replace.any():
        return rgb

    out = rgb.copy()
    margin = 48
    strip = 1024
    wf = solid.astype(np.float32)
    for y0 in range(0, h, strip):
        y1 = min(h, y0 + strip)
        ya, yb = max(0, y0 - margin), min(h, y1 + margin)
        sub_rep = replace[y0:y1]
        if not sub_rep.any():
            continue
        filled = _push_pull(rgb[ya:yb].astype(np.float32), wf[ya:yb])[y0 - ya: y1 - ya]
        sel = sub_rep
        out[y0:y1][sel] = np.clip(filled[sel] + 0.5, 0, 255).astype(np.uint8)
    return out


def erode_alpha(alpha: np.ndarray, px: float) -> np.ndarray:
    """تصغير الحواف. القيم الصحيحة (المعالجة الكاملة) تعمل كما كانت؛ والكسرية تُستخدم في المعاينة
    المصغّرة لتمثيل نفس النسبة من الصورة (مثلاً 1 بكسل من 3000 = 0.33 بكسل من 1000)."""
    if px <= 1e-3:
        return alpha
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    whole = int(px)
    frac = px - whole
    out = cv2.erode(alpha, k, iterations=whole) if whole else alpha
    if frac > 1e-3:
        nxt = cv2.erode(out, k)
        out = np.rint(out.astype(np.float32) * (1.0 - frac) + nxt.astype(np.float32) * frac).astype(np.uint8)
    return out


def smooth_alpha(alpha: np.ndarray) -> np.ndarray:
    """تنعيم حواف الألفا الخشنة مع حماية الخطوط الرفيعة من الاختفاء."""
    h, w = alpha.shape
    sigma = float(np.clip(max(h, w) / 2500.0, 0.7, 1.6))
    binary = alpha >= 128
    b8 = binary.astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    band = (cv2.dilate(b8, k3, iterations=2) - cv2.erode(b8, k3, iterations=2)) > 0
    if not band.any():
        return alpha
    a = alpha.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(a, (0, 0), sigma)
    sharpened = np.clip((blurred - 0.5) * 2.0 + 0.5, 0.0, 1.0)
    new = np.where(band, sharpened * 255.0, alpha.astype(np.float32))
    # حماية العناصر الرفيعة (أقل من 3 بكسل): ما يزول بالـ opening هو عنصر رفيع
    thin = binary & (cv2.morphologyEx(b8, cv2.MORPH_OPEN, k3) == 0)
    new = np.where(thin, np.maximum(new, alpha.astype(np.float32)), new)
    return np.clip(new + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4) الشفافية الجزئية
# ---------------------------------------------------------------------------
_RAMP = 0.12  # نصف عرض منطقة الانتقال حول العتبة


def sharpen_alpha(alpha: np.ndarray, threshold_pct: int) -> np.ndarray:
    """يدفع شبه الشفاف نحو شفاف/معتم حول العتبة، مع إبقاء نصف بكسل انتقال يحفظ موضع الحافة بدقة.

    القطع النهائي الحاد (0/255) يتم على الألفا بعد التكبير لتخرج حافة ناعمة المحيط بلا تسنن.
    """
    t = threshold_pct / 100.0
    x = alpha.astype(np.float32) / 255.0
    y = np.clip(0.5 + (x - t) / (2.0 * _RAMP), 0.0, 1.0)
    return np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# مراحل مشتركة بين المعالجة الكاملة والمعاينة المباشرة
# ---------------------------------------------------------------------------
def compute_bg_alpha(
    rgb: np.ndarray, params: Params, ai_mask: Optional[np.ndarray] = None
) -> tuple[np.ndarray, Optional[np.ndarray], bool]:
    """مرحلة إزالة الخلفية. ترجع (ألفا، لون الخلفية للوضع السادة وإلا None، هل اللون يدوي).

    ai_mask: قناع جاهز من الذكاء الاصطناعي (تستخدمه المعاينة لتفادي إعادة الحساب).
    """
    bg_rgb: Optional[np.ndarray] = None
    manual = False
    if params.bg_mode == "solid":
        alpha, bg_rgb, manual = remove_solid_background(rgb, params.bg_color, params.bg_tolerance)
        hint = "خفّض «حساسية اللون» أو اختر وضع «صورة واقعية»."
    else:
        alpha = ai_mask if ai_mask is not None else remove_background(rgb)
        hint = "جرّب صورة أوضح، أو أوقف خيار «إزالة الخلفية» إذا كانت الصورة جاهزة."
    coverage = float((alpha >= 128).mean())
    if alpha.max() < 20 or coverage < 0.0005:
        raise UserError("لم نستطع تحديد التصميم في الصورة بعد إزالة الخلفية. " + hint)
    return alpha, bg_rgb, manual


def white_to_alpha(
    rgb: np.ndarray, alpha: np.ndarray, threshold: int
) -> tuple[np.ndarray, np.ndarray]:
    """«الأبيض إلى شفاف» (color to alpha): يحوّل القريب من الأبيض إلى شفاف بتدرج ناعم.

    المقياس هو المسافة اللونية (Lab ΔE) عن الأبيض النقي، فالكريمي والبيج (ΔE أكبر من ~12)
    لا يُعتبران أبيض عند القيم المعتادة. يشمل كل البكسلات (حتى البقع البيضاء داخل التصميم).
    اللون يُفكّ مزجه عن الأبيض (c = a·f + (1−a)·255) فلا تبقى هالة بيضاء على الحواف.
    يرجع (rgb جديد، alpha جديد).
    """
    hi = 4.0 + 0.10 * threshold  # ΔE الذي يصبح عنده اللون معتماً كلياً
    lo = hi * 0.4  # ΔE الذي يصبح عنده شفافاً كلياً
    d = _lab_distance(rgb, np.array([255, 255, 255], dtype=np.uint8))
    a = np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    a_safe = np.maximum(a, 0.02)[..., None]
    f = (rgb.astype(np.float32) - 255.0 * (1.0 - a[..., None])) / a_safe
    new_rgb = np.clip(f, 0, 255).astype(np.uint8)
    new_alpha = np.clip(alpha.astype(np.float32) * a, 0, 255).round().astype(np.uint8)
    # املأ المناطق الشفافة الجديدة بألوان الجوار (حتى لا ينزف الأبيض/الأسود عند التكبير)
    return defringe(new_rgb, new_alpha), new_alpha


def cleanup_alpha(alpha: np.ndarray, erode_px: float, hard_threshold: Optional[int]) -> np.ndarray:
    """تصغير الحواف ← تنعيم ← (اختيارياً) دفع الشفافية الجزئية نحو شفاف/معتم حول العتبة."""
    alpha = erode_alpha(alpha, erode_px)
    alpha = smooth_alpha(alpha)
    if hard_threshold is not None:
        alpha = sharpen_alpha(alpha, hard_threshold)
    return alpha


# ---------------------------------------------------------------------------
# الخط الكامل
# ---------------------------------------------------------------------------
def process_image(
    src: Path,
    out_dir: Path,
    params: Params,
    progress: ProgressCB,
    warnings: Optional[list[str]] = None,
    notes: Optional[list[str]] = None,
) -> Result:
    """warnings/notes اختياريان: يمكن تمرير قوائم مشتركة ليراها المستخدم مباشرة أثناء المعالجة."""
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings = [] if warnings is None else warnings
    notes = [] if notes is None else notes

    # ---- 1) فحص الصورة ----
    progress(0.01, "فحص الصورة…")
    tw, th = validate_params(params)
    rgb, alpha, meta = load_image(src)
    h, w = alpha.shape
    transparent = meta["transparent"]

    before_path = out_dir / "before.png"
    if transparent:
        save_preview(np.dstack([rgb, alpha]), before_path)
    else:
        before_path = out_dir / "before.jpg"
        save_preview(np.dstack([rgb, np.full_like(alpha, 255)]), before_path, as_jpeg=True)

    # مقاس المحتوى داخل لوحة الطباعة (نحافظ على النسبة بدون تشويه)
    if tw / w <= th / h:
        cw, ch = tw, min(th, max(1, round(h * tw / w)))
    else:
        cw, ch = min(tw, max(1, round(w * th / h))), th
    scale = cw / w
    eff_dpi = config.PRINT_DPI / scale

    if eff_dpi < config.LOW_DPI_WARNING:
        warnings.append(
            f"الصورة ضعيفة جداً مقارنة بمقاس الطباعة: دقتها الفعلية عند {params.width_cm:g}×{params.height_cm:g} سم "
            f"حوالي {eff_dpi:.0f} DPI والمطلوب {config.PRINT_DPI}. سيتم تكبيرها حوالي {scale:.1f} مرة، "
            f"وقد تظهر التفاصيل ناعمة أو غير حادة. الأفضل رفع صورة أكبر أو تقليل مقاس الطباعة."
        )
    if abs((w / h) / (tw / th) - 1.0) > 0.02:
        notes.append(
            f"نسبة أبعاد الصورة ({w}×{h}) تختلف عن المقاس المطلوب، فحفظنا النسبة بدون تشويه "
            f"وأضفنا مساحة شفافة حول التصميم ليكون الملف بمقاس {tw}×{th} بكسل بالضبط."
        )

    # صورة أكبر من المطلوب: نصغّرها مبكراً (أسرع وأقل ذاكرة، ولا نخسر شيئاً من دقة الطباعة)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (cw, ch), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (cw, ch), interpolation=cv2.INTER_AREA)
        notes.append("الصورة أكبر من المطلوب لهذا المقاس، فتم تصغيرها إلى دقة الطباعة (300 DPI).")
        h, w = ch, cw
        scale = 1.0

    # ---- 2) إزالة الخلفية ----
    if transparent:
        notes.append("الصورة شفافة أصلاً، فتم تجاوز إزالة الخلفية.")
        progress(0.22, "الصورة شفافة أصلاً — تجاوز إزالة الخلفية")
    elif params.remove_bg:
        if params.bg_mode == "solid":
            progress(0.05, "إزالة الخلفية السادة…")
        else:
            progress(0.05, "إزالة الخلفية بالذكاء الاصطناعي… (قد تأخذ نصف دقيقة عند أول تشغيل)")
        alpha, bg_rgb, manual = compute_bg_alpha(rgb, params)
        if bg_rgb is not None:
            notes.append(f"لون الخلفية {'المحدد يدوياً' if manual else 'المكتشف تلقائياً'}: {rgb_to_hex(bg_rgb)}")
            if (alpha == 0).mean() < 0.001:
                notes.append("لم تُزَل خلفية تُذكر. جرّب رفع «حساسية اللون» أو اختيار لون الخلفية يدوياً.")
        progress(0.22, "تمت إزالة الخلفية")
    else:
        progress(0.22, "تم تخطي إزالة الخلفية")

    # ---- 3) تنظيف الحواف  +  4) الشفافية الجزئية ----
    progress(0.24, "إزالة الهالة حول التصميم…")
    rgb = defringe(rgb, alpha)
    if params.white_to_alpha:
        progress(0.27, "تحويل الأبيض إلى شفاف…")
        rgb, alpha = white_to_alpha(rgb, alpha, params.white_threshold)
        if (alpha > 0).sum() < 0.0005 * alpha.size:
            raise UserError("لم يبقَ شيء من التصميم بعد تحويل الأبيض إلى شفاف. خفّض عتبة الأبيض أو أوقف الخيار.")
    progress(0.30, "تصغير الحواف وتنعيمها ومعالجة الشفافية الجزئية…")
    hard = params.alpha_mode == "threshold"
    alpha = cleanup_alpha(alpha, params.erode_px, params.alpha_threshold if hard else None)
    progress(0.34, "معالجة الشفافية الجزئية…")

    # ---- 5) التكبير لدقة الطباعة ----
    cur = rgb
    cur_alpha = alpha
    remaining = cw / w
    passes = 0
    ai_used = False
    strength = params.ai_strength / 100.0
    if remaining > 1.0 + 1e-6:
        if strength <= 0:
            notes.append("قوة التكبير الذكي 0، فاستُخدم تكبير Lanczos فقط.")
        elif w * h > config.ESRGAN_MAX_INPUT_PIXELS:
            notes.append("الصورة عالية الدقة أصلاً، فاستُخدم تكبير Lanczos بدل الذكاء الاصطناعي لتسريع المعالجة.")
        else:
            while (
                remaining > (1.05 if passes == 0 else 1.6)
                and passes < config.ESRGAN_MAX_PASSES
                and cur.shape[0] * cur.shape[1] <= config.ESRGAN_MAX_INPUT_PIXELS * (1 if passes == 0 else 2)
            ):
                lo = 0.38 + 0.54 * (passes / config.ESRGAN_MAX_PASSES)
                hi = 0.38 + 0.54 * ((passes + 1) / config.ESRGAN_MAX_PASSES)
                label = "التكبير بالذكاء الاصطناعي (Real-ESRGAN)"
                if passes:
                    label += f" — المرحلة {passes + 1}"
                progress(lo, label + "…")
                ai = upscaler.upscale4x(
                    cur,
                    cur_alpha,
                    lambda f, lo=lo, hi=hi, label=label: progress(lo + (hi - lo) * f, f"{label}… {int(f * 100)}%"),
                )
                if strength < 1.0:
                    # مزج مع Lanczos يخفف أثر التكبير الذكي (مثل تشوّه نقاط الهاف تون)
                    lz = cv2.resize(cur, (ai.shape[1], ai.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    ai = cv2.addWeighted(ai, strength, lz, 1.0 - strength, 0)
                cur = ai
                cur_alpha = cv2.resize(cur_alpha, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_LINEAR)
                remaining /= 4.0
                passes += 1
                ai_used = True
    progress(0.93, "ضبط المقاس النهائي…")

    # المقاس النهائي بالضبط (الألوان من آخر نتيجة، والألفا من المصدر مباشرة لأدق حافة)
    interp_rgb = cv2.INTER_AREA if cur.shape[1] > cw else cv2.INTER_LANCZOS4
    final_rgb = cv2.resize(cur, (cw, ch), interpolation=interp_rgb) if cur.shape[:2] != (ch, cw) else cur
    interp_a = cv2.INTER_AREA if alpha.shape[1] > cw else cv2.INTER_CUBIC
    final_alpha = cv2.resize(alpha, (cw, ch), interpolation=interp_a) if alpha.shape != (ch, cw) else alpha
    if hard:
        final_alpha = np.where(final_alpha >= 128, 255, 0).astype(np.uint8)
    final_rgb = final_rgb.copy()
    final_rgb[final_alpha == 0] = 0  # لون مخفي تحت الشفافية الكاملة: نظّفه (يصغّر الملف)

    # ---- 6) التصدير ----
    progress(0.95, "تصدير الملف…")
    canvas = np.zeros((th, tw, 4), dtype=np.uint8)
    ox, oy = (tw - cw) // 2, (th - ch) // 2
    canvas[oy: oy + ch, ox: ox + cw, :3] = final_rgb
    canvas[oy: oy + ch, ox: ox + cw, 3] = final_alpha

    result_path = out_dir / "result.png"
    Image.fromarray(canvas, "RGBA").save(
        result_path, "PNG", dpi=(config.PRINT_DPI, config.PRINT_DPI), compress_level=4
    )
    preview_path = out_dir / "preview.png"
    save_preview(canvas, preview_path)
    progress(1.0, "اكتملت المعالجة")

    return Result(
        result_path=result_path,
        preview_path=preview_path,
        before_path=before_path,
        warnings=warnings,
        notes=notes,
        info={
            "input_px": [meta["width"], meta["height"]],
            "output_px": [tw, th],
            "output_cm": [params.width_cm, params.height_cm],
            "dpi": config.PRINT_DPI,
            "effective_dpi": round(eff_dpi),
            "ai_upscale": ai_used,
            "ai_passes": passes,
            "file_bytes": result_path.stat().st_size,
        },
    )
