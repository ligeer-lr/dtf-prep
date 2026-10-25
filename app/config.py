"""إعدادات التطبيق. كل القيم قابلة للتعديل عبر متغيرات البيئة."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# ---- الطباعة ----
PRINT_DPI = 300

# ---- حدود الرفع والإخراج ----
MAX_UPLOAD_MB = _int("DTF_MAX_UPLOAD_MB", 20)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
# أقصى حجم للصورة المرفوعة بالبكسل (حماية الذاكرة)
MAX_INPUT_PIXELS = _int("DTF_MAX_INPUT_MP", 60) * 1_000_000
# أقصى مقاس للإخراج
MIN_SIDE_CM = _float("DTF_MIN_SIDE_CM", 1.0)
MAX_SIDE_CM = _float("DTF_MAX_SIDE_CM", 120.0)
MAX_OUTPUT_PIXELS = _int("DTF_MAX_OUTPUT_MP", 80) * 1_000_000

ALLOWED_FORMATS = {"PNG", "JPEG", "MPO", "WEBP"}

# ---- فحص جودة الصورة ----
# أقل من هذه الدقة الفعلية (DPI عند مقاس الطباعة) تظهر رسالة تنبيه
LOW_DPI_WARNING = _float("DTF_LOW_DPI_WARNING", 100.0)

# ---- إزالة الخلفية ----
# isnet-general-use (الافتراضي، أخف) أو birefnet-general / birefnet-general-lite (أدق وأبطأ)
REMBG_MODEL = os.environ.get("REMBG_MODEL", "isnet-general-use")

# ---- التكبير بـ Real-ESRGAN ----
ESRGAN_TILE = _int("DTF_ESRGAN_TILE", 192)
ESRGAN_PAD = _int("DTF_ESRGAN_PAD", 16)
# الصور الأكبر من هذا (ميجابكسل) تُكبَّر بطريقة Lanczos بدل الذكاء الاصطناعي لأن CPU بطيء عليها
ESRGAN_MAX_INPUT_PIXELS = int(_float("DTF_ESRGAN_MAX_INPUT_MP", 2.0) * 1_000_000)
ESRGAN_MAX_PASSES = _int("DTF_ESRGAN_MAX_PASSES", 2)

# ---- المسارات والملفات المؤقتة ----
MODELS_DIR = Path(os.environ.get("DTF_MODELS_DIR", BASE_DIR / "models"))
TMP_ROOT = Path(os.environ.get("DTF_TMP_DIR", Path(tempfile.gettempdir()) / "dtf-prep"))
JOB_TTL_SECONDS = _int("DTF_JOB_TTL_SECONDS", 30 * 60)
PREVIEW_MAX_SIDE = 1400

# rembg يقرأ هذا المتغير لتحديد مكان تنزيل نماذجه
os.environ.setdefault("U2NET_HOME", str(MODELS_DIR))

# ---- المعاينة المباشرة والجلسات ----
LIVE_PREVIEW_SIDE = _int("DTF_LIVE_PREVIEW_SIDE", 1000)  # أطول ضلع للنسخة المصغّرة
SESSION_TTL_SECONDS = _int("DTF_SESSION_TTL_SECONDS", 60 * 60)
MAX_SESSIONS = _int("DTF_MAX_SESSIONS", 12)
