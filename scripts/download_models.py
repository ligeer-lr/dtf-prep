"""ينزّل نماذج الذكاء الاصطناعي مسبقاً (إزالة الخلفية + Real-ESRGAN) حتى لا يتأخر أول طلب.

الاستخدام:  python scripts/download_models.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(message)s")

from app.esrgan import ensure_onnx_model  # noqa: E402
from app.pipeline import _get_rembg_session  # noqa: E402

if __name__ == "__main__":
    print("→ Real-ESRGAN:", ensure_onnx_model())
    _get_rembg_session()
    print("→ نموذج إزالة الخلفية جاهز")
