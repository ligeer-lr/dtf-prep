"""يولّد صورة تجريبية (شعار على خلفية بيضاء بصيغة JPG) لاختبار الخط الكامل."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


def make_logo(width: int = 640, height: int = 420, background=(255, 255, 255)) -> Image.Image:
    im = Image.new("RGB", (width, height), background)
    d = ImageDraw.Draw(im)
    # دائرة حمراء مع حلقة سوداء
    cx, cy, r = width // 2, height // 2 - 20, min(width, height) // 3
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(200, 20, 40), outline=(10, 10, 10), width=10)
    # نجمة بيضاء داخل الدائرة (تختفي على التيشيرت الأبيض)
    import math

    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = r * (0.62 if i % 2 == 0 else 0.26)
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    d.polygon(pts, fill=(255, 255, 255))
    # نص أسود
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(24, height // 9))
    except OSError:
        font = ImageFont.load_default()
    text = "DTF PREP"
    tb = d.textbbox((0, 0), text, font=font)
    d.text(((width - (tb[2] - tb[0])) // 2, height - (tb[3] - tb[1]) - 30), text, fill=(15, 15, 15), font=font)
    # ظل ناعم خفيف تحت الدائرة لاختبار الشفافية الجزئية
    return im.filter(ImageFilter.GaussianBlur(0.6))


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_logo.jpg")
    make_logo().save(out, "JPEG", quality=82)
    print(out)
