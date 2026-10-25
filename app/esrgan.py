"""تكبير الصور بـ Real-ESRGAN (نموذج realesr-general-x4v3) على CPU عبر ONNX Runtime.

لماذا ONNX؟ لأن النموذج صغير (≈5MB) وبنيته بسيطة (SRVGGNetCompact)، فنبني منه
رسماً بيانياً ONNX مباشرة من أوزان .pth الرسمية، ونشغّله بـ onnxruntime بدون الحاجة
إلى PyTorch وقت التشغيل (صورة Docker أخف بكثير وتشغيل أسرع على CPU).
"""

from __future__ import annotations

import collections
import io
import logging
import os
import pickle
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import config

log = logging.getLogger("dtf.esrgan")

WEIGHTS_NAME = "realesr-general-x4v3.pth"
ONNX_NAME = "realesr-general-x4v3.onnx"
WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/" + WEIGHTS_NAME
)
SCALE = 4
NUM_CONV = 32  # عدد طبقات الـ conv المتوسطة في SRVGGNetCompact


# --------------------------------------------------------------------------
# قراءة أوزان .pth بدون PyTorch
# --------------------------------------------------------------------------
class _Unpickler(pickle.Unpickler):
    """قارئ pickle آمن يستخرج الـ tensors كمصفوفات numpy فقط (لا ينفّذ أي كود)."""

    def __init__(self, fp, zf: zipfile.ZipFile, prefix: str):
        super().__init__(fp)
        self._zf = zf
        self._prefix = prefix

    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2":
            return self._rebuild_tensor
        if name == "OrderedDict":
            return collections.OrderedDict
        if name.endswith("Storage"):
            return name  # اسم النوع فقط
        raise pickle.UnpicklingError(f"غير مسموح: {module}.{name}")

    def persistent_load(self, pid):
        # pid = ('storage', storage_type, key, location, numel)
        return pid

    def _rebuild_tensor(self, storage, offset, size, stride, *_args, **_kwargs):
        _tag, stype, key, _loc, numel = storage
        dtype = {"FloatStorage": np.float32, "HalfStorage": np.float16}.get(stype)
        if dtype is None:
            raise pickle.UnpicklingError(f"نوع غير مدعوم: {stype}")
        raw = self._zf.read(f"{self._prefix}data/{key}")
        flat = np.frombuffer(raw, dtype=dtype, count=numel)
        n = int(np.prod(size)) if len(size) else 1
        flat = flat[offset: offset + n]
        return flat.reshape(tuple(size)).astype(np.float32)


def load_pth_state_dict(path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl[: -len("data.pkl")]
        obj = _Unpickler(io.BytesIO(zf.read(pkl)), zf, prefix).load()
    return dict(obj.get("params", obj))


# --------------------------------------------------------------------------
# بناء رسم ONNX لشبكة SRVGGNetCompact
# --------------------------------------------------------------------------
def build_onnx(state: dict[str, np.ndarray]) -> bytes:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes, inits = [], []

    def add_init(name, arr):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.float32), name))

    prev = "input"
    for i in range(NUM_CONV + 2):  # body.0 .. body.65 = conv+prelu، ثم body.66 = conv الأخير
        w, b = state[f"body.{2 * i}.weight"], state[f"body.{2 * i}.bias"]
        add_init(f"w{i}", w)
        add_init(f"b{i}", b)
        conv_out = f"conv{i}"
        nodes.append(
            helper.make_node(
                "Conv", [prev, f"w{i}", f"b{i}"], [conv_out],
                kernel_shape=[3, 3], pads=[1, 1, 1, 1], name=f"Conv_{i}",
            )
        )
        if i == NUM_CONV + 1:
            prev = conv_out
            break
        slope = state[f"body.{2 * i + 1}.weight"].reshape(-1, 1, 1)
        add_init(f"s{i}", slope)
        act_out = f"act{i}"
        nodes.append(helper.make_node("PRelu", [conv_out, f"s{i}"], [act_out], name=f"PRelu_{i}"))
        prev = act_out

    nodes.append(
        helper.make_node("DepthToSpace", [prev], ["shuffled"], blocksize=SCALE, mode="CRD", name="PixelShuffle")
    )
    # الأساس: تكبير nearest للصورة الأصلية (residual) كما في النموذج الرسمي
    inits.append(numpy_helper.from_array(np.array([1, 1, SCALE, SCALE], dtype=np.float32), "scales"))
    inits.append(numpy_helper.from_array(np.array([], dtype=np.float32), "roi"))
    nodes.append(
        helper.make_node(
            "Resize", ["input", "roi", "scales"], ["base"],
            mode="nearest", coordinate_transformation_mode="asymmetric",
            nearest_mode="floor", name="NearestUp",
        )
    )
    nodes.append(helper.make_node("Add", ["shuffled", "base"], ["output"], name="Residual"))

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "H", "W"])
    graph = helper.make_graph(nodes, "realesr_general_x4v3", [inp], [out], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


# --------------------------------------------------------------------------
# تنزيل الأوزان وتجهيز النموذج
# --------------------------------------------------------------------------
def _download(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            log.info("تنزيل %s (محاولة %d)", url, attempt)
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size < 1_000_000:
                raise RuntimeError("الملف المنزَّل أصغر من المتوقع")
            os.replace(tmp, dest)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * attempt)
    raise RuntimeError(f"تعذّر تنزيل أوزان Real-ESRGAN: {last}")


def ensure_onnx_model(models_dir: Path | None = None) -> Path:
    """يضمن وجود ملف ONNX جاهز (ينزّل الأوزان ويحوّلها عند أول تشغيل)."""
    models_dir = models_dir or config.MODELS_DIR
    onnx_path = models_dir / ONNX_NAME
    if onnx_path.exists() and onnx_path.stat().st_size > 1_000_000:
        return onnx_path
    pth = models_dir / WEIGHTS_NAME
    if not pth.exists():
        _download(WEIGHTS_URL, pth)
    log.info("بناء نموذج ONNX من الأوزان…")
    data = build_onnx(load_pth_state_dict(pth))
    models_dir.mkdir(parents=True, exist_ok=True)
    tmp = onnx_path.with_suffix(".part")
    tmp.write_bytes(data)
    os.replace(tmp, onnx_path)
    return onnx_path


class Upscaler:
    """يكبّر صورة RGB بمعامل ×4 على أجزاء (tiles) لتجنب استهلاك الذاكرة."""

    def __init__(self) -> None:
        self._session = None
        self._lock = threading.Lock()

    def _get_session(self):
        with self._lock:
            if self._session is None:
                import onnxruntime as ort

                path = ensure_onnx_model()
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = os.cpu_count() or 2
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._session = ort.InferenceSession(
                    str(path), sess_options=opts, providers=["CPUExecutionProvider"]
                )
            return self._session

    def upscale4x(
        self,
        rgb: np.ndarray,
        alpha: Optional[np.ndarray] = None,
        progress: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """rgb: uint8 HxWx3 → uint8 (4H)x(4W)x3.

        alpha (اختياري): الأجزاء الشفافة بالكامل (وما حولها) تُتخطّى لتوفير الوقت.
        """
        sess = self._get_session()
        h, w, _ = rgb.shape
        tile, pad = config.ESRGAN_TILE, config.ESRGAN_PAD
        padded = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="reflect" if min(h, w) > pad else "edge")
        out = np.zeros((h * SCALE, w * SCALE, 3), dtype=np.uint8)

        ys = list(range(0, h, tile))
        xs = list(range(0, w, tile))
        total = len(ys) * len(xs)
        done = 0
        for y0 in ys:
            y1 = min(y0 + tile, h)
            for x0 in xs:
                x1 = min(x0 + tile, w)
                done += 1
                if alpha is not None:
                    ya, yb = max(0, y0 - pad), min(h, y1 + pad)
                    xa, xb = max(0, x0 - pad), min(w, x1 + pad)
                    if not alpha[ya:yb, xa:xb].any():
                        if progress:
                            progress(done / total)
                        continue
                patch = padded[y0: y1 + 2 * pad, x0: x1 + 2 * pad]
                inp = (patch.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
                res = sess.run(None, {"input": inp})[0][0]
                res = np.clip(res * 255.0 + 0.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)
                p4 = pad * SCALE
                res = res[p4: p4 + (y1 - y0) * SCALE, p4: p4 + (x1 - x0) * SCALE]
                out[y0 * SCALE: y1 * SCALE, x0 * SCALE: x1 * SCALE] = res
                if progress:
                    progress(done / total)
        return out


upscaler = Upscaler()
