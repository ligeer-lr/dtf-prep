FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DTF_MODELS_DIR=/app/models \
    U2NET_HOME=/app/models \
    DTF_TMP_DIR=/tmp/dtf-prep \
    NUMBA_CACHE_DIR=/tmp/numba \
    OMP_NUM_THREADS=2

WORKDIR /app

# libgomp مطلوبة لـ onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY static ./static
COPY scripts ./scripts

# تنزيل النماذج وقت البناء (فيصير أول طلب سريعاً) ثم التشغيل بمستخدم عادي (مطلوب في Hugging Face Spaces)
RUN python scripts/download_models.py \
    && useradd -m -u 1000 user \
    && mkdir -p /tmp/dtf-prep /tmp/numba \
    && chown -R user:user /app /tmp/dtf-prep /tmp/numba
USER user

# Hugging Face Spaces يستخدم 7860، وRailway يمرّر PORT تلقائياً
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
