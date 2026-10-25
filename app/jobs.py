"""إدارة المهام: تنفيذ خلفي، تتبّع التقدم، وحذف الملفات المؤقتة تلقائياً."""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config
from .pipeline import Params, Result, UserError, process_image

log = logging.getLogger("dtf.jobs")

GENERIC_ERROR = "حدث خطأ غير متوقع أثناء المعالجة. حاول مرة أخرى، وإذا تكرر جرّب صورة أو مقاساً مختلفاً."


@dataclass
class Job:
    id: str
    dir: Path
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    stage: str = "في الانتظار…"
    error: Optional[str] = None
    result: Optional[Result] = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)

    def public(self, queue_pos: int = 0) -> dict:
        d = {
            "id": self.id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "stage": self.stage,
            "error": self.error,
            "queue_position": queue_pos,
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "info": None,
        }
        if self.result:
            d["info"] = self.result.info
        return d


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        # معالجة واحدة في كل مرة: العملية ثقيلة على CPU والذاكرة
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dtf-worker")
        self._order: list[str] = []
        config.TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self._wipe_root()
        threading.Thread(target=self._janitor, name="dtf-janitor", daemon=True).start()

    # ---- إنشاء / استعلام ----
    def new_job_dir(self) -> tuple[str, Path]:
        jid = uuid.uuid4().hex[:16]
        d = config.TMP_ROOT / jid
        d.mkdir(parents=True, exist_ok=True)
        return jid, d

    def submit(self, jid: str, job_dir: Path, upload: Path, params: Params) -> Job:
        job = Job(id=jid, dir=job_dir)
        with self._lock:
            self._jobs[jid] = job
            self._order.append(jid)
        self._pool.submit(self._run, job, upload, params)
        return job

    def get(self, jid: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(jid)
            if job:
                job.touched = time.time()
            return job

    def queue_position(self, jid: str) -> int:
        with self._lock:
            waiting = [j for j in self._order if self._jobs.get(j) and self._jobs[j].status == "queued"]
        return waiting.index(jid) + 1 if jid in waiting else 0

    def delete(self, jid: str) -> bool:
        with self._lock:
            job = self._jobs.get(jid)
            if not job or job.status in ("queued", "running"):
                return False
            self._jobs.pop(jid, None)
            if jid in self._order:
                self._order.remove(jid)
        shutil.rmtree(job.dir, ignore_errors=True)
        return True

    # ---- التنفيذ ----
    def _run(self, job: Job, upload: Path, params: Params) -> None:
        job.status = "running"
        job.stage = "بدء المعالجة…"

        def cb(frac: float, stage: str) -> None:
            job.progress = max(job.progress, min(1.0, frac))
            job.stage = stage
            job.touched = time.time()

        try:
            job.result = process_image(upload, job.dir, params, cb, job.warnings, job.notes)
            job.progress, job.stage, job.status = 1.0, "اكتملت المعالجة", "done"
        except UserError as e:
            job.status, job.error = "error", str(e)
        except MemoryError:
            job.status = "error"
            job.error = "نفدت الذاكرة أثناء المعالجة. صغّر مقاس الطباعة أو ارفع صورة أصغر وحاول مجدداً."
        except Exception:  # noqa: BLE001
            log.exception("فشلت المعالجة للمهمة %s", job.id)
            job.status, job.error = "error", GENERIC_ERROR
        finally:
            job.touched = time.time()
            # الملف المرفوع الأصلي لم نعد بحاجة إليه
            try:
                upload.unlink(missing_ok=True)
            except OSError:
                pass
            if job.status == "error":
                # لا نحتفظ بأي ملفات ناتجة عن مهمة فاشلة
                for p in job.dir.glob("*"):
                    p.unlink(missing_ok=True)

    # ---- التنظيف ----
    def _wipe_root(self) -> None:
        for p in config.TMP_ROOT.glob("*"):
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)

    def _janitor(self) -> None:
        while True:
            time.sleep(60)
            try:
                self.cleanup_expired()
            except Exception:  # noqa: BLE001
                log.exception("فشل تنظيف الملفات المؤقتة")

    def cleanup_expired(self, now: Optional[float] = None) -> int:
        now = now or time.time()
        expired: list[Job] = []
        with self._lock:
            for jid, job in list(self._jobs.items()):
                if job.status in ("done", "error") and now - job.touched > config.JOB_TTL_SECONDS:
                    expired.append(job)
                    self._jobs.pop(jid, None)
                    if jid in self._order:
                        self._order.remove(jid)
        for job in expired:
            shutil.rmtree(job.dir, ignore_errors=True)
        # مجلدات يتيمة (مثلاً رفع فشل قبل إنشاء المهمة)
        known = {j.dir.name for j in self._jobs.values()}
        for d in config.TMP_ROOT.glob("*"):
            if d.is_dir() and d.name not in known and now - d.stat().st_mtime > config.JOB_TTL_SECONDS:
                shutil.rmtree(d, ignore_errors=True)
        return len(expired)


manager = JobManager()
