"""Фоновые пакетные джобы распознавания.

DocumentExtractor не потокобезопасен и разделяется между запросами (общий
экземпляр `pipeline` в api.py), поэтому:
- вся экстракция — одиночная и пакетная — сериализуется глобальным
  EXTRACTION_LOCK;
- пакет выполняется последовательно в одном фоновом потоке;
- одновременно активен только один джоб; повторный запуск → JobBusy (409).

Логи пишутся в PII-safe режиме: только индексы файлов, этапы и типы ошибок,
без имён исходных файлов и значений полей (см. logging_utils.py).
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from excel_manager import ExcelManager
from logging_utils import get_logger, reset_correlation, set_correlation

logger = get_logger("jobs")

# Единая блокировка любой экстракции (одиночной и пакетной).
EXTRACTION_LOCK = threading.RLock()

# Сколько завершённых джобов хранить в памяти (результаты забирает веб-клиент).
KEEP_FINISHED_JOBS = 20

# Литералы статусов.
RUNNING = "running"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"

_PENDING = "pending"
_PROCESSING = "processing"


class JobBusy(RuntimeError):
    """Экстракция уже выполняется (другой джоб или одиночный запрос)."""


class exclusive_extraction:
    """Контекст-менеджер: захватывает EXTRACTION_LOCK без ожидания.

    Используется и одиночным /api/extract, и фоном джоба. Если экстракция уже
    идёт — поднимает JobBusy, чтобы запрос получил быстрый 409, а не висел.
    """

    def __enter__(self):
        if not EXTRACTION_LOCK.acquire(blocking=False):
            raise JobBusy("Экстракция уже выполняется (пакетная обработка). Дождитесь её завершения.")
        return self

    def __exit__(self, exc_type, exc, tb):
        EXTRACTION_LOCK.release()
        return False


def friendly_error(exc: Exception) -> str:
    """Преобразует ошибку в понятное для Web UI сообщение (без персональных данных)."""
    raw = str(exc) or "Неизвестная ошибка компилятора"
    lower = raw.lower()
    if "61" in raw or "connection refused" in lower or "connecterror" in lower:
        return (
            "Сервер ИИ отказал в подключении (ошибка 61). "
            "Проверьте, что Ollama/LM Studio запущен, IP и порт указаны верно, "
            "а доступ к серверу разрешён в локальной сети."
        )
    return raw


@dataclass
class JobItem:
    filename: str
    status: str = _PENDING
    message: str = ""
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@dataclass
class Job:
    job_id: str
    total: int = 0
    status: str = RUNNING
    cancelled: bool = False
    completed: int = 0
    current_message: str = ""
    error: Optional[str] = None
    items: List[JobItem] = field(default_factory=list)
    _active: bool = field(default=True, repr=False)


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()

    def _active_job(self) -> Optional[Job]:
        with self._lock:
            for job in self._jobs.values():
                if job.status == RUNNING and not job.cancelled:
                    return job
        return None

    def start(self, request, pipeline) -> str:
        """Создаёт и запускает джоб. Поднимает JobBusy, если активен другой."""
        with self._lock:
            if self._active_job() is not None:
                raise JobBusy("Пакетная обработка уже выполняется. Дождитесь её завершения.")
            job = Job(job_id=uuid.uuid4().hex, total=len(request.documents))
            job.items = [JobItem(filename=document.filename) for document in request.documents]
            self._jobs[job.job_id] = job
        thread = threading.Thread(
            target=self._run,
            args=(job.job_id, request, pipeline),
            name=f"docai-batch-{job.job_id[:8]}",
            daemon=True,
        )
        thread.start()
        return job.job_id

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != RUNNING:
                return False
            job.cancelled = True
            return True

    def _update(self, job: Job) -> None:
        """Обновление снапшота джоба под блокировкой; хранилище ограничено по памяти."""
        with self._lock:
            if job.status in (DONE, ERROR, CANCELLED):
                finished = sorted(
                    (j for j in self._jobs.values() if j.status in (DONE, ERROR, CANCELLED)),
                    key=lambda j: j.job_id,
                )
                if len(finished) > KEEP_FINISHED_JOBS:
                    for stale in finished[:-KEEP_FINISHED_JOBS]:
                        self._jobs.pop(stale.job_id, None)

    # --- Выполнение -----------------------------------------------------

    def _run(self, job_id: str, request, pipeline) -> None:
        token = set_correlation(job_id)
        try:
            job = self.get(job_id)
            if job is None:
                return
            with exclusive_extraction():
                job = self.get(job_id)
                for index, document in enumerate(request.documents):
                    if job.cancelled:
                        break
                    item = job.items[index]
                    item.status = _PROCESSING
                    item.message = f"Файл {index + 1} из {job.total}"
                    job.current_message = f"Файл {index + 1} из {job.total}"
                    job.completed = index
                    try:
                        data = pipeline.extract(
                            document.image_base64,
                            request.target_columns,
                            document.filename,
                            request.ocr_priority,
                            request.detector,
                            request.model,
                        )
                        item.data = data
                        item.status = DONE
                        item.error = None
                        item.message = ""
                        logger.info("файл_%d из %d: ok", index + 1, job.total)
                    except Exception as exc:  # ошибка одного файла не валит джоб
                        item.status = ERROR
                        item.error = friendly_error(exc)
                        item.message = ""
                        logger.warning("файл_%d из %d: error=%s",
                                       index + 1, job.total, type(exc).__name__)
                    job.completed = index + 1
                if job.cancelled:
                    job.status = CANCELLED
                    job.current_message = "Обработка отменена"
                    return
                self._annotate_duplicates(job)
                job.status = DONE
                job.current_message = "Готово"
                logger.info("джоб %s завершён: %d файлов", job_id[:8], job.total)
        except JobBusy:
            # Не должно случаться: джоб стартует, только когда экстракция свободна.
            logger.error("джоб %s не смог захватить блокировку экстракции", job_id[:8])
            job = self.get(job_id)
            if job is not None:
                job.status = ERROR
                job.error = "Не удалось начать обработку: экстракция занята"
        except Exception as exc:
            logger.error("джоб %s: неожиданная ошибка %s", job_id[:8], type(exc).__name__)
            job = self.get(job_id)
            if job is not None:
                job.status = ERROR
                job.error = friendly_error(exc)
        finally:
            reset_correlation(token)
            job = self.get(job_id)
            if job is not None:
                self._update(job)

    def _annotate_duplicates(self, job: Job) -> None:
        """Помечает дубли внутри пачки результатами ExcelManager.find_in_memory_duplicates."""
        records: List[Dict[str, Any]] = []
        filenames: List[str] = []
        for item in job.items:
            if item.status == DONE and item.data:
                records.append(item.data)
                filenames.append(item.filename)
        if len(records) < 2:
            return
        try:
            dupes = ExcelManager.find_in_memory_duplicates(records, filenames)
            data_items = [item for item in job.items if item.status == DONE and item.data]
            for item, found in zip(data_items, dupes):
                if found:
                    item.data["_duplicates"] = found
        except Exception as exc:
            logger.warning("дубли: ошибка %s", type(exc).__name__)
