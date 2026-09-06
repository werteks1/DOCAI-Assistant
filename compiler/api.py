import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .jobs import JobBusy, JobStore, exclusive_extraction, friendly_error
from .models import (
    BatchExtractRequest, BatchJobItem, BatchJobStarted, BatchJobState,
    ConnectRequest, CorrectionRequest, ExportRequest, ExtractRequest,
    ExtractResponse, HealthResponse, ModelRequest, ParamsModel,
    SettingsSaveRequest,
)
from . import settings_store
from excel_manager import ExcelManager
from logging_utils import get_logger, reset_correlation, set_correlation
from .pipeline import CompilerPipeline
import config

logger = get_logger("api")

app = FastAPI(title="DocAI Compiler", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173", "http://localhost:5173",
        "http://127.0.0.1:4173", "http://localhost:4173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Старт из settings.json: адрес/модель/параметры применяются до первого запроса.
_persisted = settings_store.load_settings()
pipeline = CompilerPipeline(host=_persisted.get("host"), model_name=_persisted.get("model"))
if isinstance(_persisted.get("params"), dict):
    pipeline.extractor.apply_params(_persisted["params"])
if _persisted.get("api_key"):
    pipeline.extractor.set_api_key(_persisted["api_key"])
job_store = JobStore()

# Множество статусов, при которых джоб завершён и можно отдавать data файлов.
_FINISHED_STATUSES = ("done", "error", "cancelled")


def _http_409_job_busy(exc: JobBusy) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


def _mask_key(key: str) -> str:
    """Маскирует API-ключ для вывода в интерфейс."""
    key = key or ""
    if len(key) <= 4:
        return "*" * len(key) if key else ""
    return key[:2] + "…" + key[-1]


def _apply_params_model(params: ParamsModel) -> None:
    payload = {
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_tokens": params.max_tokens,
        "timeout_seconds": params.timeout_seconds,
        "extra_system_prompt": params.extra_system_prompt,
    }
    pipeline.extractor.apply_params(payload)


def _active_state() -> dict[str, object]:
    ex = pipeline.extractor
    return {
        "host": ex.host,
        "backend": ex.backend,
        "model": ex.model_name,
        "paddle_available": pipeline.paddle_available,
        "models": list(pipeline.available_models),
        "connected": bool(pipeline.available_models),
        "params": ex.params_snapshot(),
    }


def _settings_response(saved: dict) -> dict[str, object]:
    api_key = saved.get("api_key") or ""
    return {
        "exists": bool(saved),
        "saved": {
            "host": saved.get("host") or "",
            "model": saved.get("model") or "",
            "api_key_set": bool(api_key),
            "api_key_hint": _mask_key(api_key),
            "params": dict(saved.get("params") or {}),
        },
        "active": _active_state(),
    }


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        paddle_available=pipeline.paddle_available,
        model=pipeline.extractor.model_name,
        host=pipeline.extractor.host,
        backend=pipeline.extractor.backend,
        models=pipeline.available_models,
    )


@app.post("/api/connect")
def connect(request: ConnectRequest) -> dict[str, object]:
    try:
        with exclusive_extraction():
            return pipeline.connect(request.host, request.api_key)
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc


@app.post("/api/model")
def choose_model(request: ModelRequest) -> dict[str, str]:
    try:
        return {"model": pipeline.set_model(request.model)}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# --- Настройки модели/сервера (отдельное приложение /settings) ----------

@app.get("/api/settings")
def get_settings() -> dict[str, object]:
    """Текущее состояние + что сохранено в settings.json (ключ маскирован)."""
    return _settings_response(settings_store.load_settings())


@app.post("/api/settings")
def save_settings(request: SettingsSaveRequest) -> dict[str, object]:
    """Применяет и сохраняет настройки. Смена адреса проверяется подключением."""
    try:
        with exclusive_extraction():
            host = (request.host or "").strip()
            if host and host != pipeline.extractor.host:
                info = pipeline.connect(
                    host,
                    request.api_key if request.api_key is not None else pipeline.extractor.api_key,
                )
                if not info["connected"]:
                    raise ValueError(
                        "Сервер ИИ по этому адресу недоступен: не удалось получить список "
                        "моделей. Адрес не сохранён — проверьте адрес и запущен ли сервер."
                    )
            if request.api_key is not None:
                pipeline.extractor.set_api_key(request.api_key)
            if request.params is not None:
                _apply_params_model(request.params)
            if request.model:
                model = request.model.strip()
                if pipeline.available_models and model not in pipeline.available_models:
                    raise ValueError("Выбранная модель не найдена на подключённом сервере")
                pipeline.extractor.model_name = model
            saved = settings_store.save_settings({
                "host": pipeline.extractor.host,
                "model": pipeline.extractor.model_name,
                "api_key": pipeline.extractor.api_key,
                "params": pipeline.extractor.params_snapshot(),
            })
        return _settings_response(saved)
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc


@app.post("/api/settings/reset")
def reset_settings() -> dict[str, object]:
    """Сбрасывает настройки: удаляет settings.json и возвращает заводские значения."""
    try:
        with exclusive_extraction():
            settings_store.remove_settings()
            ex = pipeline.extractor
            ex.apply_default_params()
            ex.set_api_key("")
            ex.set_host(config.OLLAMA_HOST)
            ex.model_name = config.DEFAULT_MODEL
            try:
                models = ex.get_available_models()
                pipeline.available_models = models
                if ex.model_name not in models:
                    qwen25 = [m for m in models if "2.5-vl" in m.lower()]
                    if qwen25: ex.model_name = qwen25[0]
                    elif models: ex.model_name = models[0]
            except Exception:
                pipeline.available_models = []
        return _settings_response({})
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc


@app.get("/settings", include_in_schema=False)
def settings_page() -> FileResponse:
    """Отдельное приложение «Настройки» — самодостаточная HTML-страница."""
    return FileResponse(STATIC_DIR / "settings.html")


@app.post("/api/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest) -> ExtractResponse:
    token = set_correlation(uuid.uuid4().hex[:12])
    try:
        with exclusive_extraction():
            data = pipeline.extract(
                request.image_base64,
                request.target_columns,
                request.filename,
                request.ocr_priority,
                request.detector,
                request.model,
            )
        return ExtractResponse(data=data)
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    except Exception as exc:
        logger.warning("extract: error=%s", type(exc).__name__)
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc
    finally:
        reset_correlation(token)


@app.post("/api/batch", status_code=202, response_model=BatchJobStarted)
def batch_start(request: BatchExtractRequest) -> BatchJobStarted:
    try:
        job_id = job_store.start(request, pipeline)
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    return BatchJobStarted(job_id=job_id, total=len(request.documents))


@app.get("/api/batch/{job_id}", response_model=BatchJobState)
def batch_status(job_id: str) -> BatchJobState:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Джоб не найден")
    finished = job.status in _FINISHED_STATUSES
    return BatchJobState(
        job_id=job.job_id,
        status=job.status,
        cancelled=job.cancelled,
        total=job.total,
        completed=job.completed,
        current_message=job.current_message,
        error=job.error,
        items=[
            BatchJobItem(
                filename=item.filename,
                status=item.status,
                message=item.message,
                error=item.error,
                data=item.data if finished else None,
            )
            for item in job.items
        ],
    )


@app.post("/api/batch/{job_id}/cancel")
def batch_cancel(job_id: str) -> dict[str, str]:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Джоб не найден")
    if job.status not in ("running",):
        return {"status": "finished"}
    cancelled = job_store.cancel(job_id)
    return {"status": "cancelling" if cancelled else "finished"}


@app.post("/api/export/excel")
def export_excel(request: ExportRequest) -> Response:
    try:
        content = ExcelManager.export_records_bytes(request.records)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=docai_results.xlsx"},
    )


@app.post("/api/reference/corrections")
def add_correction(request: CorrectionRequest) -> dict[str, str]:
    pipeline.extractor.reference_dictionary.add_correction(
        request.category, request.wrong, request.right
    )
    return {"status": "saved"}
