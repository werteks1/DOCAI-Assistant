import uuid

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from .jobs import JobBusy, JobStore, exclusive_extraction, friendly_error
from .models import (
    BatchExtractRequest, BatchJobItem, BatchJobStarted, BatchJobState,
    ConnectRequest, CorrectionRequest, ExportRequest, ExtractRequest,
    ExtractResponse, HealthResponse, ModelRequest,
)
from excel_manager import ExcelManager
from logging_utils import get_logger, reset_correlation, set_correlation
from .pipeline import CompilerPipeline

logger = get_logger("api")

app = FastAPI(title="DocAI Compiler", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173", "http://localhost:5173",
        "http://127.0.0.1:4173", "http://localhost:4173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
pipeline = CompilerPipeline()
job_store = JobStore()

# Множество статусов, при которых джоб завершён и можно отдавать data файлов.
_FINISHED_STATUSES = ("done", "error", "cancelled")


def _http_409_job_busy(exc: JobBusy) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


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
        return pipeline.connect(request.host)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc


@app.post("/api/model")
def choose_model(request: ModelRequest) -> dict[str, str]:
    try:
        return {"model": pipeline.set_model(request.model)}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
