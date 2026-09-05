from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

import config
from .models import (
    BatchExtractRequest, BatchExtractResponse, BatchItem, ConnectRequest,
    CorrectionRequest, ExportRequest, ExtractRequest, ExtractResponse, HealthResponse, ModelRequest,
)
from excel_manager import ExcelManager
from .pipeline import CompilerPipeline


app = FastAPI(title="DocAI Compiler", version="1.0.0")
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


def _friendly_error(exc: Exception) -> str:
    """Преобразует сетевые ошибки Ollama в понятное сообщение для Web UI."""
    raw = str(exc)
    lower = raw.lower()
    if "61" in raw or "connection refused" in lower or "connecterror" in lower:
        return (
            "Сервер ИИ отказал в подключении (ошибка 61). "
            "Проверьте, что Ollama/LM Studio запущен, IP и порт указаны верно, "
            "а доступ к серверу разрешён в локальной сети."
        )
    return raw or "Неизвестная ошибка компилятора"


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
        raise HTTPException(status_code=422, detail=_friendly_error(exc)) from exc


@app.post("/api/model")
def choose_model(request: ModelRequest) -> dict[str, str]:
    try:
        return {"model": pipeline.set_model(request.model)}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest) -> ExtractResponse:
    try:
        data = pipeline.extract(
            request.image_base64,
            request.target_columns,
            request.filename,
            request.ocr_priority,
            request.detector,
            request.model,
        )
        return ExtractResponse(data=data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=_friendly_error(exc)) from exc


@app.post("/api/batch", response_model=BatchExtractResponse)
def batch_extract(request: BatchExtractRequest) -> BatchExtractResponse:
    results: list[BatchItem] = []
    for document in request.documents:
        try:
            data = pipeline.extract(
                document.image_base64,
                request.target_columns,
                document.filename,
                request.ocr_priority,
                request.detector,
                request.model,
            )
            results.append(BatchItem(filename=document.filename, data=data))
        except Exception as exc:
            results.append(BatchItem(filename=document.filename, error=_friendly_error(exc)))
    return BatchExtractResponse(results=results)


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
