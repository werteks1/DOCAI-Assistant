from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field
import config


class ExtractRequest(BaseModel):
    filename: str = "document.png"
    image_base64: str = Field(min_length=16)
    target_columns: Optional[list[str]] = None
    ocr_priority: Literal["auto", "paddle", "vlm"] = "auto"
    detector: Literal["PP-OCRv6_medium_det", "PP-OCRv6_small_det"] = config.PADDLE_DETECTOR
    model: Optional[str] = Field(default=None, max_length=200)


class BatchDocument(BaseModel):
    filename: str = "document.png"
    image_base64: str = Field(min_length=16)


class BatchExtractRequest(BaseModel):
    documents: list[BatchDocument] = Field(min_length=1, max_length=50)
    target_columns: Optional[list[str]] = None
    ocr_priority: Literal["auto", "paddle", "vlm"] = "auto"
    detector: Literal["PP-OCRv6_medium_det", "PP-OCRv6_small_det"] = config.PADDLE_DETECTOR
    model: Optional[str] = Field(default=None, max_length=200)


class BatchJobStarted(BaseModel):
    job_id: str
    total: int


class BatchJobItem(BaseModel):
    filename: str
    status: Literal["pending", "processing", "done", "error"] = "pending"
    message: str = ""
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class BatchJobState(BaseModel):
    job_id: str
    status: Literal["running", "done", "error", "cancelled"] = "running"
    cancelled: bool = False
    total: int = 0
    completed: int = 0
    current_message: str = ""
    error: Optional[str] = None
    items: list[BatchJobItem] = Field(default_factory=list)


class ExportRequest(BaseModel):
    records: list[Dict[str, Any]] = Field(min_length=1, max_length=500)


class CorrectionRequest(BaseModel):
    category: str
    wrong: str
    right: str


class ConnectRequest(BaseModel):
    host: str = Field(min_length=1, max_length=255)


class ModelRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)


class HealthResponse(BaseModel):
    status: str
    paddle_available: bool
    model: str
    host: str
    backend: str
    models: list[str] = Field(default_factory=list)


class ExtractResponse(BaseModel):
    data: Dict[str, Any]
