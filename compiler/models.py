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
    # Опциональный Bearer-ключ для OpenAI-совместимого сервера (LM Studio не требует).
    api_key: Optional[str] = Field(default=None, max_length=255)


class ModelRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)


class ParamsModel(BaseModel):
    """Параметры запроса VLM. None в top_p/timeout_seconds = серверный дефолт."""
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=200000)
    timeout_seconds: Optional[int] = Field(default=None, ge=1, le=3600)
    extra_system_prompt: Optional[str] = Field(default=None, max_length=4000)


class SettingsSaveRequest(BaseModel):
    """Частичное обновление настроек. None в api_key = не менять; "" = очистить."""
    host: Optional[str] = Field(default=None, max_length=255)
    api_key: Optional[str] = Field(default=None, max_length=255)
    model: Optional[str] = Field(default=None, max_length=200)
    params: Optional[ParamsModel] = None


class HealthResponse(BaseModel):
    status: str
    paddle_available: bool
    model: str
    host: str
    backend: str
    models: list[str] = Field(default_factory=list)


class ExtractResponse(BaseModel):
    data: Dict[str, Any]


# --- Авторизация и роли -------------------------------------------------

class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class AuthUser(BaseModel):
    username: str
    role: Literal["admin", "operator"]
    must_change: bool


class LoginResponse(BaseModel):
    token: str
    user: AuthUser


class MeResponse(BaseModel):
    user: AuthUser


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=4, max_length=256)


class UserView(BaseModel):
    id: int
    username: str
    role: Literal["admin", "operator"]
    must_change: bool
    disabled: bool
    created_at: str
    last_login_at: Optional[str] = None


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=4, max_length=256)
    role: Literal["admin", "operator"] = "operator"
    must_change: bool = True


class UserResetPasswordRequest(BaseModel):
    password: str = Field(min_length=4, max_length=256)
    must_change: bool = True


class UserSetDisabledRequest(BaseModel):
    disabled: bool
