import os
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import settings_store
from . import auth_store
from .auth import current_user, enforce_change, require_admin
from .jobs import JobBusy, JobStore, exclusive_extraction, friendly_error
from .models import (
    BatchExtractRequest, BatchJobItem, BatchJobStarted, BatchJobState,
    AuthUser, ChangePasswordRequest, ConnectRequest, CorrectionRequest,
    ExportRequest, ExtractRequest, ExtractResponse, HealthResponse,
    LoginRequest, LoginResponse, MeResponse, ModelRequest, ParamsModel,
    SettingsSaveRequest, UserCreateRequest, UserResetPasswordRequest,
    UserSetDisabledRequest, UserView,
)
from excel_manager import ExcelManager
from logging_utils import get_logger, reset_correlation, set_correlation
from .pipeline import CompilerPipeline
import config

logger = get_logger("api")

app = FastAPI(title="DocAI Compiler", version="1.3.0")

_allowed_origins = [
    "http://127.0.0.1:5173", "http://localhost:5173",
    "http://127.0.0.1:4173", "http://localhost:4173",
]
# Дополнительные источники из DOCIA_ALLOWED_ORIGINS (через запятую) для LAN.
for _origin in os.getenv("DOCIA_ALLOWED_ORIGINS", "").split(","):
    _origin = _origin.strip()
    if _origin and _origin not in _allowed_origins:
        _allowed_origins.append(_origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=True,
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

# Создаёт docia.db со схемой и сидирует admin/admin (must_change=1), если пусто.
auth_store.init_db()

# Множество статусов, при которых джоб завершён и можно отдавать data файлов.
_FINISHED_STATUSES = ("done", "error", "cancelled")


def _log_action(
    username: str,
    action: str,
    *,
    ok: bool = True,
    elapsed_ms: Optional[int] = None,
    detail: Optional[dict] = None,
) -> None:
    """PII-safe запись в журнал активности со служебными полями модели/бэкенда."""
    auth_store.log_activity(
        username,
        action,
        ok=ok,
        elapsed_ms=elapsed_ms,
        model=pipeline.extractor.model_name,
        backend=pipeline.extractor.backend,
        detail=detail,
    )


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
def connect(request: ConnectRequest, admin: dict = Depends(require_admin)) -> dict[str, object]:
    try:
        with exclusive_extraction():
            info = pipeline.connect(request.host, request.api_key)
        _log_action(admin["username"], "connect", ok=bool(info.get("connected")),
                    detail={"host_set": bool(request.host)})
        return info
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    except Exception as exc:
        _log_action(admin["username"], "connect", ok=False)
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc


@app.post("/api/model")
def choose_model(request: ModelRequest, admin: dict = Depends(require_admin)) -> dict[str, str]:
    try:
        model = pipeline.set_model(request.model)
        _log_action(admin["username"], "model", ok=True)
        return {"model": model}
    except Exception as exc:
        _log_action(admin["username"], "model", ok=False)
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# --- Настройки модели/сервера (отдельное приложение /settings) ----------

@app.get("/api/settings")
def get_settings(_admin: dict = Depends(require_admin)) -> dict[str, object]:
    """Текущее состояние + что сохранено в settings.json (ключ маскирован)."""
    return _settings_response(settings_store.load_settings())


@app.post("/api/settings")
def save_settings(
    request: SettingsSaveRequest, admin: dict = Depends(require_admin)
) -> dict[str, object]:
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
        _log_action(admin["username"], "settings_save", ok=True,
                    detail={"host_set": bool(host)})
        return _settings_response(saved)
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    except Exception as exc:
        _log_action(admin["username"], "settings_save", ok=False)
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc


@app.post("/api/settings/reset")
def reset_settings(_admin: dict = Depends(require_admin)) -> dict[str, object]:
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
        _log_action("settings", "settings_reset", ok=True)
        return _settings_response({})
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    except Exception as exc:
        _log_action("settings", "settings_reset", ok=False)
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc


@app.get("/settings", include_in_schema=False)
def settings_page() -> FileResponse:
    """Отдельное приложение «Настройки» — самодостаточная HTML-страница."""
    return FileResponse(STATIC_DIR / "settings.html")


# --- Авторизация: вход / выход / смена пароля / кто я ------------------

def _auth_user(user: dict) -> AuthUser:
    return AuthUser(
        username=user["username"], role=user["role"], must_change=user["must_change"]
    )


@app.post("/api/auth/login", response_model=LoginResponse)
def auth_login(request: LoginRequest) -> LoginResponse:
    result = auth_store.verify_login(request.username, request.password)
    if result.locked:
        _log_action(request.username, "login", ok=False)
        raise HTTPException(
            status_code=429,
            detail="Слишком много неудачных попыток. Подождите 15 минут и попробуйте снова.",
        )
    if result.user is None:
        _log_action(request.username, "login", ok=False)
        raise HTTPException(status_code=401, detail="Неверный логин или пароль.")
    token = auth_store.create_session(result.user["id"])
    _log_action(result.user["username"], "login", ok=True)
    return LoginResponse(token=token, user=_auth_user(result.user))


@app.post("/api/auth/logout")
def auth_logout(
    user: dict = Depends(current_user),
    authorization: str = Header(default=""),
) -> dict[str, str]:
    token = authorization.partition(" ")[2].strip() if authorization else ""
    auth_store.delete_session(token)
    _log_action(user["username"], "logout", ok=True)
    return {"status": "ok"}


@app.get("/api/auth/me", response_model=MeResponse)
def auth_me(user: dict = Depends(current_user)) -> MeResponse:
    return MeResponse(user=_auth_user(user))


@app.post("/api/auth/change-password")
def auth_change_password(
    request: ChangePasswordRequest,
    user: dict = Depends(current_user),
) -> dict[str, str]:
    """Смена пароля текущим пользователем. Обязательна, если must_change."""
    if not auth_store.verify_user_password(user["id"], request.current_password):
        _log_action(user["username"], "change_password", ok=False)
        raise HTTPException(status_code=400, detail="Текущий пароль указан неверно.")
    try:
        # Свою сессию не отзываем, чтобы не разлогинивать сразу после смены.
        auth_store.set_password(
            user["id"], request.new_password, must_change=False, revoke_sessions=False
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _log_action(user["username"], "change_password", ok=True)
    return {"status": "ok"}


# --- Пользователи (только администратор) ------------------------------

@app.get("/api/users", response_model=list[UserView])
def users_list(_admin: dict = Depends(require_admin)) -> list[dict]:
    return auth_store.list_users()


@app.post("/api/users", response_model=UserView)
def users_create(
    request: UserCreateRequest, admin: dict = Depends(require_admin)
) -> dict:
    try:
        created = auth_store.create_user(
            request.username, request.password, request.role,
            must_change=request.must_change,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _log_action(admin["username"], "user_create", ok=True,
                detail={"target": created["username"]})
    return created


@app.post("/api/users/{user_id}/reset-password")
def users_reset_password(
    user_id: int,
    request: UserResetPasswordRequest,
    admin: dict = Depends(require_admin),
) -> dict[str, str]:
    target = auth_store.get_user_by_id(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    try:
        auth_store.set_password(
            user_id, request.password, must_change=request.must_change,
            revoke_sessions=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _log_action(admin["username"], "user_reset_password", ok=True,
                detail={"target": target["username"]})
    return {"status": "ok"}


@app.post("/api/users/{user_id}/disable")
def users_set_disabled(
    user_id: int,
    request: UserSetDisabledRequest,
    admin: dict = Depends(require_admin),
) -> dict[str, str]:
    target = auth_store.get_user_by_id(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    if request.disabled and target["id"] == admin["id"]:
        raise HTTPException(status_code=400, detail="Нельзя заблокировать собственную учётку")
    auth_store.set_disabled(user_id, request.disabled)
    _log_action(admin["username"],
                "user_disable" if request.disabled else "user_enable",
                ok=True, detail={"target": target["username"]})
    return {"status": "ok"}


@app.delete("/api/users/{user_id}")
def users_delete(user_id: int, admin: dict = Depends(require_admin)) -> dict[str, str]:
    target = auth_store.get_user_by_id(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    if target["id"] == admin["id"]:
        raise HTTPException(status_code=400, detail="Нельзя удалить собственную учётку")
    auth_store.delete_user(user_id)
    _log_action(admin["username"], "user_delete", ok=True,
                detail={"target": target["username"]})
    return {"status": "ok"}


# --- Статистика использования (только администратор) ------------------

@app.get("/api/stats")
def stats(_admin: dict = Depends(require_admin)) -> dict:
    return auth_store.activity_summary()


@app.post("/api/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest, user: dict = Depends(enforce_change)) -> ExtractResponse:
    token = set_correlation(uuid.uuid4().hex[:12])
    started = time.monotonic()
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
        _log_action(user["username"], "extract", ok=True,
                    elapsed_ms=int((time.monotonic() - started) * 1000))
        return ExtractResponse(data=data)
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    except Exception as exc:
        logger.warning("extract: error=%s", type(exc).__name__)
        _log_action(user["username"], "extract", ok=False,
                    elapsed_ms=int((time.monotonic() - started) * 1000))
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc
    finally:
        reset_correlation(token)


@app.post("/api/batch", status_code=202, response_model=BatchJobStarted)
def batch_start(
    request: BatchExtractRequest, user: dict = Depends(enforce_change)
) -> BatchJobStarted:
    try:
        job_id = job_store.start(request, pipeline, username=user["username"])
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    _log_action(user["username"], "batch_start", ok=True,
                detail={"files": len(request.documents)})
    return BatchJobStarted(job_id=job_id, total=len(request.documents))


@app.get("/api/batch/{job_id}", response_model=BatchJobState)
def batch_status(job_id: str, user: dict = Depends(enforce_change)) -> BatchJobState:
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
def batch_cancel(
    job_id: str, user: dict = Depends(enforce_change)
) -> dict[str, str]:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Джоб не найден")
    if job.status not in ("running",):
        return {"status": "finished"}
    cancelled = job_store.cancel(job_id)
    _log_action(user["username"], "batch_cancel", ok=True,
                detail={"job": job_id[:8]})
    return {"status": "cancelling" if cancelled else "finished"}


@app.post("/api/export/excel")
def export_excel(request: ExportRequest, user: dict = Depends(enforce_change)) -> Response:
    try:
        content = ExcelManager.export_records_bytes(request.records)
    except Exception as exc:
        _log_action(user["username"], "export_excel", ok=False)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _log_action(user["username"], "export_excel", ok=True,
                detail={"records": len(request.records)})
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=docai_results.xlsx"},
    )


@app.post("/api/reference/corrections")
def add_correction(
    request: CorrectionRequest, user: dict = Depends(enforce_change)
) -> dict[str, str]:
    pipeline.extractor.reference_dictionary.add_correction(
        request.category, request.wrong, request.right
    )
    _log_action(user["username"], "correction", ok=True,
                detail={"category": request.category})
    return {"status": "saved"}


# --- LAN-режим: собранный фронтенд отдаёт сам компилятор -----------------
# Монтируется ПОСЛЕДНИМ (после всех /api/* и /settings), чтобы не перекрывать
# маршруты API. Если web/dist ещё не собран — работает dev-режим через Vite.
if config.WEB_DIST_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(config.WEB_DIST_DIR), html=True),
        name="web",
    )
    logger.info("собранный фронтенд смонтирован из %s", config.WEB_DIST_DIR)
else:
    logger.info("web/dist не найден (%s) — фронтенд ожидается на Vite :5173",
                config.WEB_DIST_DIR)
