import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import settings_store
from . import auth_store
from . import connections as connections_store
from .auth import current_user, enforce_change, require_admin
from .jobs import Job, JobBusy, JobStore, exclusive_extraction, friendly_error, make_preview
from .models import (
    BatchExtractRequest, BatchJobItem, BatchJobStarted, BatchJobState,
    AuthUser, ChangePasswordRequest, ConnectRequest, ConnectionCheckRequest,
    ConnectionCreateRequest, ConnectionUpdateRequest, CorrectionRequest,
    DuplicatesRequest, ExportRequest, ExtractRequest, ExtractResponse,
    HealthResponse, LoginRequest, LoginResponse, MeResponse, ModelRequest,
    ParamsModel, PreviewRequest, SettingsSaveRequest, UserCreateRequest,
    UserResetPasswordRequest, UserSetDisabledRequest, UserView,
)
from excel_manager import ExcelManager
from logging_utils import get_logger, reset_correlation, set_correlation
from .pipeline import CompilerPipeline
from .operations import OperationStore
from cancellation import ExtractionCancelled
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
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=True,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Старт из settings.json: активное подключение (адрес/ключ/модель) и параметры
# запроса применяются до первого обращения. Старый одиночный host мигрируется.
_persisted = settings_store.load_settings()
_connections, _active_connection_id = connections_store.from_saved(_persisted)
_active_connection = connections_store.active(_connections, _active_connection_id)
pipeline = CompilerPipeline(
    host=(_active_connection or {}).get("host"),
    model_name=(_active_connection or {}).get("model"),
)
if isinstance(_persisted.get("params"), dict):
    pipeline.extractor.apply_params(_persisted["params"])
if _active_connection and _active_connection.get("api_key"):
    pipeline.extractor.set_api_key(_active_connection["api_key"])
    # Опрос моделей после применения ключа: облачные OpenAI-совместимые API
    # (SiliconFlow, SeekAI и др.) без него отдают 401 и пустой список.
    pipeline.available_models = pipeline.extractor.get_available_models()
    if (pipeline.available_models
            and pipeline.extractor.model_name not in pipeline.available_models):
        preferred = next(
            (m for m in pipeline.available_models if "2.5-vl" in m.lower()),
            pipeline.available_models[0],
        )
        pipeline.extractor.model_name = preferred
job_store = JobStore()
operation_store = OperationStore()

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


def _connections_state(saved: Optional[dict] = None) -> tuple[list[dict], Optional[str]]:
    """Подключения из settings.json в каноническом виде (+ миграция старого host)."""
    return connections_store.from_saved(saved if saved is not None else settings_store.load_settings())


def _persist_connections(items: list[dict], active_id: Optional[str]) -> dict:
    """Сохраняет список подключений, убирая устаревшие одиночные поля."""
    return settings_store.save_settings(
        connections_store.payload(items, active_id),
        drop=("host", "api_key", "model"),
    )


def _public_connection(conn: dict, active_id: Optional[str]) -> dict[str, object]:
    """Подключение для интерфейса: ключ только маской, без открытого значения."""
    api_key = conn.get("api_key") or ""
    return {
        "id": conn["id"],
        "name": conn["name"],
        "host": conn["host"],
        "model": conn.get("model") or "",
        "api_key_set": bool(api_key),
        "api_key_hint": _mask_key(api_key),
        "active": conn["id"] == active_id,
    }


def _reset_pipeline_defaults() -> None:
    """Возвращает экстрактор к заводским значениям (без сохранения в файл)."""
    ex = pipeline.extractor
    ex.apply_default_params()
    ex.set_api_key("")
    ex.set_host(config.OLLAMA_HOST)
    ex.model_name = config.DEFAULT_MODEL
    pipeline.available_models = []


def _active_state() -> dict[str, object]:
    ex = pipeline.extractor
    items, active_id = _connections_state()
    conn = connections_store.active(items, active_id)
    backend_label = getattr(ex, "_backend_label", None)
    return {
        "host": ex.host,
        "backend": backend_label() if callable(backend_label) else ex.backend,
        "model": ex.model_name,
        "connection": conn["name"] if conn else "",
        "connection_id": conn["id"] if conn else "",
        "paddle_available": pipeline.paddle_available,
        "models": list(pipeline.available_models),
        "connected": bool(pipeline.available_models),
        "params": ex.params_snapshot(),
    }


def _settings_response(saved: dict, warning: str = "") -> dict[str, object]:
    items, active_id = connections_store.from_saved(saved)
    conn = connections_store.active(items, active_id)
    api_key = (conn or {}).get("api_key") or ""
    return {
        "exists": bool(saved),
        "warning": warning,
        "saved": {
            "connections": [_public_connection(item, active_id) for item in items],
            "active_connection_id": active_id or "",
            "limit": connections_store.MAX_CONNECTIONS,
            "host": (conn or {}).get("host") or "",
            "model": (conn or {}).get("model") or "",
            "api_key_set": bool(api_key),
            "api_key_hint": _mask_key(api_key),
            "params": dict(saved.get("params") or {}),
        },
        "active": _active_state(),
    }


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    items, active_id = _connections_state()
    conn = connections_store.active(items, active_id)
    return HealthResponse(
        status="ok",
        paddle_available=pipeline.paddle_available,
        model=pipeline.extractor.model_name,
        host=pipeline.extractor.host,
        backend=pipeline.extractor.backend,
        connection=conn["name"] if conn else "",
        models=pipeline.available_models,
    )


@app.post("/api/connect")
def connect(request: ConnectRequest, admin: dict = Depends(require_admin)) -> dict[str, object]:
    """Совместимость: проверяет адрес и делает его активным подключением.

    Подключение заводится в списке (или берётся существующее с тем же
    адресом), чтобы старые вызовы не создавали второй источник правды.
    """
    try:
        with exclusive_extraction():
            items, active_id = _connections_state()
            host = pipeline.extractor._clean_host(request.host)
            api_key = request.api_key if request.api_key is not None else ""
            conn = next((c for c in items if c["host"] == host), None)
            if conn is None:
                if len(items) >= connections_store.MAX_CONNECTIONS:
                    raise ValueError(f"Достигнут лимит подключений ({connections_store.MAX_CONNECTIONS})")
                conn = {
                    "id": connections_store.new_id({c["id"] for c in items}),
                    "name": connections_store.guess_name(host),
                    "host": host,
                    "api_key": api_key,
                    "model": "",
                }
                items.append(conn)
            else:
                conn["api_key"] = api_key
            info = pipeline.connect(host, api_key)
            ok = bool(info.get("connected"))
            if ok:
                conn["model"] = pipeline.extractor.model_name
                active_id = conn["id"]
            saved = _persist_connections(items, active_id) if ok else settings_store.load_settings()
        _log_action(admin["username"], "connect", ok=ok, detail={"host_set": bool(request.host)})
        return info
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    except Exception as exc:
        _log_action(admin["username"], "connect", ok=False)
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc


@app.get("/api/connections")
def list_connections(_admin: dict = Depends(require_admin)) -> dict[str, object]:
    """Список сохранённых подключений (ключи маскированы) и активное состояние."""
    return _settings_response(settings_store.load_settings())


@app.post("/api/connections")
def create_connection(
    request: ConnectionCreateRequest, admin: dict = Depends(require_admin)
) -> dict[str, object]:
    """Добавляет подключение. Недоступный сервер тоже сохраняется в список."""
    try:
        with exclusive_extraction():
            items, active_id = _connections_state()
            if len(items) >= connections_store.MAX_CONNECTIONS:
                raise ValueError(f"Достигнут лимит подключений ({connections_store.MAX_CONNECTIONS})")
            host = pipeline.extractor._clean_host(request.host)
            if any(c["host"] == host for c in items):
                raise ValueError("Подключение с таким адресом уже есть в списке — измените его")
            api_key = request.api_key or ""
            name = (request.name or "").strip()[:connections_store.MAX_NAME_LENGTH]
            info = pipeline.connect(host, api_key) if request.activate else pipeline.probe(host, api_key)
            reachable = bool(info.get("connected"))
            models = list(info.get("models") or [])
            chosen = ""
            if reachable:
                chosen = next(
                    (m for m in (request.model, info.get("model")) if m and (not models or m in models)),
                    "",
                )
            elif request.model:
                chosen = request.model.strip()[:200]
            conn = {
                "id": connections_store.new_id({c["id"] for c in items}),
                "name": name or connections_store.guess_name(host),
                "host": host,
                "api_key": api_key,
                "model": chosen,
            }
            items.append(conn)
            warning = ""
            if request.activate and reachable:
                active_id = conn["id"]
                if chosen:
                    pipeline.set_model(chosen)
            elif request.activate and not active_id:
                # Первое подключение выбираем активным даже недоступным:
                # сервер поднимется позже, а адрес уже выбран.
                active_id = conn["id"]
                warning = (
                    f"{host} не ответил: подключение сохранено и выбрано активным, "
                    "оно заработает после запуска сервера."
                )
            elif request.activate:
                warning = f"{host} не ответил: подключение сохранено, активно прежнее."
            saved = _persist_connections(items, active_id)
        _log_action(admin["username"], "connection_add", ok=True,
                    detail={"host_set": True, "connected": reachable})
        return _settings_response(saved, warning)
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    except Exception as exc:
        _log_action(admin["username"], "connection_add", ok=False)
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc


@app.post("/api/connections/check")
def check_connection(
    request: ConnectionCheckRequest, admin: dict = Depends(require_admin)
) -> dict[str, object]:
    """Проверяет адрес без сохранения и переключения активного подключения."""
    try:
        with exclusive_extraction():
            api_key = request.api_key
            if api_key is None and request.id:
                items, _active = _connections_state()
                conn = connections_store.find(items, request.id)
                if conn is not None:
                    api_key = conn.get("api_key") or ""
            host = pipeline.extractor._clean_host(request.host)
            info = pipeline.probe(host, api_key or "")
        _log_action(admin["username"], "connection_check", ok=bool(info.get("connected")),
                    detail={"host_set": bool(request.host)})
        return {
            "connected": bool(info.get("connected")),
            "host": host,
            "backend": info.get("backend", ""),
            "model": info.get("model", ""),
            "models": info.get("models", []),
        }
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    except Exception as exc:
        _log_action(admin["username"], "connection_check", ok=False)
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc


@app.post("/api/connections/{connection_id}")
def update_connection(
    connection_id: str, request: ConnectionUpdateRequest, admin: dict = Depends(require_admin)
) -> dict[str, object]:
    """Правит подключение. Активное переподключается, остальные только проверяются."""
    try:
        with exclusive_extraction():
            items, active_id = _connections_state()
            conn = connections_store.find(items, connection_id)
            if conn is None:
                raise HTTPException(status_code=404, detail="Подключение не найдено")
            host = pipeline.extractor._clean_host(request.host) if request.host else conn["host"]
            if any(c["id"] != conn["id"] and c["host"] == host for c in items):
                raise ValueError("Другое подключение уже использует этот адрес")
            api_key = conn["api_key"] if request.api_key is None else (request.api_key or "")
            name = request.name.strip() if request.name is not None else conn["name"]
            is_active = conn["id"] == active_id
            changed = host != conn["host"] or api_key != conn["api_key"]
            reachable = True
            info: dict = {}
            warning = ""
            if changed:
                if is_active:
                    info = pipeline.connect(host, api_key)
                    reachable = bool(info.get("connected"))
                    if not reachable:
                        warning = (
                            f"{host} не ответил: параметры сохранены, но активным "
                            "останется прежний адрес до перезапуска компилятора."
                        )
                else:
                    info = pipeline.probe(host, api_key)
                    reachable = bool(info.get("connected"))
                    if not reachable:
                        warning = f"{host} не ответил: параметры сохранены."
            conn["host"] = host
            conn["api_key"] = api_key
            conn["name"] = (name or "").strip()[:connections_store.MAX_NAME_LENGTH] or connections_store.guess_name(host)
            if request.model is not None:
                model = request.model.strip()[:200]
                if model and is_active and reachable:
                    models = list(pipeline.available_models)
                    if models and model not in models:
                        raise ValueError("Выбранная модель не найдена на подключённом сервере")
                    pipeline.set_model(model)
                conn["model"] = model
            elif is_active and reachable and changed:
                models = list(pipeline.available_models)
                chosen = next(
                    (m for m in (conn.get("model"), info.get("model")) if m and (not models or m in models)),
                    "",
                )
                if chosen:
                    pipeline.set_model(chosen)
                conn["model"] = pipeline.extractor.model_name
            saved = _persist_connections(items, active_id)
        _log_action(admin["username"], "connection_update", ok=True, detail={"host_set": bool(request.host)})
        return _settings_response(saved, warning)
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    except Exception as exc:
        _log_action(admin["username"], "connection_update", ok=False)
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc


@app.post("/api/connections/{connection_id}/activate")
def activate_connection(connection_id: str, admin: dict = Depends(require_admin)) -> dict[str, object]:
    """Делает подключение активным. Недоступный сервер не переключается."""
    try:
        with exclusive_extraction():
            items, active_id = _connections_state()
            conn = connections_store.find(items, connection_id)
            if conn is None:
                raise HTTPException(status_code=404, detail="Подключение не найдено")
            if conn["id"] == active_id:
                saved = settings_store.load_settings()
            else:
                info = pipeline.connect(conn["host"], conn.get("api_key") or "")
                if not info.get("connected"):
                    raise ValueError(f"{conn['host']} не ответил — подключение не переключено")
                models = list(pipeline.available_models)
                chosen = next(
                    (m for m in (conn.get("model"), info.get("model")) if m and (not models or m in models)),
                    "",
                )
                if chosen:
                    pipeline.set_model(chosen)
                conn["model"] = pipeline.extractor.model_name
                active_id = conn["id"]
                saved = _persist_connections(items, active_id)
        _log_action(admin["username"], "connection_activate", ok=True)
        return _settings_response(saved)
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    except Exception as exc:
        _log_action(admin["username"], "connection_activate", ok=False)
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc


@app.delete("/api/connections/{connection_id}")
def delete_connection(connection_id: str, admin: dict = Depends(require_admin)) -> dict[str, object]:
    """Удаляет подключение. Если оно было активным — переключается на доступное."""
    try:
        with exclusive_extraction():
            items, active_id = _connections_state()
            conn = connections_store.find(items, connection_id)
            if conn is None:
                raise HTTPException(status_code=404, detail="Подключение не найдено")
            was_active = conn["id"] == active_id
            items = [c for c in items if c["id"] != conn["id"]]
            warning = ""
            if was_active:
                active_id = None
                for candidate in items:
                    info = pipeline.connect(candidate["host"], candidate.get("api_key") or "")
                    if not info.get("connected"):
                        continue
                    models = list(pipeline.available_models)
                    chosen = next(
                        (m for m in (candidate.get("model"), info.get("model")) if m and (not models or m in models)),
                        "",
                    )
                    if chosen:
                        pipeline.set_model(chosen)
                    candidate["model"] = pipeline.extractor.model_name
                    active_id = candidate["id"]
                    break
                if active_id is None:
                    # Ни одно из оставшихся не отвечает — возвращаемся к серверу по умолчанию.
                    _reset_pipeline_defaults()
                    warning = "Ни одно из оставшихся подключений не ответило: включён сервер по умолчанию."
            saved = _persist_connections(items, active_id)
        _log_action(admin["username"], "connection_delete", ok=True)
        return _settings_response(saved, warning)
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    except Exception as exc:
        _log_action(admin["username"], "connection_delete", ok=False)
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc


@app.post("/api/model")
def choose_model(request: ModelRequest, admin: dict = Depends(require_admin)) -> dict[str, str]:
    try:
        with exclusive_extraction():
            model = pipeline.set_model(request.model)
            # Выбранная модель запоминается в активном подключении.
            items, active_id = _connections_state()
            conn = connections_store.active(items, active_id)
            if conn is not None:
                conn["model"] = model
                _persist_connections(items, active_id)
            else:
                settings_store.save_settings({"model": model})
        _log_action(admin["username"], "model", ok=True)
        return {"model": model}
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
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
    """Сохраняет параметры запроса и (для совместимости) правит активное подключение.

    Интерфейс консоли управляет подключениями через /api/connections, а сюда
    приходит только блок параметров; поля host/api_key/model поддержаны для
    старых вызовов и относятся к активному подключению.
    """
    try:
        with exclusive_extraction():
            ex = pipeline.extractor
            items, active_id = _connections_state()
            conn = connections_store.active(items, active_id)
            warning = ""
            connected = True
            if request.params is not None:
                _apply_params_model(request.params)
            if request.host or request.api_key is not None or request.model:
                if conn is None:
                    host = ex._clean_host(request.host or config.OLLAMA_HOST)
                    conn = {
                        "id": connections_store.new_id({c["id"] for c in items}),
                        "name": connections_store.guess_name(host),
                        "host": host,
                        "api_key": request.api_key or "",
                        "model": "",
                    }
                    items.append(conn)
                    active_id = conn["id"]
                target_host = ex._clean_host(request.host) if request.host else conn["host"]
                target_key = conn["api_key"] if request.api_key is None else (request.api_key or "")
                if target_host != conn["host"] or target_key != conn["api_key"]:
                    info = pipeline.connect(target_host, target_key)
                    connected = bool(info.get("connected"))
                    if not connected:
                        warning = (
                            f"Сервер {target_host} не ответил: настройки сохранены, "
                            "подключение к нему произойдёт после запуска сервера "
                            "и перезапуска компилятора."
                        )
                conn["host"] = target_host
                conn["api_key"] = target_key
                if request.model:
                    model = request.model.strip()
                    if connected:
                        if pipeline.available_models and model not in pipeline.available_models:
                            raise ValueError("Выбранная модель не найдена на подключённом сервере")
                        ex.model_name = model
                        conn["model"] = model
                elif connected and not conn.get("model"):
                    conn["model"] = ex.model_name
            saved = settings_store.save_settings(
                connections_store.payload(items, active_id),
                drop=("host", "api_key", "model") if items else (),
            )
        _log_action(admin["username"], "settings_save", ok=True,
                    detail={"host_set": bool(request.host), "connected": connected})
        return _settings_response(saved, warning)
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
            _reset_pipeline_defaults()
            ex = pipeline.extractor
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


def _check_model_permission(model: Optional[str], user: dict) -> None:
    if model is not None and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Выбор модели доступен только администратору")


@app.post("/api/extract/{request_id}/cancel")
def cancel_extract(request_id: str, user: dict = Depends(enforce_change)) -> dict[str, str]:
    if not operation_store.cancel(request_id, user["id"]):
        raise HTTPException(status_code=404, detail="Активный запрос не найден")
    return {"status": "cancelling"}


@app.post("/api/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest, user: dict = Depends(enforce_change)) -> ExtractResponse:
    _check_model_permission(request.model, user)
    token = set_correlation(uuid.uuid4().hex[:12])
    started = time.monotonic()
    try:
        with exclusive_extraction():
            with operation_store.track(request.request_id or uuid.uuid4().hex, user["id"]) as cancellation:
                data = pipeline.extract(
                    request.image_base64,
                    request.target_columns,
                    request.filename,
                    request.ocr_priority,
                    request.detector,
                    request.model,
                    cancellation=cancellation,
                )
        _log_action(user["username"], "extract", ok=True,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    detail={"mode": request.source})
        return ExtractResponse(data=data)
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    except ExtractionCancelled as exc:
        _log_action(user["username"], "extract", ok=False,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    detail={"mode": request.source, "cancelled": True})
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("extract: error=%s", type(exc).__name__)
        _log_action(user["username"], "extract", ok=False,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    detail={"mode": request.source})
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc
    finally:
        reset_correlation(token)


@app.post("/api/preview")
def preview_document(request: PreviewRequest, user: dict = Depends(enforce_change)) -> Response:
    """Лёгкий JPEG-превью страницы документа для окна проверки."""
    preview = make_preview(request.image_base64, request.page)
    if preview is None:
        raise HTTPException(status_code=422, detail="Не удалось построить превью документа")
    return Response(content=preview, media_type="image/jpeg")


@app.post("/api/batch/duplicates")
def batch_duplicates(
    request: DuplicatesRequest, user: dict = Depends(enforce_change)
) -> list[list[Dict[str, Any]]]:
    """Пометки дублей внутри пачки; результат выровнен по records."""
    if request.filenames and len(request.filenames) != len(request.records):
        raise HTTPException(status_code=422, detail="filenames должны соответствовать records")
    try:
        return ExcelManager.find_in_memory_duplicates(request.records, request.filenames or None)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=friendly_error(exc)) from exc


@app.post("/api/batch", status_code=202, response_model=BatchJobStarted)
def batch_start(
    request: BatchExtractRequest, user: dict = Depends(enforce_change)
) -> BatchJobStarted:
    _check_model_permission(request.model, user)
    try:
        job_id = job_store.start(request, pipeline, username=user["username"], owner_id=user["id"])
    except JobBusy as exc:
        raise _http_409_job_busy(exc) from exc
    _log_action(user["username"], "batch_start", ok=True,
                detail={"files": len(request.documents)})
    return BatchJobStarted(job_id=job_id, total=len(request.documents))


def _owned_job(job_id: str, user: dict) -> Job:
    """Документы доступны только создателю задания, включая администраторов."""
    job = job_store.get(job_id)
    if job is None or job.owner_id != user["id"]:
        raise HTTPException(status_code=404, detail="Джоб не найден")
    return job


@app.get("/api/batch/{job_id}", response_model=BatchJobState)
def batch_status(job_id: str, user: dict = Depends(enforce_change)) -> BatchJobState:
    job = _owned_job(job_id, user)
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


@app.get("/api/batch/{job_id}/{index}/preview")
def batch_preview(job_id: str, index: int, user: dict = Depends(enforce_change)) -> Response:
    """Превью исходного скана для карточки проверки (только завершённый джоб)."""
    job = _owned_job(job_id, user)
    if job.status not in _FINISHED_STATUSES:
        raise HTTPException(status_code=409, detail="Джоб ещё выполняется")
    if not 0 <= index < job.total:
        raise HTTPException(status_code=404, detail="Файл не найден")
    preview = job.previews.get(index)
    if preview is None:
        raise HTTPException(status_code=404, detail="Превью недоступно")
    return Response(content=preview, media_type="image/jpeg")


@app.post("/api/batch/{job_id}/cancel")
def batch_cancel(
    job_id: str, user: dict = Depends(enforce_change)
) -> dict[str, str]:
    job = _owned_job(job_id, user)
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

    # Без собранного фронтенда корень отдавал бы голый 404, и это выглядело
    # как «у пользователя нет прав». Показываем, что именно нужно сделать.
    _NOT_BUILT_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DocAI Assistant — интерфейс не собран</title>
<style>
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0f172a;color:#e2e8f0;
font:16px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}
div{max-width:34rem;padding:2rem;background:#1b2436;border:1px solid #2c3a52;border-radius:14px}
h1{margin:0 0 .75rem;font-size:1.3rem}p{margin:.6rem 0}code{background:#0f172a;padding:.15rem .4rem;border-radius:6px}
a{color:#7cb3ff}
</style></head><body><div>
<h1>Интерфейс не собран</h1>
<p>Сервер работает, но папки <code>web/dist</code> нет, поэтому рабочее окно оцифровки
отдать нечем. Это не про права доступа — интерфейс просто не собран.</p>
<p>На Windows достаточно запустить <code>start.bat</code>: он соберёт интерфейс сам.
Вручную: <code>cd web</code>, затем <code>npm install</code> и <code>npm run build</code>,
после чего перезапустите сервер.</p>
<p>Консоль администратора доступна уже сейчас: <a href="/settings">/settings</a>.</p>
</div></body></html>"""

    @app.get("/", include_in_schema=False)
    def web_not_built() -> HTMLResponse:
        """Понятная заглушка вместо 404, когда web/dist ещё не собран."""
        return HTMLResponse(_NOT_BUILT_PAGE, status_code=503)
