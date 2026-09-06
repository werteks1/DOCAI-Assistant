"""Хранилище учётных записей, сессий и журнала активности (SQLite, stdlib).

Пароли хранятся как PBKDF2-HMAC-SHA256 со случайной солью на пользователя —
никаких новых зависимостей, только `hashlib`/`hmac`/`secrets`/`sqlite3`.
Сессии — непрозрачные токены (`secrets.token_urlsafe`), TTL скользящий.

Правила безопасности:
* В журнал активности не попадают ПД: только `username`, `action`, время,
  счётчики и служебные поля (модель/бэкенд, ok, время выполнения). Никаких
  имён файлов, значений полей и содержимого документов (см. logging_utils.py).
* Защита от перебора: in-memory счётчик неудачных попыток на пользователя;
  после 5 подряд в окне 15 минут вход блокируется до истечения окна.
* Первый вход админа — seed `admin/admin` с `must_change=1` (сменить пароль).
* Каждое обращение — короткое соединение (`PRAGMA journal_mode=WAL`,
  `busy_timeout`), чтобы потоки FastAPI и фоновый поток джоба не конфликтовали.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import config

PBKDF2_ITERATIONS = 600_000
_SALT_BYTES = 16
TOKEN_BYTES = 32

# Защита от перебора: >= 5 неудач подряд в этом окне блокируют вход.
_MAX_FAILED = 5
_LOCK_WINDOW_S = 15 * 60

# Схема создаётся идемпотентно при старте.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    role TEXT NOT NULL CHECK (role IN ('admin', 'operator')),
    password_salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    must_change INTEGER NOT NULL DEFAULT 1,
    disabled INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    last_login_at TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    username TEXT NOT NULL,
    action TEXT NOT NULL,
    ok INTEGER NOT NULL DEFAULT 1,
    elapsed_ms INTEGER,
    model TEXT,
    backend TEXT,
    detail_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_activity_ts ON activity(ts);
CREATE INDEX IF NOT EXISTS idx_activity_username ON activity(username);
"""


@dataclass
class AuthResult:
    """Итог попытки входа."""
    user: Optional[Dict[str, Any]] = None
    locked: bool = False   # слишком много неудачных попыток — подождите


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.DOCIA_DB), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _hash_password(password: str, salt_hex: Optional[str] = None) -> tuple[str, str]:
    """Возвращает (salt_hex, hash_hex). Без соли генерирует новую."""
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return salt.hex(), digest.hex()


def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(digest.hex(), hash_hex)


def _row_to_user(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "must_change": bool(row["must_change"]),
        "disabled": bool(row["disabled"]),
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }


# --- Инициализация ----------------------------------------------------

def init_db() -> None:
    """Создаёт схему и сидирует учётку администратора, если пользователей нет."""
    config.DOCIA_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
        _purge_expired_sessions(conn)
        _seed_admin_if_empty(conn)
    finally:
        conn.close()


def _seed_admin_if_empty(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT 1 FROM users LIMIT 1").fetchone()
    if row is not None:
        return
    salt, digest = _hash_password("admin")
    conn.execute(
        "INSERT INTO users (username, role, password_salt, password_hash,"
        " must_change, disabled, created_at)"
        " VALUES (?, ?, ?, ?, 1, 0, ?)",
        ("admin", "admin", salt, digest, _iso(_now())),
    )
    conn.commit()


def _purge_expired_sessions(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM sessions WHERE expires_at < ?", (_iso(_now()),))
    conn.commit()


def _find_user_row(conn: sqlite3.Connection, username: str) -> Optional[sqlite3.Row]:
    """Ищет пользователя по имени без учёта регистра (в т.ч. кириллицы).

    Пользователей немного (школьный LAN), поэтому таблица сканируется целиком
    и сравнивается через str.casefold — SQLite NOCASE складывает только ASCII.
    """
    needle = (username or "").strip().casefold()
    if not needle:
        return None
    for row in conn.execute("SELECT * FROM users"):
        if (row["username"] or "").strip().casefold() == needle:
            return row
    return None


# --- Пользователи -----------------------------------------------------

def _get_user_by_id(conn: sqlite3.Connection, user_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        return _get_user_by_id(conn, user_id)
    finally:
        conn.close()


def list_users() -> List[Dict[str, Any]]:
    """Все учётки без парольных полей (для админ-консоли)."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, username, role, must_change, disabled, created_at, last_login_at"
            " FROM users ORDER BY id"
        ).fetchall()
        return [
            {
                "id": r["id"],
                "username": r["username"],
                "role": r["role"],
                "must_change": bool(r["must_change"]),
                "disabled": bool(r["disabled"]),
                "created_at": r["created_at"],
                "last_login_at": r["last_login_at"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def _username_ok(username: str) -> bool:
    username = (username or "").strip()
    return 1 <= len(username) <= 64 and all(
        ch.isalnum() or ch in "._-" for ch in username
    )


def create_user(username: str, password: str, role: str, must_change: bool = True) -> Dict[str, Any]:
    """Создаёт учётку. Raises ValueError при некорректных данных или дубле."""
    username = (username or "").strip()
    if not _username_ok(username):
        raise ValueError(
            "Логин: от 1 до 64 символов — буквы, цифры, точка, дефис, подчёркивание."
        )
    if len(password or "") < 4:
        raise ValueError("Пароль слишком короткий — минимум 4 символа.")
    if role not in ("admin", "operator"):
        raise ValueError("Роль должна быть admin или operator.")
    conn = _connect()
    try:
        if _find_user_row(conn, username) is not None:
            raise ValueError("Пользователь с таким логином уже существует.")
        salt, digest = _hash_password(password)
        cur = conn.execute(
            "INSERT INTO users (username, role, password_salt, password_hash,"
            " must_change, disabled, created_at)"
            " VALUES (?, ?, ?, ?, ?, 0, ?)",
            (username, role, salt, digest, 1 if must_change else 0, _iso(_now())),
        )
        conn.commit()
        user = _get_user_by_id(conn, cur.lastrowid)
        assert user is not None
        return user
    finally:
        conn.close()


def verify_user_password(user_id: int, password: str) -> bool:
    """Проверяет пароль пользователя по id (для смены «своего» пароля)."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT password_salt, password_hash FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row is None:
            return False
        return _verify_password(password or "", row["password_salt"], row["password_hash"])
    finally:
        conn.close()


def set_password(
    user_id: int,
    new_password: str,
    must_change: bool = False,
    revoke_sessions: bool = True,
) -> None:
    """Смена пароля (админом или владельцем). Raises ValueError при коротком пароле.

    При `revoke_sessions=True` все прежние сессии пользователя отзываются
    (сброс пароля админом). Для самостоятельной смены пароля текущую сессию
    можно сохранить (revoke_sessions=False), чтобы не разлогинивать пользователя.
    """
    if len(new_password or "") < 4:
        raise ValueError("Пароль слишком короткий — минимум 4 символа.")
    salt, digest = _hash_password(new_password)
    conn = _connect()
    try:
        conn.execute(
            "UPDATE users SET password_salt = ?, password_hash = ?, must_change = ?"
            " WHERE id = ?",
            (salt, digest, 1 if must_change else 0, user_id),
        )
        # Смена пароля снимает блокировку перебора и (по умолчанию) старые сессии.
        _clear_failures(_username_of(conn, user_id))
        if revoke_sessions:
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def _username_of(conn: sqlite3.Connection, user_id: int) -> str:
    row = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
    return row["username"] if row else ""


def set_disabled(user_id: int, disabled: bool) -> None:
    conn = _connect()
    try:
        conn.execute(
            "UPDATE users SET disabled = ? WHERE id = ?", (1 if disabled else 0, user_id)
        )
        if disabled:
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def delete_user(user_id: int) -> None:
    """Удаляет учётку и её сессии (каскадом)."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


# --- Вход (с защитой от перебора) -------------------------------------

_failures: Dict[str, List[float]] = {}
_failures_lock = threading.Lock()


def _clear_failures(username: str) -> None:
    with _failures_lock:
        _failures.pop(username, None)


def _prune_failures(username: str) -> int:
    """Возвращает число неудач username в текущем окне."""
    now = time.monotonic()
    with _failures_lock:
        attempts = _failures.get(username, [])
        attempts = [t for t in attempts if now - t < _LOCK_WINDOW_S]
        _failures[username] = attempts
        return len(attempts)


def _record_failure(username: str) -> None:
    with _failures_lock:
        now = time.monotonic()
        attempts = _failures.get(username, [])
        attempts = [t for t in attempts if now - t < _LOCK_WINDOW_S]
        attempts.append(now)
        _failures[username] = attempts


def verify_login(username: str, password: str) -> AuthResult:
    """Проверяет логин/пароль. Возвращает user или причину отказа."""
    username = (username or "").strip()
    if not username:
        return AuthResult(locked=False)
    if _prune_failures(username) >= _MAX_FAILED:
        return AuthResult(locked=True)
    conn = _connect()
    try:
        row = _find_user_row(conn, username)
        if row is None:
            _record_failure(username)
            return AuthResult(locked=False)
        if not _verify_password(password or "", row["password_salt"], row["password_hash"]):
            _record_failure(username)
            return AuthResult(locked=False)
        user = _row_to_user(row)
        if user["disabled"]:
            _record_failure(username)
            return AuthResult(locked=False)
        _clear_failures(username)
        conn.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (_iso(_now()), user["id"]),
        )
        conn.commit()
        return AuthResult(user=user)
    finally:
        conn.close()


# --- Сессии -----------------------------------------------------------

def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(TOKEN_BYTES)
    now = _now()
    expires = now + timedelta(hours=config.SESSION_TTL_HOURS)
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at)"
            " VALUES (?, ?, ?, ?)",
            (token, user_id, _iso(now), _iso(expires)),
        )
        conn.commit()
        return token
    finally:
        conn.close()


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Возвращает пользователя по токену; скользящее продление TTL."""
    if not token:
        return None
    now = _now()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id"
            " WHERE s.token = ? AND s.expires_at > ?",
            (token, _iso(now)),
        ).fetchone()
        if row is None:
            return None
        if row["disabled"]:
            return None
        expires = now + timedelta(hours=config.SESSION_TTL_HOURS)
        conn.execute(
            "UPDATE sessions SET expires_at = ? WHERE token = ?", (_iso(expires), token)
        )
        conn.commit()
        return _row_to_user(row)
    finally:
        conn.close()


def delete_session(token: str) -> None:
    if not token:
        return
    conn = _connect()
    try:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()


# --- Журнал активности (PII-safe) -------------------------------------

def log_activity(
    username: str,
    action: str,
    *,
    ok: bool = True,
    elapsed_ms: Optional[int] = None,
    model: Optional[str] = None,
    backend: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> None:
    """Пишет строку журнала. `detail` — только служебные счётчики/флаги, не ПД."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO activity (ts, username, action, ok, elapsed_ms, model,"
            " backend, detail_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _iso(_now()),
                username or "-",
                action,
                1 if ok else 0,
                elapsed_ms,
                model,
                backend,
                json.dumps(detail, ensure_ascii=False) if detail else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# --- Выборки для статистики ------------------------------------------

def activity_summary(limit_days: int = 30) -> Dict[str, Any]:
    """Агрегаты по пользователям и по дням + последние записи (без detail)."""
    since = _iso(_now() - timedelta(days=limit_days))
    conn = _connect()
    try:
        users = conn.execute(
            "SELECT username, COUNT(*) AS actions,"
            " SUM(CASE WHEN ok THEN 1 ELSE 0 END) AS ok_count,"
            " SUM(CASE WHEN ok THEN 0 ELSE 1 END) AS error_count,"
            " CAST(AVG(elapsed_ms) AS INTEGER) AS avg_elapsed_ms,"
            " MAX(ts) AS last_ts"
            " FROM activity WHERE ts >= ? GROUP BY username ORDER BY actions DESC",
            (since,),
        ).fetchall()
        days = conn.execute(
            "SELECT substr(ts, 1, 10) AS day, COUNT(*) AS actions,"
            " SUM(CASE WHEN ok THEN 1 ELSE 0 END) AS ok_count"
            " FROM activity WHERE ts >= ? GROUP BY day ORDER BY day",
            (since,),
        ).fetchall()
        recent = conn.execute(
            "SELECT ts, username, action, ok, elapsed_ms, model, backend"
            " FROM activity ORDER BY id DESC LIMIT 50"
        ).fetchall()
        return {
            "users": [dict(r) for r in users],
            "days": [dict(r) for r in days],
            "recent": [dict(r) for r in recent],
        }
    finally:
        conn.close()
