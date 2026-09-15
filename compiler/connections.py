"""Сохранённые подключения к серверам ИИ (несколько профилей в settings.json).

Подключение — это именованный набор «адрес + API-ключ + выбранная модель».
Активным может быть только одно: его адрес и модель применяются при старте и
при переключении. Ключи лежат локально в settings.json (.gitignore) и никогда
не отдаются наружу в открытом виде — только маска и признак «ключ задан».

Модуль намеренно не зависит от extractor/pipeline: только нормализация данных
и миграция старых настроек с одиночным адресом.
"""
from __future__ import annotations

import secrets
from typing import Any, Dict, List, Optional, Tuple

MAX_CONNECTIONS = 20
MAX_NAME_LENGTH = 60
DEFAULT_NAME = "Сервер ИИ"
MIGRATED_ID = "conn-default"
ID_PREFIX = "conn-"


def guess_name(host: str) -> str:
    """Понятное имя по адресу: локальные серверы узнаём по порту, облака по домену."""
    h = str(host or "").lower()
    if ":11434" in h:
        return "Ollama"
    if ":1234" in h:
        return "LM Studio"
    if "siliconflow" in h:
        return "SiliconFlow"
    if "generativelanguage.googleapis.com" in h:
        return "Google AI Studio"
    if "seekai" in h:
        return "SeekAI"
    return DEFAULT_NAME


def new_id(existing: set[str]) -> str:
    """Короткий уникальный идентификатор подключения."""
    for _ in range(100):
        candidate = ID_PREFIX + secrets.token_hex(4)
        if candidate not in existing:
            return candidate
    raise RuntimeError("Не удалось создать идентификатор подключения")


def _clean_text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def normalize_connection(item: Dict[str, Any], fallback_id: str = "") -> Optional[Dict[str, Any]]:
    """Приводит запись к каноническому виду; None — если адреса нет."""
    if not isinstance(item, dict):
        return None
    host = _clean_text(item.get("host"), 255)
    if not host:
        return None
    name = _clean_text(item.get("name"), MAX_NAME_LENGTH) or guess_name(host)
    return {
        "id": _clean_text(item.get("id"), 64) or fallback_id,
        "name": name,
        "host": host,
        "api_key": str(item.get("api_key") or "").strip(),
        "model": _clean_text(item.get("model"), 200),
    }


def from_saved(saved: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Читает подключения из settings.json, мигрируя старый одиночный адрес.

    Возвращает (список подключений, id активного). Миграция не пишет файл —
    он обновится при первом сохранении, поэтому вызывать функцию безопасно.
    """
    saved = saved or {}
    items: List[Dict[str, Any]] = []
    raw = saved.get("connections")
    if isinstance(raw, list) and raw:
        known: set[str] = set()
        for index, entry in enumerate(raw):
            conn = normalize_connection(entry, fallback_id=f"{ID_PREFIX}legacy{index}")
            if conn is None:
                continue
            if conn["id"] in known:
                conn["id"] = new_id(known)
            known.add(conn["id"])
            items.append(conn)
    elif saved.get("host"):
        legacy = normalize_connection(
            {
                "id": MIGRATED_ID,
                "name": "",
                "host": saved.get("host"),
                "api_key": saved.get("api_key"),
                "model": saved.get("model"),
            }
        )
        if legacy is not None:
            items.append(legacy)
    items = items[:MAX_CONNECTIONS]

    active_id = _clean_text(saved.get("active_connection_id"), 64)
    if not any(conn["id"] == active_id for conn in items):
        active_id = items[0]["id"] if items else ""
    return items, active_id or None


def find(items: List[Dict[str, Any]], connection_id: str) -> Optional[Dict[str, Any]]:
    for conn in items:
        if conn["id"] == connection_id:
            return conn
    return None


def active(items: List[Dict[str, Any]], active_id: Optional[str]) -> Optional[Dict[str, Any]]:
    return find(items, active_id or "")


def payload(items: List[Dict[str, Any]], active_id: Optional[str]) -> Dict[str, Any]:
    """Фрагмент settings.json с подключениями (старые поля хоста удаляются)."""
    return {
        "connections": items,
        "active_connection_id": active_id or "",
    }
