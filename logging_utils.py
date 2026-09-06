# -*- coding: utf-8 -*-
"""PII-safe логирование DocAI Assistant.

Политика приватности:
- В лог никогда не попадают значения распознанных полей, содержимое документов
  и имена исходных файлов (имя файла может содержать фамилию ученика).
- Логируются только нейтральные события: этап, индекс файла, число страниц,
  затраченное время, backend/модель, доступность PaddleOCR, тип ошибки.

Каждая запись несёт correlation id (contextvar) — для /api/extract это
случайный/входящий X-Request-ID, для пакетного джоба — его job_id.
"""
from __future__ import annotations

import contextvars
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

import config

_LOGGER_NAME = "docai"
_configured = False

# Correlation id: подставляется фильтром в каждую запись из текущего контекста.
CORRELATION_ID: contextvars.ContextVar = contextvars.ContextVar(
    "docai_correlation_id", default="-"
)


def set_correlation(correlation_id: Optional[str]) -> contextvars.Token:
    """Устанавливает correlation id для текущего контекста (потока/запроса)."""
    return CORRELATION_ID.set(str(correlation_id or "-"))


def reset_correlation(token: contextvars.Token) -> None:
    CORRELATION_ID.reset(token)


class _CorrelationFilter(logging.Filter):
    """Добавляет атрибут correlation_id в каждую запись."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = CORRELATION_ID.get()
        return True


def _resolve_level() -> int:
    level = config.get_log_level().upper()
    resolved = logging.getLevelName(level)
    if isinstance(resolved, int):  # имя уровня валидно
        return resolved
    return logging.INFO


def init_logging(force: bool = False) -> None:
    """Настраивает логгер один раз: stderr + опционально вращаемый файл."""
    global _configured
    if _configured and not force:
        return

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(_resolve_level())
    logger.propagate = False  # не дублируем записи в корневом логгере

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(correlation_id)s] %(name)s: %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(_CorrelationFilter())
    logger.addHandler(stream_handler)

    log_file = config.get_log_file()
    if log_file:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_CorrelationFilter())
        logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str = "") -> logging.Logger:
    """Возвращает дочерний логгер docai[.name], настраивая логгер при первом вызове."""
    init_logging()
    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)
