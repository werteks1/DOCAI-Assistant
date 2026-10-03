import logging

import uvicorn

import config


class _HealthAccessFilter(logging.Filter):
    """Не засоряет консоль access-логами частых проверок /health."""

    def filter(self, record: logging.LogRecord) -> bool:
        return " /health " not in record.getMessage()


if __name__ == "__main__":
    logging.getLogger("uvicorn.access").addFilter(_HealthAccessFilter())
    # LAN-режим: BIND_HOST по умолчанию 0.0.0.0 — компилятор отдаёт и API,
    # и собранный фронтенд (web/dist) всем ПК школьной сети на :8000.
    # Для локальной разработки: DOCIA_BIND_HOST=127.0.0.1 python -m compiler.
    # ВАЖНО: строго один воркер. Всё критичное состояние — общий pipeline,
    # EXTRACTION_LOCK, очередь пакетов (JobStore) и счётчики блокировки входа —
    # живёт в памяти процесса. Под несколькими воркерами лок перестал бы
    # сериализовать непотокобезопасный экстрактор, статус пакета опрашивался бы
    # не на том воркере (404), а блокировку перебора можно было бы обойти,
    # попав на другой процесс. Масштаб школьной LAN одного воркера не требует.
    uvicorn.run(
        "compiler.api:app",
        host=config.BIND_HOST,
        port=config.BIND_PORT,
        reload=False,
        workers=1,
    )
