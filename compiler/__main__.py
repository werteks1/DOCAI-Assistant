import uvicorn

import config


if __name__ == "__main__":
    # LAN-режим: BIND_HOST по умолчанию 0.0.0.0 — компилятор отдаёт и API,
    # и собранный фронтенд (web/dist) всем ПК школьной сети на :8000.
    # Для локальной разработки: DOCIA_BIND_HOST=127.0.0.1 python -m compiler.
    uvicorn.run(
        "compiler.api:app",
        host=config.BIND_HOST,
        port=config.BIND_PORT,
        reload=False,
    )
