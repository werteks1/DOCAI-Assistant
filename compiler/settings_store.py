"""Хранилище настроек модели/сервера (settings.json в корне проекта).

Файл держит адрес сервера ИИ, опциональный API-ключ OpenAI-совместимого
сервера и параметры запроса VLM (температура, лимит токенов, таймаут,
дополнительный системный промпт). Это источник по умолчанию для компилятора:
при старте настройки применяются до первого запроса.

Правила безопасности:
* API-ключ хранится только локально в файле, в .gitignore, и не попадает в логи.
* GET-эндпоинты возвращают ключ только в маскированном виде (bool + хвост).
* Адрес сервера ИИ не ограничен: допускается любой OpenAI-совместимый сервер.
  Администратор сам отвечает за то, доверяет ли он выбранному серверу.
"""
import json
import os
from typing import Dict, Any

import config

SETTINGS_FILE = config.BASE_DIR / "settings.json"


def load_settings() -> Dict[str, Any]:
    """Возвращает сохранённые настройки ({} если файла нет или он битый)."""
    if not SETTINGS_FILE.exists():
        return {}
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(data: Dict[str, Any], drop: tuple = ()) -> Dict[str, Any]:
    """Атомарно записывает settings.json, сливая поверх уже сохранённого.

    `drop` перечисляет ключи, которые нужно удалить (например, старые
    одиночные host/api_key после перехода на список подключений).
    """
    merged = dict(load_settings())
    for key, value in data.items():
        merged[key] = value
    for key in drop:
        merged.pop(key, None)
    tmp = SETTINGS_FILE.with_name(SETTINGS_FILE.name + ".tmp")
    tmp.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        # В файле лежат API-ключи: доступ только владельцу.
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, SETTINGS_FILE)
    return merged


def remove_settings() -> None:
    """Удаляет файл настроек (сброс к заводским значениям по умолчанию)."""
    try:
        SETTINGS_FILE.unlink()
    except FileNotFoundError:
        pass
