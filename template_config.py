# -*- coding: utf-8 -*-
"""Загрузка шаблонов бланков из templates.json.

Один бланк = один шаблон: список полей, синонимы печатных меток (для PaddleOCR)
и резервная геометрия строк (для полного прохода VLM без детектора).

Политика устойчивости:
- Встроенный DEFAULT_TEMPLATE гарантирует работу при отсутствии/порче файла;
- значение активного шаблона выбирается env DOCIA_TEMPLATE (или первым в файле);
- стандартный бланк всегда дополняется встроенной геометрией, остальные шаблоны
  используют только то, что указано в файле (для нового бланка нужно добавить
  блок в templates.json — без правки кода).

Значения геометрии стандартного бланка тюнились вручную и переносятся
побайтово: не меняйте их без повторного замера CER/WER.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import config
from logging_utils import get_logger

logger = get_logger("template")

DEFAULT_TEMPLATE_NAME = "Заявление о приеме (1/10 класс)"

# Встроенный шаблон стандартного бланка (фолбэк, если templates.json недоступен).
DEFAULT_TEMPLATE: Dict[str, Any] = {
    "fields": [
        "Дата подачи заявления",
        "ФИО поступающего ученика",
        "Дата рождения ребенка",
        "Класс / профиль обучения",
        "ФИО родителя / заявителя",
        "Паспортные данные",
        "Адрес регистрации / проживания",
        "Контактный телефон",
        "СНИЛС поступающего",
        "Особые отметки / льготы",
    ],
    "label_aliases": {
        "Дата подачи заявления": ["дата подачи", "дата подачи заявления"],
        "ФИО поступающего ученика": ["фио поступающего", "фамилия, имя, отчество поступающего", "поступающего (ребенка)"],
        "Дата рождения ребенка": ["дата и место рождения", "дата рождения ребенка", "дата рождения"],
        "Класс / профиль обучения": ["желаемый класс", "профиль обучения", "класс"],
        "ФИО родителя / заявителя": ["фио родителя", "фио заявителя", "родителя (законного представителя)"],
        "Паспортные данные": ["паспортные данные", "паспорт"],
        "Адрес регистрации / проживания": ["адрес регистрации", "фактического проживания", "адрес проживания"],
        "Контактный телефон": ["контактный номер телефона", "контактный телефон", "телефон для связи"],
        "СНИЛС поступающего": ["страховой номер", "снилс", "лицевого счета"],
        "Особые отметки / льготы": ["дополнительные сведения", "наличие льгот", "особые отметки"],
    },
    # Вертикальные доли листа под значением каждой метки (y1, y2 в долях высоты).
    "fallback_y": {
        "Дата подачи заявления": [0.23, 0.30],
        "ФИО поступающего ученика": [0.29, 0.35],
        "Дата рождения ребенка": [0.35, 0.41],
        "Класс / профиль обучения": [0.40, 0.47],
        "ФИО родителя / заявителя": [0.46, 0.52],
        "Паспортные данные": [0.51, 0.58],
        "Адрес регистрации / проживания": [0.57, 0.64],
        "Контактный телефон": [0.63, 0.70],
        "СНИЛС поступающего": [0.69, 0.76],
        "Особые отметки / льготы": [0.75, 0.83],
    },
    # Горизонтальные границы полос для полей, где метка узкая.
    "fallback_x": {
        "Дата подачи заявления": [0.08, 0.58],
        "Паспортные данные": [0.08, 0.96],
        "СНИЛС поступающего": [0.08, 0.78],
        "Контактный телефон": [0.08, 0.72],
    },
    # Полоса крупной даты слева от подписи внизу документа.
    "signature_date_y": [0.80, 0.90],
}


def _templates_path() -> Path:
    return config.BASE_DIR / "templates.json"


def _read_entries() -> Dict[str, Dict[str, Any]]:
    """Читает templates.json; при ошибке возвращает {} и пишет warning."""
    path = _templates_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {name: entry for name, entry in data.items() if isinstance(entry, dict)}
    except (OSError, ValueError) as exc:
        logger.warning("templates.json не прочитан (%s); используется встроенный шаблон", type(exc).__name__)
    return {}


def _pair(value: Any) -> Optional[tuple]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _as_str_list(value: Any) -> list:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _labels_dict(value: Any) -> Dict[str, tuple]:
    out = {}
    if isinstance(value, dict):
        for field, aliases in value.items():
            cleaned = _as_str_list(aliases)
            if field and cleaned:
                out[field] = tuple(cleaned)
    return out


def load_template(name: Optional[str] = None) -> Dict[str, Any]:
    """Возвращает нормализованный шаблон с полями, aliases и геометрией.

    name=None → env DOCIA_TEMPLATE, иначе первый шаблон в файле,
    иначе DEFAULT_TEMPLATE_NAME.
    """
    entries = _read_entries()
    chosen = name or config.get_template_name()
    if not chosen:
        chosen = entries and next(iter(entries)) or DEFAULT_TEMPLATE_NAME

    raw = entries.get(chosen)
    if chosen == DEFAULT_TEMPLATE_NAME:
        # Стандартный бланк всегда дополняется проверенной встроенной геометрией.
        template = deepcopy(DEFAULT_TEMPLATE)
        if raw:
            fields = _as_str_list(raw.get("fields"))
            if fields:
                template["fields"] = fields
            aliases = _labels_dict(raw.get("label_aliases"))
            if aliases:
                template["label_aliases"].update(aliases)
            for key in ("fallback_y", "fallback_x"):
                if isinstance(raw.get(key), dict):
                    for field, pair in raw[key].items():
                        normalized = _pair(pair)
                        if normalized:
                            template[key][field] = normalized
            pair = _pair(raw.get("signature_date_y"))
            if pair:
                template["signature_date_y"] = pair
        return template

    if raw is None:
        logger.warning("Шаблон %r не найден в templates.json; применён стандартный", chosen)
        return deepcopy(DEFAULT_TEMPLATE)

    return {
        "fields": _as_str_list(raw.get("fields")),
        "label_aliases": _labels_dict(raw.get("label_aliases")),
        "fallback_y": {f: _pair(p) for f, p in raw.get("fallback_y", {}).items() if _pair(p)}
        if isinstance(raw.get("fallback_y"), dict) else {},
        "fallback_x": {f: _pair(p) for f, p in raw.get("fallback_x", {}).items() if _pair(p)}
        if isinstance(raw.get("fallback_x"), dict) else {},
        "signature_date_y": _pair(raw.get("signature_date_y")),
    }


def list_templates() -> list:
    """Имена шаблонов из templates.json (для интерфейса/документации)."""
    return list(_read_entries().keys())
