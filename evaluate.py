#!/usr/bin/env python
"""Harness замера точности распознавания полей: CER/WER по размеченному набору.

Раскладка данных (private, в .gitignore):
    test_samples/<case>.<png|jpg|pdf>     — обезличенный скан бланка
    test_samples/labels/<case>.json        — эталон:
        {"fields": {"ФИО поступающего ученика": "Иванов Иван", ...}}

Метрики считаются по нормализованным значениям:
- CER  = редакционное расстояние Левенштейна / len(эталон)  (по символам)
- WER  = то же по словам
Для СНИЛС/телефона/паспорта сравниваются только цифры; для остальных —
casefold + ё→е + схлопывание пробелов.

Запуск (нужен доступный VLM-хост, см. config):
    venv/bin/python evaluate.py [--limit N] [--out report.md]

Требуется доступный сервер ИИ; тестовые образцы этим скриптом не меняются.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from compiler.pipeline import CompilerPipeline

SAMPLES_DIR = Path("test_samples")
LABELS_DIR = SAMPLES_DIR / "labels"
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".pdf"}


# --------------------------------------------------------------------------
# Нормализация и метрики (stdlib, без новых зависимостей)
# --------------------------------------------------------------------------

def _norm_text(value: Any) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    return " ".join(text.split())


def _norm_digits(value: Any) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _norm_field(field: str, value: Any) -> str:
    lowered = field.lower()
    if any(token in lowered for token in ("снилс", "телефон", "паспорт")):
        return _norm_digits(value)
    return _norm_text(value)


def _levenshtein(a: str, b: str) -> int:
    """Редакционное расстояние (DP O(n·m)) для коротких строк полей."""
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,          # удаление
                current[j - 1] + 1,       # вставка
                previous[j - 1] + (char_a != char_b),  # замена
            ))
        previous = current
    return previous[-1]


def _cer(reference: str, predicted: str) -> float:
    denominator = max(len(reference), 1)
    return _levenshtein(reference, predicted) / denominator


def _wer(reference: str, predicted: str) -> float:
    ref_words = reference.split()
    pred_words = predicted.split()
    denominator = max(len(ref_words), 1)
    return _levenshtein(ref_words, pred_words) / denominator


# --------------------------------------------------------------------------
# Загрузка набора
# --------------------------------------------------------------------------

def _find_cases() -> List[Path]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted(
        path for path in SAMPLES_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_EXT and not path.name.startswith(".")
    )


def _load_labels(case: Path) -> Dict[str, Any]:
    label_file = LABELS_DIR / f"{case.stem}.json"
    if not label_file.exists():
        raise FileNotFoundError(f"Нет эталона {label_file.name} для {case.name}")
    payload = json.loads(label_file.read_text(encoding="utf-8"))
    fields = payload.get("fields") or {}
    if not isinstance(fields, dict):
        raise ValueError(f"Эталон {label_file.name}: ожидался объект 'fields'")
    return fields


# --------------------------------------------------------------------------
# Прогон
# --------------------------------------------------------------------------

def _evaluate_case(pipeline: CompilerPipeline, case: Path) -> Dict[str, Any]:
    reference = _load_labels(case)
    image_base64 = base64.b64encode(case.read_bytes()).decode("ascii")
    columns = list(reference.keys())
    try:
        data = pipeline.extract(
            image_base64,
            target_columns=columns,
            filename=case.name,
            ocr_priority="auto",
        )
    except Exception as exc:  # ошибка одного файла не валит весь замер
        return {"case": case.name, "error": f"{type(exc).__name__}: {exc}"}

    per_field: Dict[str, Dict[str, Any]] = {}
    totals = {"cer": 0.0, "wer": 0.0, "count": 0}
    for field, expected_raw in reference.items():
        expected = _norm_field(field, expected_raw)
        predicted = _norm_field(field, data.get(field, ""))
        cer = _cer(expected, predicted)
        wer = _wer(expected, predicted)
        per_field[field] = {
            "reference": expected_raw,
            "predicted": data.get(field, ""),
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "exact": predicted == expected,
        }
        totals["cer"] += cer
        totals["wer"] += wer
        totals["count"] += 1

    return {
        "case": case.name,
        "error": None,
        "fields": per_field,
        "mean_cer": round(totals["cer"] / max(totals["count"], 1), 4),
        "mean_wer": round(totals["wer"] / max(totals["count"], 1), 4),
        "exact_fields": sum(1 for f in per_field.values() if f["exact"]),
        "field_count": totals["count"],
    }


def _format_md(results: List[Dict[str, Any]]) -> str:
    lines = [
        "# Оценка точности DocAI (CER/WER)",
        "",
        "> CER — редакционное расстояние к длине эталона, WER — то же по словам.",
        "> Значения нормализованы (СНИЛС/телефон/паспорт — только цифры, остальное — регистр/ё/пробелы).",
        "",
        "| Файл | Полей | Точных | CER (ср.) | WER (ср.) |",
        "|---|---:|---:|---:|---:|",
    ]
    ok = [r for r in results if not r.get("error")]
    for result in ok:
        lines.append(
            f"| {result['case']} | {result['field_count']} | "
            f"{result['exact_fields']} | {result['mean_cer']} | {result['mean_wer']} |"
        )
    for result in results:
        if result.get("error"):
            lines.append(f"\n`{result['case']}`: ошибка — {result['error']}")
    if ok:
        total_fields = sum(r["field_count"] for r in ok)
        exact = sum(r["exact_fields"] for r in ok)
        mean_cer = sum(r["mean_cer"] for r in ok) / len(ok)
        mean_wer = sum(r["mean_wer"] for r in ok) / len(ok)
        lines += [
            "",
            f"**Итог:** {exact}/{total_fields} полей распознаны точно; "
            f"средний CER {mean_cer:.4f}, средний WER {mean_wer:.4f}.",
        ]
    else:
        lines.append("\nНет успешных прогонов.")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Замер CER/WER на размеченном наборе")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число документов")
    parser.add_argument("--out", default="", help="Путь к файлу отчёта (.md)")
    args = parser.parse_args(argv)

    cases = _find_cases()
    if not cases:
        print(
            "Нет размеченных документов.\n"
            f"Положите обезличенные сканы в {SAMPLES_DIR}/, а эталоны в "
            f"{LABELS_DIR}/ — см. reference/evaluate_README.md"
        )
        return 2
    if args.limit:
        cases = cases[: args.limit]

    print(f"Документов в замере: {len(cases)}")
    pipeline = CompilerPipeline()

    results: List[Dict[str, Any]] = []
    for case in cases:
        print(f"  • {case.name} …")
        result = _evaluate_case(pipeline, case)
        results.append(result)
        if result.get("error"):
            print(f"      ошибка: {result['error']}")
        else:
            print(f"      CER {result['mean_cer']:.4f} · WER {result['mean_wer']:.4f} "
                  f"({result['exact_fields']}/{result['field_count']} полей точно)")

    report = _format_md(results)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\nОтчёт сохранён: {args.out}")
    else:
        print("\n" + report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
