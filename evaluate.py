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
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from compiler.pipeline import CompilerPipeline
from compiler import connections as connections_store
from compiler.settings_store import load_settings
from validator import DataValidator
import config

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
    if "дата" in lowered:
        return DataValidator.format_date(str(value or ""))
    if "телефон" in lowered:
        return _norm_text(DataValidator.format_phone(str(value or "")))
    if "снилс" in lowered:
        if "[" in str(value or ""):
            return _norm_text(value)
        return _norm_digits(value)
    return _norm_text(value)


def _digit_alignment(reference: str, predicted: str) -> List[dict]:
    """Levenshtein alignment: an omitted digit must not shift all substitutions."""
    rows, cols = len(reference), len(predicted)
    costs = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        costs[i][0] = i
    for j in range(cols + 1):
        costs[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            costs[i][j] = min(costs[i - 1][j] + 1, costs[i][j - 1] + 1,
                              costs[i - 1][j - 1] + (reference[i - 1] != predicted[j - 1]))
    aligned = []
    i, j = rows, cols
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (reference[i - 1] != predicted[j - 1]):
            aligned.append({"reference": reference[i - 1], "predicted": predicted[j - 1], "position": i})
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            aligned.append({"reference": reference[i - 1], "predicted": "∅", "position": i})
            i -= 1
        else:
            aligned.append({"reference": "∅", "predicted": predicted[j - 1], "position": i + 1})
            j -= 1
    return list(reversed(aligned))


def _configured_pipeline():
    settings = load_settings()
    items, active_id = connections_store.from_saved(settings)
    conn = connections_store.active(items, active_id) or {}
    pipeline = CompilerPipeline(host=conn.get("host"), model_name=conn.get("model"))
    if isinstance(settings.get("params"), dict):
        pipeline.extractor.apply_params(settings["params"])
    if conn.get("api_key"):
        pipeline.extractor.set_api_key(conn["api_key"])
        pipeline.available_models = pipeline.extractor.get_available_models()
    # A benchmark must not silently switch to another model.
    if conn.get("model"):
        pipeline.extractor.model_name = conn["model"]
    return pipeline


def _run_snapshot(pipeline, mode):
    ex = pipeline.extractor
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": ex.model_name, "backend": ex.backend, "ocr_priority": mode,
        "params": ex.params_snapshot(), "detector": ex.field_locator.detector_name,
        "paddle_available": pipeline.paddle_available,
        "crop_scale": config.FIELD_CROP_SCALE, "max_image_dimension": config.MAX_IMAGE_DIMENSION,
        "use_field_crops": config.USE_PADDLE_FIELD_CROPS,
        "use_paddle": config.USE_PADDLE_OCR, "use_assist": config.USE_PADDLE_OCR_ASSIST,
        "verify_fields": list(config.PADDLE_VERIFY_FIELDS),
        "use_validator": config.USE_VALIDATOR, "use_dictionary": config.USE_REFERENCE_DICTIONARY,
        "template": ex.template,
    }


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

def _evaluate_case(pipeline: CompilerPipeline, case: Path, mode: str = "auto") -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        reference = _load_labels(case)
        image_base64 = base64.b64encode(case.read_bytes()).decode("ascii")
        columns = list(reference.keys())
        data = pipeline.extract(
            image_base64,
            target_columns=columns,
            filename=case.name,
            ocr_priority=mode,
        )
        if not isinstance(data, dict) or data.get("error"):
            return {"case": case.name, "error": "Распознавание завершилось ошибкой"}
    except Exception as exc:  # ошибка одного файла не валит весь замер
        return {"case": case.name, "error": f"{type(exc).__name__}: {exc}"}

    per_field: Dict[str, Dict[str, Any]] = {}
    totals = {"cer": 0.0, "wer": 0.0, "count": 0}
    for field, expected_raw in reference.items():
        expected = _norm_field(field, expected_raw)
        predicted = _norm_field(field, data.get(field, ""))
        cer = _cer(expected, predicted)
        wer = _wer(expected, predicted)
        meta = (data.get("_fields") or {}).get(field, {})
        verified = meta.get("verified") is True and meta.get("status") == "read"
        per_field[field] = {
            "reference": expected_raw,
            "predicted": data.get(field, ""),
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "exact": predicted == expected,
            "verified": verified,
            "false_verified": verified and predicted != expected,
            "needs_review": not verified,
            "verification": meta.get("verification", {}),
        }
        if any(token in field.lower() for token in ("снилс", "телефон", "дата", "паспорт")):
            per_field[field]["digit_alignment"] = _digit_alignment(_norm_digits(expected), _norm_digits(predicted))
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
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "requests": data.get("_stats", {}).get("requests", 0),
    }


def _format_md(results: List[Dict[str, Any]], run: Optional[dict] = None) -> str:
    lines = [
        "# Оценка точности DocAI (CER/WER)",
        "",
        "> CER — редакционное расстояние к длине эталона, WER — то же по словам.",
        "> Даты и одиночные телефоны форматируются; СНИЛС сравнивается по цифрам. Паспорт включает текст.",
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
    fields = defaultdict(list)
    matrix = Counter()
    positions = Counter()
    skips = Counter()
    for result in ok:
        for field, metrics in result["fields"].items():
            fields[field].append(metrics)
            audit = metrics.get("verification", {})
            if audit.get("reason"):
                skips[(audit.get("status", "unknown"), audit["reason"])] += 1
            for pair in metrics.get("digit_alignment", []):
                matrix[(pair["reference"], pair["predicted"])] += 1
                if pair["reference"] != pair["predicted"]:
                    positions[(field, pair["position"], pair["reference"], pair["predicted"])] += 1
    lines += ["", "## Качество по полям", "",
              "| Поле | Точных / всего | Подтверждено | Ошибочных среди подтверждённых | Требуют проверки |",
              "|---|---:|---:|---:|---:|"]
    for field, entries in sorted(fields.items()):
        verified = sum(e.get("verified", False) for e in entries)
        false_verified = sum(e.get("false_verified", False) for e in entries)
        rate = f"{false_verified / verified:.1%}" if verified else "—"
        review = sum(e.get("needs_review", True) for e in entries)
        lines.append(f"| {field} | {sum(e['exact'] for e in entries)}/{len(entries)} | {verified} | "
                     f"{false_verified}/{verified} ({rate}) | {review}/{len(entries)} ({review / len(entries):.1%}) |")
    audits = [e.get("verification", {}) for entries in fields.values() for e in entries]
    lines += ["", f"Проверочных чтений: начато {sum(a.get('attempted', 0) for a in audits)}, "
              f"выполнено {sum(a.get('completed', 0) for a in audits)}."]
    lines += ["", f"Время обработки: {sum(r.get('elapsed_seconds', 0) for r in ok):.3f} с. "
              f"Запросов к ИИ: {sum(r.get('requests', 0) for r in ok)}.",
              "", "## Матрица цифр (строка — эталон, столбец — распознано)", "",
              "∅ — пропуск или вставка. Выравнивание учитывает изменение длины.", ""]
    symbols = list("0123456789") + ["∅"]
    lines += ["| Эталон | " + " | ".join(symbols) + " |", "|---|" + "---:|" * len(symbols)]
    for ref in symbols:
        lines.append(f"| {ref} | " + " | ".join(str(matrix[(ref, pred)]) for pred in symbols) + " |")
    lines += ["", "## Ошибки по позициям", "", "| Поле | Позиция | Эталон → ответ | Количество |", "|---|---:|---|---:|"]
    for (field, position, ref, pred), count in sorted(positions.items()):
        lines.append(f"| {field} | {position} | {ref} → {pred} | {count} |")
    lines += ["", "## Пропуски и ошибки проверочных чтений", "", "| Статус | Причина | Полей |", "|---|---|---:|"]
    for (status, reason), count in sorted(skips.items()):
        lines.append(f"| {status} | {reason} | {count} |")
    if run:
        lines += ["", "## Параметры прогона", "", "```json", json.dumps(run, ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Замер CER/WER на размеченном наборе")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число документов")
    parser.add_argument("--out", default="", help="Путь к файлу отчёта (.md)")
    parser.add_argument("--ocr-priority", choices=("auto", "paddle", "vlm"), default="auto")
    parser.add_argument("--json-out", default="", help="Подробные результаты и параметры в JSON")
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
    pipeline = _configured_pipeline()
    run = _run_snapshot(pipeline, args.ocr_priority)

    results: List[Dict[str, Any]] = []
    for case in cases:
        print(f"  • {case.name} …")
        result = _evaluate_case(pipeline, case, args.ocr_priority)
        results.append(result)
        if result.get("error"):
            print(f"      ошибка: {result['error']}")
        else:
            print(f"      CER {result['mean_cer']:.4f} · WER {result['mean_wer']:.4f} "
                  f"({result['exact_fields']}/{result['field_count']} полей точно)")

    report = _format_md(results, run)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"run": run, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\nОтчёт сохранён: {args.out}")
    else:
        print("\n" + report)
    return 1 if any(result.get("error") for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
