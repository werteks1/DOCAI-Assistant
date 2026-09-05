"""Локальный расширяемый справочник для проверки результатов OCR.

Справочник не подменяет изображение и не отправляется в VLM. Он используется
после распознавания для поиска близких вариантов и сохраняет предложения
исправлений в поле ``_reference_corrections``.
"""
from __future__ import annotations

import csv
import difflib
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULTS = {
    "surnames": {"Бобров", "Боброва", "Иванов", "Иванова", "Петров", "Петрова", "Смирнов", "Смирнова"},
    "first_names": {"Александр", "Александра", "Алексей", "Анна", "Владислав", "Владимир", "Мария", "Иван"},
    "patronymics": {"Александрович", "Алексеевич", "Владимирович", "Иванович", "Сергеевич", "Владиславовна"},
    "cities": {"Воронеж", "Москва", "Санкт-Петербург", "Липецк", "Белгород", "Ростов-на-Дону"},
    "organizations": {"РЖД", "МФЦ", "МВД", "ГИБДД"},
    "profiles": {"Физико-математический класс", "Гуманитарный класс", "Социально-экономический класс"},
}


class ReferenceDictionary:
    def __init__(self, root: Path | None = None):
        self.root = root or Path(__file__).resolve().parent / "reference"
        self.values: Dict[str, set[str]] = {k: set(v) for k, v in DEFAULTS.items()}
        self.corrections: Dict[str, Dict[str, str]] = {}
        self._load_files()

    def _load_files(self) -> None:
        self.root.mkdir(exist_ok=True)
        for category in self.values:
            path = self.root / f"{category}.txt"
            if not path.exists():
                continue
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    value = line.strip()
                    if value and not value.startswith("#"):
                        self.values[category].add(value)
            except OSError:
                continue
        custom = self.root / "custom_corrections.csv"
        if custom.exists():
            try:
                with custom.open("r", encoding="utf-8", newline="") as fh:
                    for row in csv.DictReader(fh):
                        category, wrong, right = row.get("category", ""), row.get("wrong", ""), row.get("right", "")
                        if category in self.values and right.strip():
                            self.values[category].add(right.strip())
                            if wrong.strip():
                                self.corrections.setdefault(category, {})[self._norm(wrong)] = right.strip()
            except (OSError, csv.Error):
                pass

    @staticmethod
    def _norm(value: str) -> str:
        return " ".join(value.casefold().replace("ё", "е").split())

    def best(self, value: str, category: str) -> Tuple[str, float]:
        candidates = self.values.get(category, set())
        if not value or not candidates:
            return value, 0.0
        value_norm = self._norm(value)
        best_value, best_score = value, 0.0
        for candidate in candidates:
            score = difflib.SequenceMatcher(None, value_norm, self._norm(candidate)).ratio()
            if score > best_score:
                best_value, best_score = candidate, score
        return best_value, best_score

    def correct_fio(self, value: str) -> Tuple[str, List[dict]]:
        parts = value.split()
        if not parts:
            return value, []
        categories = ["surnames", "first_names", "patronymics"]
        changes = []
        for index, category in enumerate(categories):
            if index >= len(parts):
                break
            direct = self.corrections.get(category, {}).get(self._norm(parts[index]))
            if direct:
                changes.append({"field_part": category, "from": parts[index], "to": direct, "score": 1.0, "source": "custom"})
                parts[index] = direct
                continue
            candidate, score = self.best(parts[index], category)
            # Порог достаточно строгий, но допускает Бобров/Бодров.
            if candidate != parts[index] and score >= (0.80 if category == "surnames" else 0.86):
                changes.append({"field_part": category, "from": parts[index], "to": candidate, "score": round(score, 3)})
                parts[index] = candidate
        return " ".join(parts), changes

    def correct_text_tokens(self, value: str, category: str, threshold: float = 0.84) -> Tuple[str, List[dict]]:
        changes = []
        words = value.split()
        for index, word in enumerate(words):
            direct = self.corrections.get(category, {}).get(self._norm(word.strip(".,")))
            if direct:
                changes.append({"from": word, "to": direct, "score": 1.0, "source": "custom"})
                words[index] = direct
                continue
            candidate, score = self.best(word.strip(".,"), category)
            if candidate != word and score >= threshold:
                words[index] = candidate
                changes.append({"from": word, "to": candidate, "score": round(score, 3)})
        return " ".join(words), changes

    def correct_data(self, data: Dict[str, object]) -> Dict[str, object]:
        corrections = {}
        for field in ("ФИО поступающего ученика", "ФИО родителя / заявителя"):
            value = data.get(field)
            if isinstance(value, str) and value.strip():
                corrected, changes = self.correct_fio(value)
                if changes:
                    data[field] = corrected
                    corrections[field] = changes
        for field in ("Адрес регистрации / проживания",):
            value = data.get(field)
            if isinstance(value, str) and value.strip():
                corrected, changes = self.correct_text_tokens(value, "cities")
                if changes:
                    data[field] = corrected
                    corrections[field] = changes
        value = data.get("Паспортные данные")
        if isinstance(value, str) and value.strip():
            corrected, changes = self.correct_text_tokens(value, "organizations", threshold=0.66)
            if changes:
                data["Паспортные данные"] = corrected
                corrections["Паспортные данные"] = changes
        if corrections:
            data["_reference_corrections"] = corrections
        return data

    def add_correction(self, category: str, wrong: str, right: str) -> None:
        """Сохраняет подтверждённую пользователем пару для следующих документов."""
        category = str(category or "").strip()
        wrong = str(wrong or "").strip()
        right = str(right or "").strip()
        if not category or not wrong or not right:
            return
        if self.corrections.get(category, {}).get(self._norm(wrong)) == right:
            return
        self.root.mkdir(exist_ok=True)
        path = self.root / "custom_corrections.csv"
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            if not exists:
                writer.writerow(["category", "wrong", "right"])
            writer.writerow([category, wrong, right])
        self.values.setdefault(category, set()).add(right)
        self.corrections.setdefault(category, {})[self._norm(wrong)] = right
