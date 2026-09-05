"""Локализация рукописных значений относительно печатных меток.

PaddleOCR используется только как детектор геометрии. Распознавание значения
выполняет VLM, поскольку стандартный OCR не является надёжным для русского
рукописного текста.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

ALIASES = ["Дата подачи заявления", "ФИО поступающего ученика", "Дата рождения ребенка", "Класс / профиль обучения", "ФИО родителя / заявителя", "Паспортные данные", "Адрес регистрации / проживания", "Контактный телефон", "СНИЛС поступающего", "Особые отметки / льготы"]

@dataclass
class Region:
    box: Tuple[int, int, int, int]
    includes_label: bool = True


class FieldLocator:
    LABELS = {
        "Дата подачи заявления": ("дата подачи", "дата подачи заявления"),
        "ФИО поступающего ученика": ("фио поступающего", "фамилия, имя, отчество поступающего", "поступающего (ребенка)"),
        "Дата рождения ребенка": ("дата и место рождения", "дата рождения ребенка", "дата рождения"),
        "Класс / профиль обучения": ("желаемый класс", "профиль обучения", "класс"),
        "ФИО родителя / заявителя": ("фио родителя", "фио заявителя", "родителя (законного представителя)"),
        "Паспортные данные": ("паспортные данные", "паспорт"),
        "Адрес регистрации / проживания": ("адрес регистрации", "фактического проживания", "адрес проживания"),
        "Контактный телефон": ("контактный номер телефона", "контактный телефон", "телефон для связи"),
        "СНИЛС поступающего": ("страховой номер", "снилс", "лицевого счета"),
        "Особые отметки / льготы": ("дополнительные сведения", "наличие льгот", "особые отметки"),
    }

    def __init__(self, min_score: float = 0.25, detector_name: str = "PP-OCRv6_medium_det"):
        self.min_score = min_score
        self.detector_name = detector_name
        self._ocr = None
        self._ocr_class = None
        self.error = ""
        try:
            import paddle  # noqa: F401
            from paddleocr import PaddleOCR
            self._ocr_class = PaddleOCR
        except Exception as exc:
            self.error = str(exc)

    @property
    def available(self) -> bool:
        return self._ocr_class is not None

    def _ensure_ocr(self) -> bool:
        if self._ocr is not None:
            return True
        if self._ocr_class is None:
            return False
        try:
            self._ocr = self._ocr_class(
                text_detection_model_name=self.detector_name,
                lang="ru", use_doc_orientation_classify=False,
                use_doc_unwarping=False, use_textline_orientation=False,
            )
            return True
        except Exception as exc:
            self.error = str(exc)
            self._ocr = None
            return False

    @staticmethod
    def _box(item: Any) -> Optional[Tuple[int, int, int, int]]:
        try:
            pts = item.tolist() if hasattr(item, "tolist") else item
            xs, ys = zip(*[(float(p[0]), float(p[1])) for p in pts])
            return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        except Exception:
            return None

    def locate(self, image: Any) -> Dict[str, Region]:
        if not self._ensure_ocr():
            return {}
        try:
            result = self._ocr.predict(image)
            rows: List[Tuple[str, Tuple[int, int, int, int], float]] = []
            for page in result:
                data = getattr(page, "json", None)
                data = data() if callable(data) else data
                data = data or getattr(page, "res", {})
                # PaddleOCR 3.x часто возвращает {"res": {...}}.
                if isinstance(data, dict) and isinstance(data.get("res"), dict):
                    data = data["res"]
                polys = data.get("dt_polys", []) if isinstance(data, dict) else []
                texts = data.get("rec_texts", []) if isinstance(data, dict) else []
                scores = data.get("rec_scores", []) if isinstance(data, dict) else []
                for poly, text, score in zip(polys, texts, scores):
                    box = self._box(poly)
                    if box and float(score) >= self.min_score:
                        rows.append((str(text).strip(), box, float(score)))
            rows.sort(key=lambda x: (x[1][1], x[1][0]))
            found: Dict[str, Region] = {}
            for field, aliases in self.LABELS.items():
                for text, label_box, _ in rows:
                    norm = re.sub(r"[^а-яё0-9 ]", " ", text.lower())
                    if not any(alias in norm for alias in aliases):
                        continue
                    lx1, ly1, lx2, ly2 = label_box
                    candidates = [r for r in rows if r[1][0] >= lx2 - 8 and abs(r[1][1] - ly1) <= max(30, (ly2-ly1)*2)]
                    # Для паспорта захватываем несколько строк до следующей метки.
                    if field == "Паспортные данные":
                        candidates = [r for r in rows if r[1][1] >= ly1 and r[1][1] <= ly1 + 180 and r[1][0] >= lx2 - 8]
                    if candidates:
                        x2 = max(r[1][2] for r in candidates)
                        y2 = max(r[1][3] for r in candidates)
                        found[field] = Region((max(0, lx2 - 8), max(0, ly1 - 8), x2 + 12, y2 + 12))
                    break
            return found
        except Exception:
            return {}
