from __future__ import annotations

import base64
import io
import pymupdf
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

import config
from document_loader import DocumentLoader
from extractor import DocumentExtractor
from logging_utils import get_logger

logger = get_logger("pipeline")


class CompilerPipeline:
    """Оркестратор: загрузка, OCR/VLM, валидатор и локальный справочник."""

    def __init__(self) -> None:
        self.extractor = DocumentExtractor(host=config.OLLAMA_HOST, model_name=config.DEFAULT_MODEL)
        self.available_models: list[str] = list(getattr(self.extractor, "available_models", []))

    @property
    def paddle_available(self) -> bool:
        return bool(getattr(self.extractor.field_locator, "available", False))

    def connect(self, host: str) -> dict[str, object]:
        """Переключает Ollama/LM Studio на разрешённый адрес и проверяет модели."""
        self.extractor.set_host(host)
        models = self.extractor.get_available_models()
        self.available_models = models
        reachable = bool(models)
        if not reachable:
            for path in ("/api/tags", "/v1/models"):
                try:
                    response = self.extractor._session.get(f"{self.extractor.host}{path}", timeout=2.5)
                    if response.status_code == 200:
                        reachable = True
                        break
                except Exception:
                    continue
        if models and self.extractor.model_name not in models:
            preferred = next((m for m in models if "2.5-vl" in m.lower()), models[0])
            self.extractor.model_name = preferred
        return {
            "host": self.extractor.host,
            "backend": self.extractor.backend,
            "model": self.extractor.model_name,
            "models": models,
            "connected": reachable,
        }

    def set_model(self, model: str) -> str:
        model = str(model or "").strip()
        if not model:
            raise ValueError("Модель не выбрана")
        if self.available_models and model not in self.available_models:
            raise ValueError("Выбранная модель не найдена на подключённом сервере")
        self.extractor.model_name = model
        return model

    def extract(
        self,
        image_base64: str,
        target_columns: Optional[list[str]] = None,
        filename: str = "",
        ocr_priority: str = "auto",
        detector: str = config.PADDLE_DETECTOR,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Извлекает поля одного документа (для PDF — объединяя все страницы)."""
        if model:
            self.set_model(model)
        if detector != self.extractor.field_locator.detector_name:
            self.extractor.set_paddle_detector(detector)
        use_field_crops = {"paddle": True, "vlm": False}.get(ocr_priority)
        raw = image_base64.split(",", 1)[-1]
        image_bytes = base64.b64decode(raw, validate=True)

        if not filename.lower().endswith(".pdf"):
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            enhanced = DocumentLoader.preprocess_for_handwriting(image)
            return self.extractor.extract_from_image(
                enhanced,
                target_columns=target_columns,
                pil_image=image,
                use_field_crops=use_field_crops,
            )

        pages = self._rasterize_pdf_pages(image_bytes)
        logger.info("pdf: страниц=%d", len(pages))
        if len(pages) == 1:
            pil_image, enhanced = pages[0]
            return self.extractor.extract_from_image(
                enhanced,
                target_columns=target_columns,
                pil_image=pil_image,
                use_field_crops=use_field_crops,
            )
        # Многостраничный PDF = один ученик: значения полей объединяются по страницам.
        return self._extract_multipage(pages, target_columns)

    def _rasterize_pdf_pages(self, image_bytes: bytes) -> List[Tuple[Image.Image, bytes]]:
        """Растеризует все страницы PDF в (оригинальный PIL, обработанные PNG-байты)."""
        with pymupdf.open(stream=image_bytes, filetype="pdf") as document:
            total = document.page_count
            if not total:
                raise ValueError("PDF не содержит страниц")
            if total > config.MAX_PDF_PAGES:
                raise ValueError(f"PDF содержит слишком много страниц: {total} (лимит {config.MAX_PDF_PAGES})")
            pages: List[Tuple[Image.Image, bytes]] = []
            matrix = pymupdf.Matrix(3, 3)
            for page in document:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                pil = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                if pil.width * pil.height > config.MAX_IMAGE_PIXELS:
                    raise ValueError(f"Страница PDF слишком большая: {pil.width}x{pil.height}")
                enhanced = DocumentLoader.preprocess_for_handwriting(pil)
                pages.append((pil, enhanced))
        return pages

    # --- Многостраничное объединение (PDF = один ученик) ---------------

    @staticmethod
    def _is_unclear_text(value: str) -> bool:
        """Значение, помеченное моделью как неразборчивое («[неразборчиво]»)."""
        return "[" in value

    def _extract_multipage(
        self,
        pages: List[Tuple[Image.Image, bytes]],
        target_columns: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """Читает каждую страницу полным проходом VLM и объединяет в один набор полей.

        Crop/fallback-геометрия тюнена под одностраничный бланк, поэтому для
        многостраничного документа используется чистый полный проход VLM
        (без Paddle-кропов) — он устойчив к нестандартной вёрстке 2+ страницы.
        """
        page_data: List[Dict[str, Any]] = []
        for index, (pil_image, enhanced) in enumerate(pages, start=1):
            logger.info("страница=%d распознавание", index)
            page_data.append(self.extractor.extract_from_image(
                enhanced,
                target_columns=target_columns,
                pil_image=None,
                use_field_crops=False,
            ))
        return self._merge_page_results(page_data)

    def _merge_page_results(self, pages_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Объединяет результаты постраничного распознавания в одну запись."""
        # Поля определяем по метаданным страниц (полный проход заполняет _fields).
        columns: List[str] = []
        for page in pages_data:
            for key in (page.get("_fields") or {}).keys():
                if key not in columns:
                    columns.append(key)

        merged: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}
        conflicts: Dict[str, List[Dict[str, Any]]] = {}
        stats = {"tokens_total": 0, "elapsed": 0.0, "backend": "", "model": "", "speed": 0.0}

        for page in pages_data:
            page_stats = page.get("_stats") or {}
            stats["tokens_total"] += page_stats.get("tokens_total", 0)
            stats["elapsed"] += page_stats.get("elapsed", 0.0)
            if page_stats.get("backend"):
                stats["backend"] = page_stats["backend"]
            if page_stats.get("model"):
                stats["model"] = page_stats["model"]
            if page_stats.get("speed"):
                stats["speed"] = page_stats["speed"]

        for field in columns:
            candidates = []  # (страница_1-based, значение, status, source)
            for page_index, page in enumerate(pages_data, start=1):
                value = page.get(field)
                if value is None or not str(value).strip():
                    continue
                field_meta = (page.get("_fields") or {}).get(field, {})
                candidates.append((
                    page_index,
                    str(value).strip(),
                    field_meta.get("status", "needs_review"),
                    field_meta.get("source", "page"),
                ))

            if not candidates:
                merged[field] = ""
                meta[field] = {"status": "needs_review", "source": "page", "page": 1}
                continue

            solid = [c for c in candidates if not self._is_unclear_text(c[1])]
            if not solid:
                # Все страницы дали неразборчивое значение — берём первое и просим проверить.
                chosen = candidates[0]
                meta[field] = {"status": "needs_review", "source": chosen[3], "page": chosen[0]}
                merged[field] = chosen[1]
                continue

            unique_solid = {c[1] for c in solid}
            if len(unique_solid) == 1:
                chosen = solid[0]
                meta[field] = {"status": chosen[2], "source": chosen[3], "page": chosen[0]}
                merged[field] = chosen[1]
                continue

            # Несколько различающихся значений на разных страницах.
            chosen = next((c for c in solid if c[2] == "read"), solid[0])
            meta[field] = {
                "status": "needs_review",
                "source": chosen[3],
                "page": chosen[0],
                "note": "значения на страницах различаются — проверьте",
            }
            conflicts[field] = [
                {"page": c[0], "value": c[1], "status": c[2]}
                for c in solid if c[1] != chosen[1]
            ]
            merged[field] = chosen[1]

        merged["_fields"] = meta
        merged["_stats"] = stats
        merged["_pdf_pages"] = len(pages_data)
        if conflicts:
            merged["_conflicts"] = conflicts

        # Справочник и валидатор применяем один раз к объединённой записи
        # (постраничные данные уже отформатированы, повтор безопасен).
        return self.extractor._postprocess_data(merged)
