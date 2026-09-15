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
from cancellation import CancellationToken, ExtractionCancelled

logger = get_logger("pipeline")


class RecognitionError(RuntimeError):
    """Распознавание документа не дало корректного результата."""


class CompilerPipeline:
    """Оркестратор: загрузка, OCR/VLM, валидатор и локальный справочник."""

    def __init__(self, host: Optional[str] = None, model_name: Optional[str] = None) -> None:
        # Адрес/модель по умолчанию приходят из конфига либо из settings.json.
        self.extractor = DocumentExtractor(
            host=host or config.OLLAMA_HOST,
            model_name=model_name or config.DEFAULT_MODEL,
        )
        self.available_models: list[str] = list(getattr(self.extractor, "available_models", []))

    @property
    def paddle_available(self) -> bool:
        return bool(getattr(self.extractor.field_locator, "available", False))

    def connect(self, host: str, api_key: Optional[str] = None) -> dict[str, object]:
        """Переключает Ollama/LM Studio на разрешённый адрес и проверяет модели.

        При неудаче подключение откатывается к предыдущему адресу, чтобы
        мёртвый сервер не «завис» в активном экземпляре экстрактора.
        """
        previous_host = self.extractor.host
        previous_backend = self.extractor.backend
        previous_model = self.extractor.model_name
        previous_key = self.extractor.api_key
        previous_models = list(self.available_models)

        self.extractor.set_host(host)
        if api_key is not None:
            self.extractor.set_api_key(api_key)
        google_sf = self.extractor._is_siliconflow_host(self.extractor.host) or self.extractor._is_google_host(self.extractor.host)
        if google_sf and not self.extractor.api_key:
            service = "Google AI Studio" if self.extractor._is_google_host(self.extractor.host) else "SiliconFlow"
            self.extractor.set_host(previous_host)
            self.extractor.backend = previous_backend
            self.extractor.model_name = previous_model
            self.extractor.set_api_key(previous_key)
            raise ValueError(f"Для {service} необходимо указать API-ключ")
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
        if not reachable:
            # Откатываем предыдущее рабочее подключение.
            self.extractor.set_host(previous_host)
            self.extractor.backend = previous_backend
            self.extractor.model_name = previous_model
            if api_key is not None:
                self.extractor.set_api_key(previous_key)
            # Возвращаем прежнее рабочее подключение вместе со списком моделей,
            # чтобы неудачная проверка нового адреса не «ломала» активный сервер.
            self.available_models = previous_models
            return {
                "host": self.extractor.host,
                "backend": self.extractor.backend,
                "model": self.extractor.model_name,
                "models": [],
                "connected": False,
            }
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

    def probe(self, host: str, api_key: Optional[str] = None) -> dict[str, object]:
        """Проверяет адрес, не переключая активное подключение.

        Используется для сохранённых профилей, которые не активны: результат
        доступности нужен, но текущий рабочий сервер не должен меняться.
        """
        extractor = self.extractor
        snapshot = (
            extractor.host,
            extractor.backend,
            extractor.model_name,
            extractor.api_key,
            list(self.available_models),
        )
        try:
            return self.connect(host, api_key)
        finally:
            extractor.set_host(snapshot[0])
            extractor.backend = snapshot[1]
            extractor.model_name = snapshot[2]
            extractor.set_api_key(snapshot[3])
            self.available_models = snapshot[4]

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
        *, cancellation: Optional[CancellationToken] = None,
    ) -> Dict[str, Any]:
        """Извлекает поля одного документа (для PDF — объединяя все страницы)."""
        self._cancellation = cancellation
        self.extractor.cancellation = cancellation
        try:
            self._check_cancelled()
            result = self._extract_document(image_base64, target_columns, filename, ocr_priority, detector, model)
            self._check_cancelled()
            return result
        finally:
            self._cancellation = None
            self.extractor.cancellation = None

    def _check_cancelled(self):
        cancellation = getattr(self, "_cancellation", None)
        if cancellation is not None:
            cancellation.check()

    def _extract_document(self, image_base64, target_columns, filename, ocr_priority, detector, model):
        if model:
            self.set_model(model)
        if detector != self.extractor.field_locator.detector_name:
            self.extractor.set_paddle_detector(detector)
        # VLM всегда читает полный лист. auto добавляет проверку критичных полей,
        # paddle проверяет все найденные поля, vlm отключает Paddle-проверку.
        use_field_crops = {"paddle": True, "vlm": False}.get(ocr_priority)
        raw = image_base64.split(",", 1)[-1]
        image_bytes = base64.b64decode(raw, validate=True)

        if not filename.lower().endswith(".pdf"):
            image = DocumentLoader.open_image(io.BytesIO(image_bytes))
            enhanced = DocumentLoader.preprocess_for_handwriting(image)
            return self._extract_page(
                enhanced,
                target_columns=target_columns,
                pil_image=image,
                use_field_crops=use_field_crops,
            )

        pages = self._rasterize_pdf_pages(image_bytes)
        logger.info("pdf: страниц=%d", len(pages))
        if len(pages) == 1:
            pil_image, enhanced = pages[0]
            return self._extract_page(
                enhanced,
                target_columns=target_columns,
                pil_image=pil_image,
                use_field_crops=use_field_crops,
            )
        # Многостраничный PDF = один ученик: значения полей объединяются по страницам.
        return self._extract_multipage(pages, target_columns)

    def _extract_page(self, image_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """На границе конвейера ошибки ИИ всегда становятся исключениями."""
        self._check_cancelled()
        result = self.extractor.extract_from_image(image_bytes, **kwargs)
        self._check_cancelled()
        if not isinstance(result, dict) or not result:
            raise RecognitionError("Модель не вернула корректные поля документа")
        if result.get("error"):
            raise RecognitionError(str(result["error"]))
        return result

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
                self._check_cancelled()
                pil = DocumentLoader.rasterize_page(page, matrix)
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
            try:
                page_data.append(self._extract_page(
                    enhanced,
                    target_columns=target_columns,
                    pil_image=None,
                    use_field_crops=False,
                    verification_skip_reason="multipage_disabled",
                ))
            except ExtractionCancelled:
                raise
            except Exception as exc:
                # Нельзя экспортировать неполное заявление как успешно прочитанное.
                raise RecognitionError(f"Не удалось распознать страницу {index} PDF: {exc}") from exc
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
        stats = {"tokens_total": 0, "elapsed": 0.0, "requests": 0, "backend": "", "model": "", "speed": 0.0}

        for page in pages_data:
            page_stats = page.get("_stats") or {}
            stats["tokens_total"] += page_stats.get("tokens_total", 0)
            stats["requests"] += page_stats.get("requests", 0)
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

        for field, field_meta in meta.items():
            page_meta = (pages_data[field_meta["page"] - 1].get("_fields") or {}).get(field, {})
            field_meta["verification"] = page_meta.get("verification", {
                "status": "skipped", "reason": "multipage_disabled", "attempted": 0, "completed": 0,
            })
            field_meta["verified"] = bool(page_meta.get("verified")) and field_meta["status"] == "read"
        merged["_fields"] = meta
        merged["_stats"] = stats
        merged["_pdf_pages"] = len(pages_data)
        if conflicts:
            merged["_conflicts"] = conflicts

        # Справочник и валидатор применяем один раз к объединённой записи
        # (постраничные данные уже отформатированы, повтор безопасен).
        return self.extractor._postprocess_data(merged)
