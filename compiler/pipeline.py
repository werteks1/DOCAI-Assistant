from __future__ import annotations

import base64
import io
import pymupdf
from typing import Any, Dict, Optional

from PIL import Image

import config
from document_loader import DocumentLoader
from extractor import DocumentExtractor


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
        if model:
            self.set_model(model)
        if detector != self.extractor.field_locator.detector_name:
            self.extractor.set_paddle_detector(detector)
        use_field_crops = {"paddle": True, "vlm": False}.get(ocr_priority)
        raw = image_base64.split(",", 1)[-1]
        image_bytes = base64.b64decode(raw, validate=True)
        if filename.lower().endswith(".pdf"):
            with pymupdf.open(stream=image_bytes, filetype="pdf") as document:
                if not document.page_count:
                    raise ValueError("PDF не содержит страниц")
                page = document.load_page(0)
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False)
                image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        else:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        enhanced = DocumentLoader.preprocess_for_handwriting(image)
        return self.extractor.extract_from_image(
            enhanced,
            target_columns=target_columns,
            pil_image=image,
            use_field_crops=use_field_crops,
        )
