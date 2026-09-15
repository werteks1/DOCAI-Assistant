# -*- coding: utf-8 -*-
"""
Модуль загрузки и адаптивной предобработки документов (сканы, фото, PDF).
Включает:
- Устранение теней и неравномерного освещения (Background Division Normalization).
- Повышение резкости и микроконтраста штрихов шариковой ручки.
- Вырезание увеличенных фрагментов (Zoom-in field crops) для двухпроходного OCR.
"""
import io
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pymupdf

import config


class DocumentLoader:
    @staticmethod
    def check_image_size(width: int, height: int) -> None:
        if width <= 0 or height <= 0 or width * height > config.MAX_IMAGE_PIXELS:
            raise ValueError(f"Изображение слишком большое или некорректное: {width}x{height}")

    @staticmethod
    def open_image(source) -> Image.Image:
        # Размер берётся из заголовка до декодирования, EXIF и RGB-конверсии.
        with Image.open(source) as image:
            DocumentLoader.check_image_size(image.width, image.height)
            return ImageOps.exif_transpose(image).convert("RGB")

    @staticmethod
    def rasterize_page(page, matrix) -> Image.Image:
        bounds = (page.rect * matrix).irect
        DocumentLoader.check_image_size(bounds.width, bounds.height)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False, colorspace=pymupdf.csRGB)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    @staticmethod
    def load_document(file_path: Union[str, Path]) -> List[Tuple[Image.Image, bytes]]:
        """
        Загружает документ (изображение или PDF).
        Возвращает список кортежей: (исходное_PIL_изображение, оптимизированные_байты_для_VLM)
        для каждой страницы документа.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            return DocumentLoader._load_pdf(path)
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            return [DocumentLoader._load_single_image(path)]
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {ext}")

    @staticmethod
    def _load_single_image(path: Path) -> Tuple[Image.Image, bytes]:
        image = DocumentLoader.open_image(path)
        enhanced_bytes = DocumentLoader.preprocess_for_handwriting(image)
        return image, enhanced_bytes

    @staticmethod
    def _load_pdf(path: Path) -> List[Tuple[Image.Image, bytes]]:
        with pymupdf.open(path) as doc:
            page_count = len(doc)
            if not page_count or page_count > config.MAX_PDF_PAGES:
                raise ValueError(f"Недопустимое число страниц PDF: {page_count} (лимит {config.MAX_PDF_PAGES})")
            pages = []
            mat = pymupdf.Matrix(3.0, 3.0)
            for page in doc:
                pil_img = DocumentLoader.rasterize_page(page, mat)
                enhanced_bytes = DocumentLoader.preprocess_for_handwriting(pil_img)
                pages.append((pil_img, enhanced_bytes))
        return pages

    @staticmethod
    def remove_shadows_and_normalize(image: Image.Image) -> Image.Image:
        """
        Удаляет неравномерные тени и выравнивает освещение бумаги через
        деление яркостного канала Y на оценку фонового освещения (Gaussian blur).
        Бумага становится равномерно белой, а чернила ручки сохраняют цвет и четкость.
        """
        rgb = image.convert("RGB")
        ycbcr = rgb.convert("YCbCr")
        y, cb, cr = ycbcr.split()

        # Оценка локального фона через крупное размытие
        radius = max(25, int(min(image.size) * 0.04))
        bg = y.filter(ImageFilter.GaussianBlur(radius=radius))

        y_arr = np.array(y, dtype=np.float32)
        bg_arr = np.array(bg, dtype=np.float32)

        # Нормализация яркости (бумага становится белой ~240-250)
        y_norm = np.clip((y_arr / (bg_arr + 1e-5)) * 240.0, 0, 255).astype(np.uint8)
        y_clean = Image.fromarray(y_norm, mode="L")

        return Image.merge("YCbCr", (y_clean, cb, cr)).convert("RGB")

    @staticmethod
    def crop_document_contour(image: Image.Image) -> Image.Image:
        """
        Автоматически определяет границы белого листа бумаги на темном/нейтральном фоне стола
        и кадрирует изображение точно по документу с безопасным отступом.
        Устраняет непроизводительные затраты разрешения на поля стола (до 25-30% полезных пикселей).
        """
        w, h = image.size
        tw = 400
        th = max(100, int(tw * h / w))
        thumb = image.resize((tw, th), Image.Resampling.BILINEAR).convert("L")
        arr = np.array(thumb, dtype=np.float32)

        # Определение порога контраста между бумагой и фоном стола
        p25 = np.percentile(arr, 25)
        p75 = np.percentile(arr, 75)
        thresh = (p25 + p75) / 2.0
        if thresh < 110:
            thresh = 110

        mask = arr > thresh
        row_frac = np.mean(mask, axis=1)
        col_frac = np.mean(mask, axis=0)

        r_idx = np.where(row_frac > 0.25)[0]
        c_idx = np.where(col_frac > 0.25)[0]

        if len(r_idx) == 0 or len(c_idx) == 0:
            return image

        r_min, r_max = r_idx[0], r_idx[-1]
        c_min, c_max = c_idx[0], c_idx[-1]

        box_area = (r_max - r_min) * (c_max - c_min)
        total_area = tw * th
        ratio = box_area / max(1.0, total_area)

        # Если обнаружен фон стола (документ занимает от 35% до 95% площади кадра)
        if 0.35 <= ratio <= 0.95:
            pad_x = int(w * 0.015)
            pad_y = int(h * 0.015)
            x1 = max(0, int((c_min / tw) * w) - pad_x)
            x2 = min(w, int((c_max / tw) * w) + pad_x)
            y1 = max(0, int((r_min / th) * h) - pad_y)
            y2 = min(h, int((r_max / th) * h) + pad_y)
            if (x2 - x1) > 200 and (y2 - y1) > 200:
                return image.crop((x1, y1, x2, y2))

        return image

    @staticmethod
    def preprocess_for_handwriting_img(image: Image.Image, auto_crop: bool = True) -> Image.Image:
        """
        Внутренняя обработка PIL-изображения:
        - Автоматическая обрезка фона стола (Auto-Crop)
        - Выравнивание освещения и удаление теней (Background Division)
        - Адаптивный микроконтраст тонких штрихов шариковой ручки (UnsharpMask)
        """
        img = DocumentLoader.crop_document_contour(image) if auto_crop else image
        img = DocumentLoader.remove_shadows_and_normalize(img)

        w, h = img.size
        max_dim = config.MAX_IMAGE_DIMENSION
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Мягкий контраст
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(config.IMAGE_ENHANCE_CONTRAST)

        # Вытягивание бледных штрихов шариковой ручки
        img = img.filter(ImageFilter.UnsharpMask(radius=1.8, percent=130, threshold=2))

        # Резкость
        sharpener = ImageEnhance.Sharpness(img)
        img = sharpener.enhance(config.IMAGE_ENHANCE_SHARPNESS)

        return img

    @staticmethod
    def preprocess_for_handwriting(image: Image.Image) -> bytes:
        """
        Полный конвейер предобработки для отправки в VLM.
        Возвращает PNG-байты высокой четкости.
        """
        img = DocumentLoader.preprocess_for_handwriting_img(image)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def get_top_bottom_crops(image: Image.Image) -> Tuple[bytes, bytes]:
        """
        Разбивает лист А4 на 2 перекрывающихся блока:
        1. Верхняя половина (0% - 53% высоты): дата подачи, ребенок, класс, родитель.
        2. Нижняя половина (44% - 100% высоты): родитель, паспорт, адрес, телефон, СНИЛС, льготы, подпись и дата.
        Каждый блок масштабируется до ширины 2400px, обеспечивая двойное увеличение рукописного текста.
        """
        cleaned_doc = DocumentLoader.crop_document_contour(image)
        w, h = cleaned_doc.size

        # Верхний блок (0 - 53%)
        top_h = int(h * 0.53)
        top_img = cleaned_doc.crop((0, 0, w, top_h))
        if top_img.size[0] < 2200:
            scale = 2200.0 / max(1, top_img.size[0])
            top_img = top_img.resize((2200, int(top_img.size[1] * scale)), Image.Resampling.LANCZOS)
        top_img = DocumentLoader.preprocess_for_handwriting_img(top_img, auto_crop=False)
        top_buf = io.BytesIO()
        top_img.save(top_buf, format="PNG")

        # Нижний блок (44% - 100%)
        bot_y1 = int(h * 0.44)
        bot_img = cleaned_doc.crop((0, bot_y1, w, h))
        if bot_img.size[0] < 2200:
            scale = 2200.0 / max(1, bot_img.size[0])
            bot_img = bot_img.resize((2200, int(bot_img.size[1] * scale)), Image.Resampling.LANCZOS)
        bot_img = DocumentLoader.preprocess_for_handwriting_img(bot_img, auto_crop=False)
        bot_buf = io.BytesIO()
        bot_img.save(bot_buf, format="PNG")

        return top_buf.getvalue(), bot_buf.getvalue()

    @staticmethod
    def get_field_crop(image: Image.Image, y_start_ratio: float, y_end_ratio: float) -> bytes:
        """
        Вырезает горизонтальную полосу документа (zoom-in) для точечного распознавания отдельного поля.
        Гарантирует высокое разрешение фрагмента для мелкого рукописного текста.
        """
        # Сначала убираем фон стола, если он есть
        img = DocumentLoader.crop_document_contour(image)
        w, h = img.size
        y1 = max(0, int(h * max(0.0, y_start_ratio)))
        y2 = min(h, int(h * min(1.0, y_end_ratio)))
        if y2 <= y1:
            y2 = min(h, y1 + 100)

        cropped = img.crop((0, y1, w, y2))
        cw, ch = cropped.size

        # Масштабируем полосу до комфортного размера, если она узкая
        if cw < 1800:
            scale = 1800.0 / max(1, cw)
            cropped = cropped.resize((1800, int(ch * scale)), Image.Resampling.LANCZOS)

        cropped = DocumentLoader.preprocess_for_handwriting_img(cropped, auto_crop=False)
        buffer = io.BytesIO()
        cropped.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def get_bbox_crop(image: Image.Image, box, scale: float = 2.0) -> bytes:
        """Crop original pixels with padding; do not re-detect page boundaries."""
        import math
        x1, y1, x2, y2 = box
        pad = max(4, (y2 - y1) * 0.12)
        bounds = (max(0, math.floor(x1 - pad)), max(0, math.floor(y1 - pad)),
                  min(image.width, math.ceil(x2 + pad)), min(image.height, math.ceil(y2 + pad)))
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError("Пустая область поля")
        crop = image.crop(bounds).convert("RGB")
        factor = min(max(1.0, scale), max(1.0, config.MAX_IMAGE_DIMENSION / max(crop.size)))
        if factor > 1:
            crop = crop.resize((round(crop.width * factor), round(crop.height * factor)), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def get_bbox_crop_views(image: Image.Image, box, scale: float = 2.0) -> List[bytes]:
        """Return color, softly enhanced and high-contrast grayscale field views."""
        original = DocumentLoader.get_bbox_crop(image, box, scale)
        crop = Image.open(io.BytesIO(original)).convert("RGB")
        enhanced = DocumentLoader.preprocess_for_handwriting_img(crop, auto_crop=False)
        enhanced_buffer = io.BytesIO()
        enhanced.save(enhanced_buffer, format="PNG")

        grayscale = ImageOps.autocontrast(crop.convert("L"), cutoff=1)
        grayscale = ImageEnhance.Contrast(grayscale).enhance(1.65)
        grayscale_buffer = io.BytesIO()
        grayscale.save(grayscale_buffer, format="PNG")
        return [original, enhanced_buffer.getvalue(), grayscale_buffer.getvalue()]
