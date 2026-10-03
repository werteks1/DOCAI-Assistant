"""
Конфигурационный файл проекта DocAI Assistant
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEST_SAMPLES_DIR = BASE_DIR / "test_samples"
TEST_SAMPLES_DIR.mkdir(exist_ok=True)

OLLAMA_HOST = os.getenv("DOCIA_OLLAMA_HOST", "http://192.168.0.19:1234")
DEFAULT_MODEL = os.getenv("DOCIA_MODEL", "qwen2.5-vl:7b")
SILICONFLOW_HOST = "https://api.siliconflow.com"

# --- Авторизация и развёртывание -------------------------------------
# LAN-режим (несколько школьных ПК): компилятор слушает 0.0.0.0 и сам отдаёт
# собранный фронтенд из web/dist. Dev-режим оставляет BIND_HOST=127.0.0.1.
BIND_HOST = os.getenv("DOCIA_BIND_HOST", "0.0.0.0")
BIND_PORT = int(os.getenv("DOCIA_BIND_PORT", "8000"))
WEB_DIST_DIR = Path(os.getenv("DOCIA_WEB_DIST", str(BASE_DIR / "web" / "dist")))

# SQLite-хранилище учёток, сессий и журнала активности (файл в .gitignore).
DOCIA_DB = Path(os.getenv("DOCIA_DB", str(BASE_DIR / "docia.db")))
# Время жизни сессии (часы); скользящее — продлевается при каждом обращении.
SESSION_TTL_HOURS = int(os.getenv("DOCIA_SESSION_TTL_HOURS", "24"))
MAX_PDF_PAGES = 20
MAX_IMAGE_PIXELS = 25_000_000
# Верхняя граница на base64-строку одного изображения/PDF в запросе. Ограничивает
# память до декодирования (защита от раздувания тела запроса). ~80 МБ base64 ≈
# 60 МБ бинарных данных — с запасом на 20-страничный скан-PDF.
MAX_IMAGE_BASE64_LEN = 80 * 1024 * 1024

DEFAULT_EXCEL_COLUMNS = [
    "№ п/п",
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
    "Имя файла источника",
    "Статус проверки"
]

# Мягкая предобработка (без артефактов пережима курсива)
IMAGE_ENHANCE_CONTRAST = 1.15   # Мягкий контраст, чтобы не утолщать петли букв
IMAGE_ENHANCE_SHARPNESS = 1.15  # Легкая резкость без шума и ореолов вокруг букв
MAX_IMAGE_DIMENSION = 2800      # Высокое разрешение для максимальной детализации почерка

# Optional local detector; unavailable engines fall back to VLM.
USE_PADDLE_OCR = True
FIELD_CROP_SCALE = 3.0
# Полный лист всегда сначала читает VLM. В режиме auto Paddle только находит
# области критичных полей для независимой multi-view проверки той же VLM.
USE_PADDLE_FIELD_CROPS = True
USE_PADDLE_OCR_ASSIST = True
PADDLE_VERIFY_FIELDS = (
    "Дата подачи заявления",
    "ФИО поступающего ученика",
    "Дата рождения ребенка",
    "ФИО родителя / заявителя",
    "Паспортные данные",
    "Контактный телефон",
    "СНИЛС поступающего",
)
# Сохраняем проверенный слой форматирования после распознавания.
USE_VALIDATOR = True
USE_REFERENCE_DICTIONARY = True
PADDLE_DETECTOR = "PP-OCRv6_medium_det"
PADDLE_DETECTOR_OPTIONS = ["PP-OCRv6_medium_det", "PP-OCRv6_small_det"]

# --- Логирование (PII-safe) -----------------------------------------
def get_log_level() -> str:
    """Уровень лога из DOCIA_LOG_LEVEL (имя уровня logging, например INFO)."""
    return os.getenv("DOCIA_LOG_LEVEL", "INFO").strip() or "INFO"


def get_log_file() -> str:
    """Путь файла лога из DOCIA_LOG_FILE. Пусто — только stderr."""
    return os.getenv("DOCIA_LOG_FILE", "").strip()


# --- Шаблон бланка ---------------------------------------------------
def get_template_name() -> str:
    """Имя активного шаблона бланка из DOCIA_TEMPLATE (пусто = по умолчанию)."""
    return os.getenv("DOCIA_TEMPLATE", "").strip()
