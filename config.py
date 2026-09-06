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
# По умолчанию разрешены только loopback и RFC1918-сети для локального LAN-сервера.
ALLOWED_AI_NETWORKS = ["127.0.0.0/8", "::1/128", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]
MAX_PDF_PAGES = 20
MAX_IMAGE_PIXELS = 25_000_000

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
# Полосовой режим требует настройки координат под конкретный шаблон.
# По умолчанию сохраняем проверенный полный проход VLM с контекстом страницы.
USE_PADDLE_FIELD_CROPS = True
USE_PADDLE_OCR_ASSIST = True
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
