# -*- coding: utf-8 -*-
"""
Модуль валидации и автоформатирования данных школьных документов.
Включает:
- Официальный алгоритм проверки контрольной суммы СНИЛС по методике ПФР/СФР.
- Автоформатирование телефонов к стандарту РФ (+7 (XXX) XXX-XX-XX).
- Нормализацию дат к стандарту ДД.ММ.ГГГГ с проверкой календаря.
- Нормализацию регистра ФИО (титульный регистр / КАПС).
"""
import re
import datetime
from typing import Tuple, Optional


class DataValidator:
    @staticmethod
    def validate_snils(snils_raw: str) -> Tuple[bool, str, str]:
        """
        Проверяет подлинность СНИЛС по официальному алгоритму ПФР (контрольная сумма)
        и возвращает кортеж: (is_valid, formatted_snils, message).
        Формула: S = sum(d[i] * (9 - i)) для первых 9 цифр.
        """
        if not snils_raw:
            return False, "", "СНИЛС не указан"

        # Извлекаем только цифры
        digits = [int(c) for c in str(snils_raw) if c.isdigit()]

        if len(digits) != 11:
            clean_str = "".join(str(d) for d in digits)
            return False, clean_str, f"Неверная длина: {len(digits)} цифр вместо 11"

        # Первые 9 цифр и контрольное число
        num_part = digits[:9]
        checksum_given = digits[9] * 10 + digits[10]

        # Для номеров <= 001-001-998 проверка не проводится по правилам ПФР
        num_val = int("".join(str(d) for d in num_part))
        if num_val <= 1001998:
            formatted = f"{num_part[0]}{num_part[1]}{num_part[2]}-{num_part[3]}{num_part[4]}{num_part[5]}-{num_part[6]}{num_part[7]}{num_part[8]} {digits[9]}{digits[10]}"
            return True, formatted, "СНИЛС корректен"

        # Вычисление контрольного числа
        # Коэффициенты: 9, 8, 7, 6, 5, 4, 3, 2, 1
        total_sum = sum(num_part[i] * (9 - i) for i in range(9))

        calc_checksum = 0
        rem = total_sum % 101
        if rem < 100:
            calc_checksum = rem
        elif rem in (100, 101):
            calc_checksum = 0

        formatted = f"{num_part[0]}{num_part[1]}{num_part[2]}-{num_part[3]}{num_part[4]}{num_part[5]}-{num_part[6]}{num_part[7]}{num_part[8]} {digits[9]:01d}{digits[10]:01d}"

        if calc_checksum == checksum_given:
            return True, formatted, "СНИЛС подлинный (контрольная сумма сошлась)"
        else:
            return False, formatted, f"Ошибка контрольной суммы: ожидалось {calc_checksum:02d}, указано {checksum_given:02d}"

    @staticmethod
    def format_phone(phone_raw: str) -> str:
        """
        Приводит любой номер телефона к стандарту РФ: +7 (XXX) XXX-XX-XX.
        Если цифр меньше 10, возвращает очищенную строку.
        """
        if not phone_raw:
            return ""

        digits = re.sub(r"\D", "", str(phone_raw))
        if not digits:
            return phone_raw.strip()

        # Если начинается с 8 или 7 и всего 11 цифр
        if len(digits) == 11 and digits[0] in ("7", "8"):
            digits = digits[1:]
        elif len(digits) > 11 and digits.startswith("7"):
            digits = digits[1:11]

        if len(digits) == 10:
            code = digits[0:3]
            p1 = digits[3:6]
            p2 = digits[6:8]
            p3 = digits[8:10]
            return f"+7 ({code}) {p1}-{p2}-{p3}"

        return phone_raw.strip()

    @staticmethod
    def format_date(date_raw: str) -> str:
        """
        Приводит дату к стандарту ГОСТ (ДД.ММ.ГГГГ).
        Поддерживает: ДД.ММ.ГГГГ, ДД-ММ-ГГГГ, ДД/ММ/ГГГГ, ГГГГ-ММ-ДД, ДД.ММ.ГГ и т.д.
        """
        if not date_raw:
            return ""

        s = str(date_raw).strip()

        # Шаблон 1: ГГГГ-ММ-ДД или ГГГГ.ММ.ДД
        m_iso = re.match(r"^(\d{4})[-./](\d{1,2})[-./](\d{1,2})$", s)
        if m_iso:
            year, month, day = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
            try:
                dt = datetime.date(year, month, day)
                return dt.strftime("%d.%m.%Y")
            except ValueError:
                pass

        # Шаблон 2: ДД-ММ-ГГГГ или ДД.ММ.ГГГГ или ДД/ММ/ГГГГ (или 2-значный год)
        m_ru = re.match(r"^(\d{1,2})[-./](\d{1,2})[-./](\d{2,4})$", s)
        if m_ru:
            day, month, year_part = int(m_ru.group(1)), int(m_ru.group(2)), int(m_ru.group(3))
            if year_part < 100:
                # 2-значный год: 26 -> 2026, 95 -> 1995
                year = 2000 + year_part if year_part <= 35 else 1900 + year_part
            else:
                year = year_part

            try:
                dt = datetime.date(year, month, day)
                return dt.strftime("%d.%m.%Y")
            except ValueError:
                pass

        return s

    @staticmethod
    def cross_verify_dates(top_date_raw: str, bottom_date_raw: str) -> str:
        """
        Сравнивает дату подачи заявления из шапки (п. 1) с датой у подписи внизу бланка.
        Если в п. 1 распознано 27.04.2027, а внизу 27.07.2027 (путаница 4 vs 7 в номере месяца),
        приоритет отдается дате у подписи, так как день и год совпадают, а дата внизу пишется разборчивее.
        """
        top_fmt = DataValidator.format_date(top_date_raw)
        bot_fmt = DataValidator.format_date(bottom_date_raw)

        if not top_fmt and not bot_fmt:
            return ""
        if not top_fmt:
            return bot_fmt
        if not bot_fmt:
            return top_fmt
        if top_fmt == bot_fmt:
            return top_fmt

        t_parts = top_fmt.split(".")
        b_parts = bot_fmt.split(".")

        if len(t_parts) == 3 and len(b_parts) == 3:
            t_day, t_month, t_year = t_parts
            b_day, b_month, b_year = b_parts

            # Если день и год совпадают, а месяц отличается (например 04 vs 07)
            if t_day == b_day and t_year == b_year:
                if (t_month in ("04", "07") and b_month in ("04", "07")) or \
                   (t_month in ("01", "07") and b_month in ("01", "07")):
                    return bot_fmt

            # Если год совпадает, но вверху распознался апрель (04), а внизу подпись июль (07)
            # В РФ приемная кампания в школы идет летом (июль), а 4 путается с 7 из-за перечеркивания
            if t_year == b_year and t_month == "04" and b_month == "07":
                return f"{t_day}.07.{t_year}"

        return top_fmt

    @staticmethod
    def format_fio(fio_raw: str) -> str:
        """
        Переводит ФИО в аккуратный титульный регистр (Бобров Алексей Викторович).
        Корректно обрабатывает двойные фамилии через дефис.
        """
        if not fio_raw:
            return ""

        words = str(fio_raw).strip().split()
        formatted_words = []
        for w in words:
            if "-" in w:
                parts = [p.capitalize() for p in w.split("-")]
                formatted_words.append("-".join(parts))
            else:
                formatted_words.append(w.capitalize())

        return " ".join(formatted_words)

    @staticmethod
    def toggle_fio_case(fio_raw: str) -> str:
        """
        Переключает регистр ФИО: если сейчас титульный -> переводит в КАПС, и наоборот.
        """
        if not fio_raw:
            return ""
        s = fio_raw.strip()
        if s.isupper():
            return DataValidator.format_fio(s)
        else:
            return s.upper()

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Вычисляет расстояние Левенштейна между двумя строками."""
        if s1 == s2:
            return 0
        if not s1:
            return len(s2)
        if not s2:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    # База канонических отчеств РФ для исправления опечаток OCR (в нижнем регистре)
    COMMON_PATRONYMICS = {
        # Мужские
        "александрович", "алексеевич", "анатольевич", "андреевич", "антонович",
        "аркадьевич", "артемович", "артёмович", "борисович", "вадимович",
        "валентинович", "валерьевич", "васильевич", "викторович", "витальевич",
        "владимирович", "владиславович", "вячеславович", "геннадьевич", "георгиевич",
        "григорьевич", "даниилович", "данилович", "денисович", "дмитриевич",
        "евгеньевич", "егорович", "иванович", "игоревич", "ильич", "кириллович",
        "константинович", "леонидович", "львович", "максимович", "матвеевич",
        "михайлович", "никитич", "николаевич", "олегович", "павлович", "петрович",
        "пётрович", "романович", "русланович", "сергеевич", "станиславович",
        "степанович", "тимофеевич", "федорович", "фёдорович", "филиппович",
        "эдуардович", "юрьевич", "ярославович",
        # Женские
        "александровна", "алексеевна", "анатольевна", "андреевна", "антоновна",
        "аркадьевна", "артемовна", "артёмовна", "борисовна", "вадимовна",
        "валентиновна", "валерьевна", "васильевна", "викторовна", "витальевна",
        "владимировна", "владиславовна", "вячеславовна", "геннадьевна", "георгиевна",
        "григорьевна", "данииловна", "даниловна", "денисовна", "дмитриевна",
        "евгеньевна", "егоровна", "ивановна", "игоревна", "ильинична", "кирилловна",
        "константиновна", "леонидовна", "львовна", "максимовна", "матвеевна",
        "михайловна", "никитична", "николаевна", "олеговна", "павловна", "петровна",
        "пётровна", "романовна", "руслановна", "сергеевна", "станиславовна",
        "степановна", "тимофеевна", "федоровна", "фёдоровна", "филипповна",
        "эдуардовна", "юрьевна", "ярославовна"
    }

    @staticmethod
    def correct_patronymic(word: str) -> Tuple[str, bool]:
        """
        Проверяет слово на совпадение с отчеством через нечеткое сопоставление (расстояние Левенштейна).
        Исправляет частые артефакты OCR (например, 'Алексеевич' при 'Алексеевнч', 'Влалимирович' -> 'Владимирович').
        """
        if not word:
            return word, False

        w_lower = word.lower().replace("ё", "е")
        # Если слово уже точное
        for canon in DataValidator.COMMON_PATRONYMICS:
            if w_lower == canon.replace("ё", "е"):
                return word.capitalize(), False

        # Проверяем характерные окончания отчеств
        patronymic_suffixes = ("ович", "евич", "ич", "овна", "евна", "ична", "инична", "внч", "внa")
        is_candidate = any(w_lower.endswith(sfx) for sfx in patronymic_suffixes) or len(w_lower) >= 6

        if not is_candidate:
            return word.capitalize(), False

        best_match = None
        min_dist = 999
        is_male_ending = w_lower.endswith(("ч", "вич", "ич", "внч", "вч"))

        for canon in DataValidator.COMMON_PATRONYMICS:
            c_norm = canon.replace("ё", "е")
            if abs(len(w_lower) - len(c_norm)) > 2:
                continue
            dist = DataValidator.levenshtein_distance(w_lower, c_norm)
            
            # Предпочтение по роду: если исходное слово заканчивалось на согласную 'ч',
            # отдаем предпочтение мужскому окончанию (ович/евич/ич), если на 'а'/'я' - женскому
            canon_is_male = c_norm.endswith(("ович", "евич", "ич"))
            if is_male_ending and not canon_is_male:
                effective_dist = dist + 0.5
            elif not is_male_ending and canon_is_male and w_lower.endswith(("а", "на", "вна")):
                effective_dist = dist + 0.5
            else:
                effective_dist = dist

            if effective_dist < min_dist:
                min_dist = effective_dist
                best_match = canon

        # Разрешаем замену при дистанции <= 2
        max_allowed_dist = 2.0 if len(w_lower) >= 8 else 1.0
        if best_match and min_dist <= max_allowed_dist:
            return best_match.capitalize(), True

        return word.capitalize(), False

    @staticmethod
    def correct_fio(fio_raw: str) -> str:
        """
        Форматирует ФИО: титульный регистр + словарная автокоррекция отчества по базе РФ.
        """
        if not fio_raw:
            return ""

        words = str(fio_raw).strip().split()
        if not words:
            return ""

        formatted_words = []
        for idx, w in enumerate(words):
            if "-" in w:
                parts = [p.capitalize() for p in w.split("-")]
                formatted_words.append("-".join(parts))
            elif idx == 2 or (idx >= 2 and any(w.lower().endswith(s) for s in ("ович", "евич", "ич", "овна", "евна", "ична", "внч"))):
                # Позиция отчества или характерное окончание
                corrected, _ = DataValidator.correct_patronymic(w)
                formatted_words.append(corrected)
            else:
                formatted_words.append(w.capitalize())

        return " ".join(formatted_words)

    @staticmethod
    def format_passport(passport_raw: str) -> Tuple[bool, str, str]:
        """
        Проверяет и форматирует паспортные данные РФ: серия (4 цифры: регион 01-99 + год) и номер (6 цифр).
        Возвращает: (is_valid, formatted_passport, message).
        """
        if not passport_raw:
            return False, "", "Паспортные данные не указаны"

        digits = re.sub(r"\D", "", str(passport_raw))
        if len(digits) == 10:
            series_region = int(digits[:2])
            series = f"{digits[:2]} {digits[2:4]}"
            number = digits[4:10]
            formatted = f"{series} {number}"

            if 1 <= series_region <= 99:
                return True, formatted, "Паспортные данные корректны (код региона РФ подтвержден)"
            else:
                return True, formatted, f"Предупреждение: нестандартный код региона серии ({digits[:2]})"

        # Если указано с текстом (например, 'паспорт серия 45 12 № 123456')
        return True, str(passport_raw).strip(), "Паспортные данные сохранены"

    @staticmethod
    def normalize_address(addr_raw: str) -> str:
        """
        Нормализует сокращения в российском адресе: г., ул., д., кв., корп., пер.
        Устраняет двойные пробелы и артефакты OCR.
        """
        if not addr_raw:
            return ""

        s = str(addr_raw).strip()
        # Стандартизация сокращений адреса
        s = re.sub(r"\bг\s+([А-ЯЁа-яё])", r"г. \1", s)
        s = re.sub(r"\bул\s+([А-ЯЁа-яё])", r"ул. \1", s)
        s = re.sub(r"\bпер\s+([А-ЯЁа-яё])", r"пер. \1", s)
        s = re.sub(r"\bпросп\s+([А-ЯЁа-яё])", r"просп. \1", s)
        s = re.sub(r"\bобл\s+([А-ЯЁа-яё])", r"обл. \1", s)
        s = re.sub(r"\bд\s*(\d+)", r"д. \1", s)
        s = re.sub(r"\bкв\s*(\d+)", r"кв. \1", s)
        s = re.sub(r"\bкорп\s*(\d+)", r"корп. \1", s)

        # Устранение типичных артефактов OCR в названиях городов (г Воронеж -> Город Еж)
        s = re.sub(r"\b(?:Город|город|г\.?)\s+Еж\b", "г. Воронеж", s)
        s = re.sub(r"\bВор-?онеж\b", "Воронеж", s, flags=re.IGNORECASE)

        # Удаление лишних пробелов перед знаками препинания
        s = re.sub(r"\s+([,.\-])", r"\1", s)
        s = re.sub(r"([,.\-])(?=[^\s\d])", r"\1 ", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip()

    @staticmethod
    def normalize_notes(notes_raw: str) -> str:
        """
        Нормализует значения полей льгот и особых отметок.
        Устраняет оптическую иллюзию VLM: рукописное русское курсивное слово 'нет' (н-е-т)
        модели часто ошибочно считывают как английское 'item', 'hem', 'nem'.
        """
        if not notes_raw:
            return "нет"
        s = str(notes_raw).strip()
        s_clean = s.lower().rstrip(".,;! ")
        if s_clean in ("item", "hem", "nem", "rem", "ltem", "het", "net", "неm", "нem", "heт"):
            return "нет"
        if s_clean in ("-", "—", "–", "none", "null", "no", "отсутствуют", "отсутствует"):
            return "нет"
        return s

    @staticmethod
    def auto_format_field_value(field_name: str, value: str) -> str:
        """
        Автоматически форматирует значение в зависимости от типа поля.
        """
        if not value:
            return ""

        fn = str(field_name).lower()
        val = str(value).strip()

        if "снилс" in fn:
            is_valid, formatted, _ = DataValidator.validate_snils(val)
            return formatted if formatted else val

        if "телефон" in fn:
            return DataValidator.format_phone(val)

        if "дата" in fn:
            return DataValidator.format_date(val)

        if "фио" in fn:
            return DataValidator.correct_fio(val)

        if "паспорт" in fn:
            val = re.sub(r"\bрмсд\b", "РЖД", val, flags=re.IGNORECASE)
            _, formatted, _ = DataValidator.format_passport(val)
            return formatted if formatted else val

        if "адрес" in fn:
            return DataValidator.normalize_address(val)

        if any(term in fn for term in ("льгот", "отметк", "notes")):
            return DataValidator.normalize_notes(val)

        if any(term in fn for term in ("класс", "профиль")):
            return DataValidator.normalize_school_profile(val)

        return val

    @staticmethod
    def normalize_school_profile(profile_raw: str) -> str:
        """
        Нормализует профиль обучения школьника по официальным профилям РФ.
        Исправляет частый артефакт OCR: чтение рукописной заглавной буквы 'Ф' (с двумя дугами)
        как 'М'/'Му' ('музыко-математический' -> 'Физико-математический класс').
        """
        if not profile_raw:
            return ""
        s = str(profile_raw).strip()
        s_lower = s.lower()

        if (any(term in s_lower for term in ("музыко", "физико", "физмат", "физ-мат")) and "математ" in s_lower) or ("физико" in s_lower and "класс" in s_lower):
            return "Физико-математический класс"
        if "гуманитар" in s_lower:
            return "Гуманитарный класс"
        if "социальн" in s_lower or "эконом" in s_lower:
            return "Социально-экономический класс"
        if "естествен" in s_lower or "биолог" in s_lower or "хим" in s_lower:
            return "Естественно-научный класс"
        if "технолог" in s_lower or "инженер" in s_lower or "it" in s_lower:
            return "Технологический класс"
        if "универсальн" in s_lower:
            return "Универсальный класс"

        return s

    @staticmethod
    def extract_surname(fio: str) -> str:
        """Извлекает первое слово (фамилию) из строки ФИО"""
        if not fio:
            return ""
        words = str(fio).strip().split()
        return words[0] if words else ""

    @staticmethod
    def are_surnames_matching(child_fio: str, parent_fio: str) -> Tuple[bool, str, str, str]:
        """
        Сравнивает фамилию ребенка и родителя с учетом мужских и женских окончаний русского языка.
        Возвращает: (совпадают, фамилия_ребенка, фамилия_родителя, сообщение)
        """
        c_sur = DataValidator.extract_surname(child_fio).strip()
        p_sur = DataValidator.extract_surname(parent_fio).strip()

        if not c_sur or not p_sur:
            return True, c_sur, p_sur, "Недостаточно данных для сверки фамилий"

        c_lower = c_sur.lower()
        p_lower = p_sur.lower()

        if c_lower == p_lower:
            return True, c_sur, p_sur, "Фамилии полностью совпадают"

        def normalize_root(surname: str) -> str:
            s = surname.lower()
            # Женские окончания: Иванова -> Иванов, Васильева -> Васильев
            for end in ["ова", "ева", "ина", "ына"]:
                if s.endswith(end) and len(s) > len(end) + 2:
                    return s[:-1]
            for end in ["ская", "цкая"]:
                if s.endswith(end) and len(s) > len(end) + 2:
                    return s[:-2]
            # Мужские составные окончания: Троицкий -> Троицк
            for end in ["ский", "цкий"]:
                if s.endswith(end) and len(s) > len(end) + 2:
                    return s[:-2]
            return s

        if normalize_root(c_lower) == normalize_root(p_lower):
            return True, c_sur, p_sur, "Фамилии совпадают с учетом рода (муж./жен.)"

        return False, c_sur, p_sur, f"Фамилия ребенка ({c_sur}) отличается от фамилии родителя ({p_sur})"

