"""
Модуль для создания, профессионального форматирования и ведения Excel (.xlsx).
Включает чередование строк ("зебру"), цветовую индикацию статусов, динамические колонки и автоподбор ширины.
"""
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional
import re
import io
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

import config


class ExcelManager:
    HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    
    ROW_EVEN_FILL = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
    ROW_ODD_FILL = PatternFill(start_color="F7F9FC", end_color="F7F9FC", fill_type="solid")
    
    STATUS_VERIFIED_FILL = PatternFill(start_color="E8F5E9", end_color="E8F5E9", fill_type="solid")
    STATUS_VERIFIED_FONT = Font(name="Calibri", size=10, bold=True, color="2E7D32")
    
    DATA_FONT = Font(name="Calibri", size=11, color="202020")
    BORDER_THIN = Border(
        left=Side(style='thin', color='E0E0E0'),
        right=Side(style='thin', color='E0E0E0'),
        top=Side(style='thin', color='E0E0E0'),
        bottom=Side(style='thin', color='E0E0E0')
    )

    @staticmethod
    def _safe_excel_value(value: Any) -> Any:
        """Не допускает интерпретацию данных OCR как формулы Excel/DDE."""
        if value is None:
            return ""
        if not isinstance(value, str):
            return value
        value = value.replace("\x00", "").replace("\r", " ").replace("\n", " ").strip()
        if value and value[0] in "=+-@\t":
            return "'" + value
        return value

    @staticmethod
    def create_new_workbook(file_path: Union[str, Path], columns: List[str] = None) -> Path:
        """Создает новый профессионально оформленный журнал Excel"""
        path = Path(file_path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if not columns:
            columns = config.DEFAULT_EXCEL_COLUMNS

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Журнал заявлений"
        ws.views.sheetView[0].showGridLines = True

        for col_idx, col_name in enumerate(columns, start=1):
            cell = ws.cell(row=1, column=col_idx, value=col_name)
            cell.fill = ExcelManager.HEADER_FILL
            cell.font = ExcelManager.HEADER_FONT
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = ExcelManager.BORDER_THIN

        ws.row_dimensions[1].height = 30
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}1"

        ExcelManager._autofit_columns(ws)
        wb.save(path)
        return path

    @staticmethod
    def export_records_bytes(records: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> bytes:
        """Формирует журнал в памяти для скачивания из Web API."""
        if not records:
            raise ValueError("Нет записей для экспорта")
        columns = columns or [
            "№ п/п", *[c for c in config.DEFAULT_EXCEL_COLUMNS if c not in {"№ п/п", "Имя файла источника", "Статус проверки"}],
            "Имя файла источника", "Статус проверки",
        ]
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Журнал заявлений"
        for idx, name in enumerate(columns, 1):
            cell = ws.cell(1, idx, ExcelManager._safe_excel_value(name))
            cell.fill = ExcelManager.HEADER_FILL
            cell.font = ExcelManager.HEADER_FONT
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = ExcelManager.BORDER_THIN
        ws.row_dimensions[1].height = 30
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}1"
        for row_number, record in enumerate(records, 2):
            fill = ExcelManager.ROW_ODD_FILL if row_number % 2 else ExcelManager.ROW_EVEN_FILL
            for idx, name in enumerate(columns, 1):
                value = row_number - 1 if name == "№ п/п" else record.get(name, "")
                cell = ws.cell(row_number, idx, ExcelManager._safe_excel_value(value))
                cell.font = ExcelManager.DATA_FONT
                cell.fill = fill
                cell.border = ExcelManager.BORDER_THIN
                cell.alignment = Alignment(vertical="center", wrap_text=True)
                if "статус" in name.lower():
                    cell.fill = ExcelManager.STATUS_VERIFIED_FILL
                    cell.font = ExcelManager.STATUS_VERIFIED_FONT
        ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{max(2, len(records) + 1)}"
        ExcelManager._autofit_columns(ws)
        output = io.BytesIO()
        wb.save(output)
        wb.close()
        return output.getvalue()

    @staticmethod
    def get_columns(file_path: Union[str, Path]) -> List[str]:
        """Считывает заголовки из первой строки файла"""
        path = Path(file_path)
        if not path.exists():
            return []

        wb = openpyxl.load_workbook(path, read_only=True)
        ws = wb.active
        headers = []
        for row in ws.iter_rows(min_row=1, max_row=1, values_only=True):
            for cell_value in row:
                if cell_value is not None:
                    headers.append(str(cell_value).strip())
        wb.close()
        return headers

    @staticmethod
    def read_all_rows(file_path: Union[str, Path]) -> Tuple[List[str], List[List[str]]]:
        """Считывает заголовки и все строки таблицы для отображения в GUI"""
        path = Path(file_path)
        if not path.exists():
            return [], []

        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb.active
        headers = []
        rows = []

        for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
            if row_idx == 1:
                headers = [str(c or "").strip() for c in row if c is not None]
            else:
                if any(c is not None and str(c).strip() for c in row):
                    row_vals = [str(c) if c is not None else "" for c in row[:len(headers)]]
                    rows.append(row_vals)

        wb.close()
        return headers, rows

    @staticmethod
    def _normalize_header(name: str) -> str:
        """Нормализует строку заголовка для сравнения"""
        if not name:
            return ""
        return re.sub(r'[^a-zа-яё0-9]', '', str(name).lower())

    @staticmethod
    def find_matching_column(existing_columns: Dict[str, int], field_name: str) -> Optional[int]:
        """
        Умно находит индекс существующей колонки для входящего поля field_name.
        Учитывает точные совпадения, нормализацию и семантические группы синонимов.
        """
        if field_name in existing_columns:
            return existing_columns[field_name]

        norm_field = ExcelManager._normalize_header(field_name)

        # 1. Совпадение по очищенному имени
        for col_name, col_idx in existing_columns.items():
            if ExcelManager._normalize_header(col_name) == norm_field:
                return col_idx

        # 2. Семантические группы синонимов
        def get_semantic_group(s: str) -> Optional[str]:
            norm = ExcelManager._normalize_header(s)
            if any(k in norm for k in ["родител", "заявител", "матер", "отц", "представител"]):
                return "фио_родителя"
            if "снилс" in norm:
                return "снилс"
            if "рождени" in norm:
                return "дата_рождения"
            if any(k in norm for k in ["подач", "заявлени", "согласи", "обращен"]):
                return "дата_подачи"
            if "паспорт" in norm:
                return "паспорт"
            if any(k in norm for k in ["адрес", "проживан", "регистрац"]):
                return "адрес"
            if any(k in norm for k in ["телефон", "связ", "мобильн"]):
                return "телефон"
            if any(k in norm for k in ["класс", "профиль", "направлен", "литер"]):
                return "класс"
            if any(k in norm for k in ["льгот", "особ"]):
                return "льготы"
            if any(k in norm for k in ["файл", "источник"]):
                return "файл"
            if "статус" in norm:
                return "статус"
            if "предмет" in norm:
                return "предметы"
            if "подпис" in norm:
                return "подпись"
            if any(k in norm for k in ["фио", "ученик", "ребенок", "учащ", "поступающ"]):
                return "фио_ученика"
            return None

        target_group = get_semantic_group(field_name)
        if target_group:
            for col_name, col_idx in existing_columns.items():
                if get_semantic_group(col_name) == target_group:
                    return col_idx

        return None

    @staticmethod
    def append_record(
        file_path: Union[str, Path],
        record_data: Dict[str, Any],
        custom_columns: Optional[List[str]] = None
    ) -> int:
        """
        Добавляет строку в журнал с автоформатированием 'зебры', автоподбором колонок и защитой от дублей.
        """
        path = Path(file_path)
        if not path.exists():
            if custom_columns:
                cols = ["№ п/п"] + [c for c in custom_columns if not c.lower().startswith("№")] + ["Имя файла источника", "Статус проверки"]
            else:
                cols = ["№ п/п"] + [k for k in record_data.keys() if k not in ["№ п/п", "Имя файла источника", "Статус проверки"]] + ["Имя файла источника", "Статус проверки"]
            ExcelManager.create_new_workbook(path, columns=cols)

        wb = openpyxl.load_workbook(path)
        ws = wb.active

        col_mapping = {}
        for col_idx in range(1, ws.max_column + 1):
            header = ws.cell(row=1, column=col_idx).value
            if header:
                col_mapping[str(header).strip()] = col_idx

        # Сопоставляем входящие поля со столбцами таблицы
        field_to_col_idx = {}
        for field, value in record_data.items():
            if field.lower().startswith("№") or "п/п" in field.lower():
                continue

            matched_col = ExcelManager.find_matching_column(col_mapping, field)
            if matched_col is not None:
                field_to_col_idx[field] = matched_col
            else:
                # Новое поле, которого еще нет: вставляем ПЕРЕД служебными колонками (файл, статус)
                status_cols = [c for c, idx in col_mapping.items() if "файл" in c.lower() or "статус" in c.lower()]
                if status_cols:
                    insert_pos = min(col_mapping[c] for c in status_cols)
                    ws.insert_cols(insert_pos)
                    cell = ws.cell(row=1, column=insert_pos, value=ExcelManager._safe_excel_value(field))
                    cell.fill = ExcelManager.HEADER_FILL
                    cell.font = ExcelManager.HEADER_FONT
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                    cell.border = ExcelManager.BORDER_THIN
                    for c_name in list(col_mapping.keys()):
                        if col_mapping[c_name] >= insert_pos:
                            col_mapping[c_name] += 1
                    col_mapping[field] = insert_pos
                    field_to_col_idx[field] = insert_pos
                else:
                    new_col_idx = ws.max_column + 1
                    cell = ws.cell(row=1, column=new_col_idx, value=ExcelManager._safe_excel_value(field))
                    cell.fill = ExcelManager.HEADER_FILL
                    cell.font = ExcelManager.HEADER_FONT
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                    cell.border = ExcelManager.BORDER_THIN
                    col_mapping[field] = new_col_idx
                    field_to_col_idx[field] = new_col_idx

        new_row = ws.max_row + 1
        is_odd_row = (new_row % 2 != 0)
        row_fill = ExcelManager.ROW_ODD_FILL if is_odd_row else ExcelManager.ROW_EVEN_FILL

        # Заполняем всю строку базовыми стилями
        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=new_row, column=col_idx)
            cell.font = ExcelManager.DATA_FONT
            cell.border = ExcelManager.BORDER_THIN
            cell.fill = row_fill
            cell.alignment = Alignment(vertical="center")

        # Номер п/п
        next_index = max(1, new_row - 1)
        first_cell = ws.cell(row=new_row, column=1, value=next_index)
        first_cell.alignment = Alignment(horizontal="center", vertical="center")

        # Записываем значения данных
        for field, value in record_data.items():
            if field.lower().startswith("№") or "п/п" in field.lower():
                continue
            col_idx = field_to_col_idx.get(field)
            if col_idx:
                cell = ws.cell(row=new_row, column=col_idx, value=ExcelManager._safe_excel_value(value))
                header_val = str(ws.cell(row=1, column=col_idx).value or "").lower()
                if "статус" in header_val:
                    cell.fill = ExcelManager.STATUS_VERIFIED_FILL
                    cell.font = ExcelManager.STATUS_VERIFIED_FONT
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                elif any(k in header_val for k in ["дата", "телефон", "снилс", "№"]):
                    cell.alignment = Alignment(horizontal="center", vertical="center")

        ws.row_dimensions[new_row].height = 24
        ExcelManager._autofit_columns(ws)
        wb.save(path)
        wb.close()
        return new_row
    @staticmethod
    def delete_rows(file_path: Union[str, Path], row_indices: List[int]) -> bool:
        """Удаляет список строк из таблицы Excel (row_idx >= 2) в обратном порядке и пересчитывает № п/п"""
        path = Path(file_path)
        if not path.exists():
            return False

        wb = openpyxl.load_workbook(path)
        ws = wb.active

        valid_indices = sorted(set([idx for idx in row_indices if 2 <= idx <= ws.max_row]), reverse=True)
        if not valid_indices:
            wb.close()
            return False

        for row_idx in valid_indices:
            ws.delete_rows(row_idx)

        # Пересчитываем номера № п/п в первой колонке и переназначаем зебру
        for current_row in range(2, ws.max_row + 1):
            first_cell = ws.cell(row=current_row, column=1)
            first_cell.value = current_row - 1

            is_odd = (current_row % 2 != 0)
            row_fill = ExcelManager.ROW_ODD_FILL if is_odd else ExcelManager.ROW_EVEN_FILL

            for col_idx in range(1, ws.max_column + 1):
                cell = ws.cell(row=current_row, column=col_idx)
                header_val = str(ws.cell(row=1, column=col_idx).value or "").lower()
                if "статус" not in header_val:
                    cell.fill = row_fill

        wb.save(path)
        wb.close()
        return True

    @staticmethod
    def delete_row(file_path: Union[str, Path], row_idx: int) -> bool:
        """Удаляет одну строку из таблицы Excel (row_idx >= 2) и пересчитывает порядковые номера № п/п"""
        return ExcelManager.delete_rows(file_path, [row_idx])

    @staticmethod
    def update_row(
        file_path: Union[str, Path],
        row_idx: int,
        updated_data: Dict[str, Any]
    ) -> bool:
        """Обновляет значения ячеек указанной строки в Excel"""
        path = Path(file_path)
        if not path.exists():
            return False

        wb = openpyxl.load_workbook(path)
        ws = wb.active

        if row_idx < 2 or row_idx > ws.max_row:
            wb.close()
            return False

        col_mapping = {}
        for col_idx in range(1, ws.max_column + 1):
            header = ws.cell(row=1, column=col_idx).value
            if header:
                col_mapping[str(header).strip()] = col_idx

        for header, col_idx in col_mapping.items():
            if header in updated_data:
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.value = ExcelManager._safe_excel_value(updated_data[header])
                if "статус" in header.lower():
                    val = str(updated_data[header]).lower()
                    if "проверено" in val:
                        cell.fill = ExcelManager.STATUS_VERIFIED_FILL
                        cell.font = ExcelManager.STATUS_VERIFIED_FONT
                    else:
                        cell.fill = ExcelManager.ROW_EVEN_FILL

        ExcelManager._autofit_columns(ws)
        wb.save(path)
        wb.close()
        return True

    @staticmethod
    def _autofit_columns(ws):
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val = str(cell.value or "")
                if "\n" in val:
                    val = max(val.split("\n"), key=len)
                if len(val) > max_len:
                    max_len = len(val)
            ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 48)

    @staticmethod
    def find_duplicates(
        file_path: Union[str, Path],
        record_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Ищет потенциальные дубликаты в базе Excel:
        1. По точному совпадению СНИЛС (11 цифр).
        2. По совпадению ФИО ребенка/ученика + Дата рождения.
        """
        path = Path(file_path)
        if not path.exists():
            return []

        headers, rows = ExcelManager.read_all_rows(path)
        if not headers or not rows:
            return []

        # Индексы ключевых колонок в headers
        col_indices = {}
        for idx, h in enumerate(headers):
            norm = ExcelManager._normalize_header(h)
            if "снилс" in norm:
                col_indices["snils"] = idx
            elif any(k in norm for k in ["фиопоступающего", "фиоребенка", "фиоученика", "фио"]):
                if "родител" not in norm and "заявител" not in norm and "snils" not in col_indices:
                    col_indices["fio"] = idx
            elif "рождени" in norm:
                col_indices["birth_date"] = idx
            elif "файл" in norm:
                col_indices["file_name"] = idx

        # Извлекаем данные входящей записи
        incoming_snils_digits = ""
        incoming_fio_norm = ""
        incoming_birth_date = ""

        for k, v in record_data.items():
            norm_k = ExcelManager._normalize_header(k)
            if "снилс" in norm_k:
                incoming_snils_digits = "".join(c for c in str(v) if c.isdigit())
            elif any(w in norm_k for w in ["фиопоступающего", "фиоребенка", "фиоученика"]) or (norm_k == "фио"):
                incoming_fio_norm = ExcelManager._normalize_header(str(v))
            elif "рождени" in norm_k:
                incoming_birth_date = "".join(c for c in str(v) if c.isdigit())

        duplicates = []
        for orig_idx, row in enumerate(rows):
            excel_row_idx = orig_idx + 2
            rec_num = row[0] if row else str(orig_idx + 1)
            row_fio = row[col_indices["fio"]] if "fio" in col_indices and col_indices["fio"] < len(row) else ""
            row_snils = row[col_indices["snils"]] if "snils" in col_indices and col_indices["snils"] < len(row) else ""
            row_birth = row[col_indices["birth_date"]] if "birth_date" in col_indices and col_indices["birth_date"] < len(row) else ""
            row_file = row[col_indices["file_name"]] if "file_name" in col_indices and col_indices["file_name"] < len(row) else ""

            row_snils_digits = "".join(c for c in str(row_snils) if c.isdigit())
            row_fio_norm = ExcelManager._normalize_header(str(row_fio))
            row_birth_digits = "".join(c for c in str(row_birth) if c.isdigit())

            match_reason = None
            if incoming_snils_digits and len(incoming_snils_digits) == 11 and incoming_snils_digits == row_snils_digits:
                match_reason = f"Совпадение по СНИЛС ({row_snils})"
            elif incoming_fio_norm and row_fio_norm and (incoming_fio_norm == row_fio_norm):
                if incoming_birth_date and row_birth_digits and (incoming_birth_date == row_birth_digits):
                    match_reason = f"Совпадение по ФИО и дате рождения ({row_fio}, {row_birth})"
                elif not incoming_birth_date:
                    match_reason = f"Совпадение по ФИО ({row_fio})"

            if match_reason:
                duplicates.append({
                    "row_excel_idx": excel_row_idx,
                    "num": rec_num,
                    "fio": row_fio,
                    "snils": row_snils,
                    "birth_date": row_birth,
                    "file_name": row_file,
                    "match_reason": match_reason
                })

        return duplicates

    @staticmethod
    def _digits_only(value: Any) -> str:
        return "".join(c for c in str(value or "") if c.isdigit())

    @staticmethod
    def _record_duplicate_keys(record: Dict[str, Any]) -> Tuple[str, str, str]:
        """Ключи для поиска дублей по записи: (snils_цифры, fio_норм, дата_рождения_цифры)."""
        snils = ""
        fio = ""
        birth = ""
        for key, value in record.items():
            norm = ExcelManager._normalize_header(key)
            if "снилс" in norm:
                snils = ExcelManager._digits_only(value)
            elif any(w in norm for w in ["родител", "заявител", "представител"]):
                continue
            elif "рождени" in norm:
                birth = ExcelManager._digits_only(value)
            elif any(w in norm for w in ["фио", "ученик", "ребенок", "учащ", "поступающ"]) or norm == "фио":
                fio = ExcelManager._normalize_header(str(value))
        return snils, fio, birth

    @staticmethod
    def find_in_memory_duplicates(
        records: List[Dict[str, Any]],
        filenames: Optional[List[str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Ищет дубли в памяти внутри одной пачки результатов (без файла на диске).

        Возвращает список, выровненный по `records`: для каждой записи — список
        совпадений с более ранними записями пачки:
        [{index, duplicate_of, reason}, ...]. Первая запись группы остаётся оригиналом.
        """
        total = len(records)
        filenames = filenames or ["" for _ in range(total)]
        keys = [ExcelManager._record_duplicate_keys(record) for record in records]

        def _format_snils(digits: str) -> str:
            return f"{digits[:3]}-{digits[3:6]}-{digits[6:9]} {digits[9:]}"

        result: List[List[Dict[str, Any]]] = [[] for _ in range(total)]
        for current in range(total):
            cur_snils, cur_fio, cur_birth = keys[current]
            for earlier in range(current):
                prev_snils, prev_fio, prev_birth = keys[earlier]
                reason = None
                if cur_snils and len(cur_snils) == 11 and cur_snils == prev_snils:
                    reason = f"Совпадение по СНИЛС ({_format_snils(cur_snils)})"
                elif cur_fio and prev_fio and cur_fio == prev_fio:
                    if cur_birth and prev_birth and cur_birth == prev_birth:
                        reason = "Совпадение по ФИО и дате рождения"
                    elif not cur_birth or not prev_birth:
                        reason = "Совпадение по ФИО"
                if reason:
                    result[current].append({
                        "index": earlier,
                        "duplicate_of": filenames[earlier],
                        "reason": reason,
                    })
        return result

    @staticmethod
    def get_analytics_summary(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Формирует сводные аналитические данные для дашборда:
        - Общее число заявлений
        - Проверено / Ожидает проверки
        - Распределение по классам и профилям
        """
        path = Path(file_path)
        default_stats = {
            "total_count": 0,
            "verified_count": 0,
            "pending_count": 0,
            "classes": {},
            "top_profile": "Нет данных",
            "top_profile_pct": 0
        }
        if not path.exists():
            return default_stats

        headers, rows = ExcelManager.read_all_rows(path)
        if not headers or not rows:
            return default_stats

        class_col_idx = None
        status_col_idx = None
        for idx, h in enumerate(headers):
            norm = ExcelManager._normalize_header(h)
            if "класс" in norm or "профиль" in norm:
                class_col_idx = idx
            elif "статус" in norm:
                status_col_idx = idx

        total_count = len(rows)
        verified_count = 0
        pending_count = 0
        classes_dist: Dict[str, int] = {}

        for row in rows:
            # Статус
            if status_col_idx is not None and status_col_idx < len(row):
                st = str(row[status_col_idx]).lower()
                if "проверен" in st:
                    verified_count += 1
                else:
                    pending_count += 1
            else:
                verified_count += 1

            # Класс / профиль
            if class_col_idx is not None and class_col_idx < len(row):
                cl = str(row[class_col_idx]).strip()
                if cl:
                    cl_clean = cl.upper()
                    classes_dist[cl_clean] = classes_dist.get(cl_clean, 0) + 1

        top_profile = "Нет данных"
        top_pct = 0
        if classes_dist and total_count > 0:
            top_item = max(classes_dist.items(), key=lambda x: x[1])
            top_profile = top_item[0]
            top_pct = int((top_item[1] / total_count) * 100)

        return {
            "total_count": total_count,
            "verified_count": verified_count,
            "pending_count": pending_count,
            "classes": classes_dist,
            "top_profile": top_profile,
            "top_profile_pct": top_pct
        }
