import io
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from PIL import Image

from document_loader import DocumentLoader
from extractor import DocumentExtractor
from validator import DataValidator


class HybridVerificationTests(unittest.TestCase):
    def setUp(self):
        self.extractor = DocumentExtractor.__new__(DocumentExtractor)
        self.region = SimpleNamespace(box=(10, 20, 110, 60), includes_label=True)

    def test_matching_numeric_views_verify_primary_value(self):
        data = {
            "Контактный телефон": "+7 (900) 123-45-67",
            "_fields": {},
        }

        self.extractor._merge_verification(
            data,
            "Контактный телефон",
            {"value": "8 900 123 45 67", "status": "read"},
            self.region,
        )

        self.assertEqual(data["Контактный телефон"], "+7 (900) 123-45-67")
        self.assertEqual(data["_fields"]["Контактный телефон"]["status"], "read")
        self.assertTrue(data["_fields"]["Контактный телефон"]["verified"])

    def test_disagreement_keeps_full_page_and_exposes_alternative(self):
        data = {
            "ФИО поступающего ученика": "Бобров Алексей Викторович",
            "_fields": {},
        }

        self.extractor._merge_verification(
            data,
            "ФИО поступающего ученика",
            {"value": "Бодров Алексей Викторович", "status": "read"},
            self.region,
        )

        self.assertEqual(data["ФИО поступающего ученика"], "Бобров Алексей Викторович")
        metadata = data["_fields"]["ФИО поступающего ученика"]
        self.assertEqual(metadata["status"], "needs_review")
        self.assertEqual(metadata["alternative"], "Бодров Алексей Викторович")

    def test_valid_crop_recovers_invalid_primary_value(self):
        data = {"Контактный телефон": "[неразборчиво]", "_fields": {}}

        self.extractor._merge_verification(
            data,
            "Контактный телефон",
            {"value": "8 900 123-45-67", "status": "read"},
            self.region,
        )

        self.assertEqual(data["Контактный телефон"], "8 900 123-45-67")
        self.assertEqual(data["_fields"]["Контактный телефон"]["status"], "read")

    def test_crop_views_include_three_independent_png_variants(self):
        image = Image.new("RGB", (160, 100), "white")
        views = DocumentLoader.get_bbox_crop_views(image, self.region.box, scale=2.0)

        self.assertEqual(len(views), 3)
        for view in views:
            self.assertTrue(view.startswith(b"\x89PNG"))
            with Image.open(io.BytesIO(view)) as parsed:
                self.assertGreater(parsed.width, 0)
                self.assertGreater(parsed.height, 0)

    def test_phone_comparison_treats_russian_prefixes_as_equal(self):
        first = self.extractor._comparison_value("Контактный телефон", "+7 (900) 123-45-67")
        second = self.extractor._comparison_value("Контактный телефон", "8 900 123 45 67")

        self.assertEqual(first, second)

    def test_two_numeric_checks_confirm_the_full_page_value(self):
        data = {"Дата рождения ребенка": "14.07.2014", "_fields": {}}
        checks = [
            {"value": "14.07.2014"},
            {"value": "14/07/2014"},
            {"value": "17.07.2014"},
        ]

        self.extractor._merge_numeric_verifications(
            data, "Дата рождения ребенка", checks, self.region
        )

        self.assertEqual(data["Дата рождения ребенка"], "14.07.2014")
        self.assertEqual(data["_fields"]["Дата рождения ребенка"]["status"], "read")

    def test_valid_primary_is_not_replaced_by_conflicting_numeric_majority(self):
        data = {"Дата рождения ребенка": "14.07.2014", "_fields": {}}
        checks = [
            {"value": "17.07.2014"},
            {"value": "17.07.2014"},
            {"value": "14.07.2014"},
        ]

        self.extractor._merge_numeric_verifications(
            data, "Дата рождения ребенка", checks, self.region
        )

        self.assertEqual(data["Дата рождения ребенка"], "14.07.2014")
        metadata = data["_fields"]["Дата рождения ребенка"]
        self.assertEqual(metadata["status"], "needs_review")
        self.assertEqual(metadata["alternative"], "17.07.2014")

    def test_blind_adjudication_handles_any_digit_pair_and_primary_conflict(self):
        field = "Контактный телефон"
        for alternative in ("9001234568", "900123456"):
            with self.subTest(alternative=alternative):
                self.extractor.is_cancelled = False
                self.extractor._send_to_ai = Mock(side_effect=[
                    {"value": alternative}, {"value": alternative}, {"value": alternative},
                    {"value": "9001234567"},
                ])
                data = {field: "9001234567"}
                self.extractor._verify_numeric_field(data, field, [b"a", b"b", b"c"], self.region)
                meta = data["_fields"][field]
                self.assertTrue(meta["verified"])
                self.assertEqual(meta["verification"]["attempted"], 4)
                self.assertEqual(meta["verification"]["completed"], 4)
                prompt = self.extractor._send_to_ai.call_args.args[0]
                self.assertNotIn(alternative, prompt)
                self.assertNotIn("9001234567", prompt)

    def test_new_adjudication_answer_is_only_an_alternative(self):
        field = "Контактный телефон"
        self.extractor.is_cancelled = False
        self.extractor._send_to_ai = Mock(side_effect=[
            {"value": "9001234568"}, {"value": "9001234569"}, {"value": "9001234560"},
            {"value": "9001234561"},
        ])
        data = {field: "9001234567"}
        self.extractor._verify_numeric_field(data, field, [b"a"] * 3, self.region)
        self.assertEqual(data[field], "9001234567")
        self.assertFalse(data["_fields"][field]["verified"])
        self.assertIn("9001234561", data["_fields"][field]["alternative"])

    def test_skipped_verification_has_explicit_reason(self):
        field = "Контактный телефон"
        self.extractor._extract_custom_columns = Mock(return_value={field: "9001234567"})
        self.extractor.field_locator = SimpleNamespace(available=False)
        data = self.extractor.extract_from_image(b"page", [field], pil_image=object())
        audit = data["_fields"][field]["verification"]
        self.assertEqual(audit["reason"], "paddle_unavailable")
        self.assertEqual(audit["attempted"], 0)
        self.assertEqual(data["_stats"]["requests"], 1)

    def test_extraction_preserves_audit_across_merges(self):
        field = "Контактный телефон"
        self.extractor._extract_custom_columns = Mock(return_value={field: "9001234567"})
        self.extractor.field_locator = SimpleNamespace(
            available=True, error="", locate=Mock(return_value={field: self.region})
        )
        self.extractor._send_to_ai = Mock(return_value={"value": "9001234567"})
        with patch.object(DocumentLoader, "get_bbox_crop_views", return_value=[b"a"] * 3):
            data = self.extractor.extract_from_image(b"page", [field], pil_image=object())
        audit = data["_fields"][field]["verification"]
        self.assertEqual(audit, {"status": "completed", "reason": "", "attempted": 3, "completed": 3})
        self.assertEqual(data["_stats"]["requests"], 4)

    def test_family_surname_is_suggested_without_replacement(self):
        data = {"ФИО поступающего ученика": "Бобров Иван", "ФИО родителя / заявителя": "Бодров Игорь"}
        self.extractor._reconcile_family_surname(data)
        self.assertEqual(data["ФИО родителя / заявителя"], "Бодров Игорь")
        self.assertEqual(data["_fields"]["ФИО родителя / заявителя"]["alternative"], "Бобров Игорь")

    def test_failed_crop_reading_is_counted_and_does_not_confirm_value(self):
        field = "Контактный телефон"
        self.extractor._extract_custom_columns = Mock(return_value={field: "9001234567"})
        self.extractor.field_locator = SimpleNamespace(
            available=True, error="", locate=Mock(return_value={field: self.region})
        )
        self.extractor._send_to_ai = Mock(return_value={"error": "timeout"})
        with patch.object(DocumentLoader, "get_bbox_crop_views", return_value=[b"a"] * 3):
            data = self.extractor.extract_from_image(b"page", [field], pil_image=object())
        audit = data["_fields"][field]["verification"]
        self.assertEqual(audit["status"], "partial")
        self.assertEqual(audit["attempted"], 3)
        self.assertEqual(audit["completed"], 0)
        self.assertFalse(data["_fields"][field]["verified"])

    def test_missing_region_and_disabled_mode_have_distinct_reasons(self):
        field = "Контактный телефон"
        for enabled, expected in ((True, "region_not_found"), (False, "mode_disabled")):
            self.extractor._extract_custom_columns = Mock(return_value={field: "9001234567"})
            self.extractor.field_locator = SimpleNamespace(available=True, error="", locate=Mock(return_value={}))
            data = self.extractor.extract_from_image(b"page", [field], pil_image=object(), use_field_crops=enabled)
            self.assertEqual(data["_fields"][field]["verification"]["reason"], expected)

    def test_dictionary_suggestion_is_exposed_without_changing_name(self):
        field = "ФИО поступающего ученика"
        self.extractor._extract_custom_columns = Mock(return_value={
            field: "Бодров Иван", "_reference_suggestions": {field: "Бобров Иван"},
        })
        data = self.extractor.extract_from_image(b"page", [field], use_field_crops=False)
        self.assertEqual(data[field], "Бодров Иван")
        self.assertEqual(data["_fields"][field]["alternative"], "Бобров Иван")

    def test_prompts_do_not_contain_sample_specific_answers(self):
        for field in ("Контактный телефон", "Паспортные данные", "Дата подачи заявления"):
            prompt = self.extractor._field_prompt(field)
            for sample in ("907", "РЖД", "ГУСД", "4 и 7"):
                self.assertNotIn(sample, prompt)
        self.extractor._send_to_ai = Mock(return_value={})
        self.extractor._extract_custom_columns(b"page", ["Дата подачи заявления"])
        self.assertNotIn("4 и 7", self.extractor._send_to_ai.call_args.args[0])

    def merge_numeric(self, field, primary, values):
        data = {field: primary}
        self.extractor._merge_numeric_verifications(
            data, field, [{"value": value} for value in values], self.region
        )
        return data, data["_fields"][field]

    def test_digit_majority_recovers_phone_without_matching_whole_readings(self):
        field = "Контактный телефон"
        data, meta = self.merge_numeric(field, "[неразборчиво]", [
            "8 901 123 45 67", "+7 (900) 128-45-67", "9001234568",
        ])
        self.assertEqual(data[field], "+7 (900) 123-45-67")
        self.assertTrue(meta["verified"])
        self.assertEqual(meta["unresolved_positions"], [])

    def test_digit_majority_keeps_valid_primary_and_offers_consensus(self):
        field = "Контактный телефон"
        primary = "+7 (900) 123-45-69"
        data, meta = self.merge_numeric(field, primary, [
            "9011234567", "9001284567", "9001234568",
        ])
        self.assertEqual(data[field], primary)
        self.assertFalse(meta["verified"])
        self.assertIn("+7 (900) 123-45-67", meta["alternative"])

    def test_digit_tie_records_position_without_inventing_value(self):
        field = "Контактный телефон"
        data, meta = self.merge_numeric(field, "", [
            "9001234567", "9001234568", "9001234569",
        ])
        self.assertEqual(data[field], "")
        self.assertEqual(meta["unresolved_positions"], [10])
        self.assertNotIn("digit_consensus", meta)
        self.assertEqual(meta["status"], "needs_review")

    def test_unequal_lengths_and_mixed_passport_text_are_not_spliced(self):
        for field, values in [
            ("Контактный телефон", ["9001234567", "900123456", "9011234567"]),
            ("Паспортные данные", ["45 12 123456 МВД", "45 12 123457 МВД", "45 12 123458 МВД"]),
        ]:
            with self.subTest(field=field):
                _, meta = self.merge_numeric(field, "", values)
                self.assertNotIn("digit_consensus", meta)
                self.assertEqual(meta["status"], "needs_review")

    def test_date_digit_majority_recovers_invalid_primary(self):
        field = "Дата рождения ребенка"
        data, meta = self.merge_numeric(field, "45.13.2026", [
            "17.07.2014", "14.04.2014", "14.07.2017",
        ])
        self.assertEqual(data[field], "14.07.2014")
        self.assertTrue(meta["verified"])

    def test_invalid_consensus_date_is_not_accepted(self):
        field = "Дата рождения ребенка"
        data, meta = self.merge_numeric(field, "", [
            "21.02.2026", "31.03.2026", "31.02.2027",
        ])
        self.assertEqual(data[field], "")
        self.assertEqual(meta["digit_consensus"], "31.02.2026")
        self.assertFalse(meta["verified"])

    def test_matching_invalid_dates_are_never_verified(self):
        field = "Дата рождения ребенка"
        for date in ("31.02.2026", "45.13.2026", "29.02.2025"):
            with self.subTest(date=date):
                data, meta = self.merge_numeric(field, date, [date] * 3)
                self.assertEqual(data[field], date)
                self.assertFalse(meta["verified"])
                self.extractor._merge_verification(data, field, {"value": date}, self.region)
                self.assertFalse(data["_fields"][field]["verified"])

    def test_calendar_valid_crop_recovers_impossible_date(self):
        field = "Дата рождения ребенка"
        data = {field: "31.02.2026"}
        self.extractor._merge_verification(
            data, field, {"value": "28.02.2026"}, self.region
        )
        self.assertEqual(data[field], "28.02.2026")
        self.assertTrue(data["_fields"][field]["verified"])

    def test_equivalent_date_formats_vote_together(self):
        field = "Дата рождения ребенка"
        _, meta = self.merge_numeric(field, "29.02.2024", [
            "2024-02-29", "29/2/24", "29.02.2024",
        ])
        self.assertTrue(meta["verified"])

    def test_unclear_partial_readings_do_not_cast_votes(self):
        field = "Контактный телефон"
        data = {field: ""}
        self.extractor._merge_numeric_verifications(data, field, [
            {"value": "9001234567", "status": "unclear"},
            {"value": "9001234567 [неразборчиво]"},
            {"value": "9001234567"},
        ], self.region)
        self.assertEqual(data[field], "")
        self.assertFalse(data["_fields"][field]["verified"])

    def test_synthesized_snils_requires_valid_checksum(self):
        field = "СНИЛС поступающего"
        data, meta = self.merge_numeric(field, "", [
            "11223344695", "11223344585", "11223344596",
        ])
        self.assertEqual(data[field], "112-233-445 95")
        self.assertTrue(meta["verified"])
        data, meta = self.merge_numeric(field, "", [
            "11223344694", "11223344584", "11223344593",
        ])
        self.assertEqual(data[field], "")
        self.assertFalse(meta["verified"])


class CalendarValidationTests(unittest.TestCase):
    def test_calendar_boundaries_and_malformed_dates(self):
        for value in ("31.02.2026", "29.02.1900", "00.01.2026", "01.13.2026",
                      "01.01.0000", "01.01.026", "", "[неразборчиво]"):
            with self.subTest(value=value):
                self.assertIsNone(DataValidator.parse_date(value))
                self.assertEqual(DataValidator.format_date(value), value)

    def test_supported_formats_and_leap_years(self):
        for value, expected in [
            ("2000-02-29", "29.02.2000"), ("29/2/24", "29.02.2024"),
            ("1-2-95", "01.02.1995"), ("1.2.26", "01.02.2026"),
            ("01.01.0001", "01.01.0001"),
        ]:
            with self.subTest(value=value):
                self.assertIsNotNone(DataValidator.parse_date(value))
                self.assertEqual(DataValidator.format_date(value), expected)


if __name__ == "__main__":
    unittest.main()
