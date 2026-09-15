import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evaluate import _digit_alignment, _evaluate_case, _format_md, _norm_field, _configured_pipeline
from reference_dictionary import ReferenceDictionary
from validator import DataValidator
from compiler.pipeline import CompilerPipeline


class QualityTests(unittest.TestCase):
    def test_alignment_does_not_turn_deletion_into_many_substitutions(self):
        for reference, predicted, expected in [("123456", "12456", ("3", "∅")),
                                                ("12456", "123456", ("∅", "3"))]:
            changes = [p for p in _digit_alignment(reference, predicted) if p["reference"] != p["predicted"]]
            self.assertEqual(len(changes), 1)
            self.assertEqual((changes[0]["reference"], changes[0]["predicted"]), expected)
            self.assertEqual(changes[0]["position"], 3)

    def test_phone_preserves_extensions_multiple_numbers_and_excess_digits(self):
        for value in ("79001234567123", "+7 900 123-45-67 доб. 123", "79001234567; 79007654321",
                      "9001234567 [неразборчиво]", "12345/67890"):
            self.assertEqual(DataValidator.format_phone(value), value)
        self.assertEqual(DataValidator.format_phone("8 900 123 45 67"), "+7 (900) 123-45-67")

    def test_dictionary_fio_changes_are_suggestions(self):
        dictionary = ReferenceDictionary.__new__(ReferenceDictionary)
        dictionary.values = {"surnames": {"Бобров"}, "first_names": set(), "patronymics": set()}
        dictionary.corrections = {"surnames": {"бодров": "Бобров"}}
        field = "ФИО поступающего ученика"
        data = dictionary.correct_data({field: "Бодров Иван"})
        self.assertEqual(data[field], "Бодров Иван")
        self.assertEqual(data["_reference_suggestions"][field], "Бобров Иван")

    def test_metrics_count_false_confirmation_and_requests(self):
        field = "Контактный телефон"
        pipeline = SimpleNamespace(extract=Mock(return_value={
            field: "9001234568", "_fields": {field: {"verified": True, "status": "read"}},
            "_stats": {"requests": 5},
        }))
        with patch("evaluate._load_labels", return_value={field: "9001234567"}), patch.object(Path, "read_bytes", return_value=b"scan"):
            result = _evaluate_case(pipeline, Path("case.png"), "vlm")
        self.assertTrue(result["fields"][field]["false_verified"])
        self.assertEqual(result["requests"], 5)
        self.assertEqual(pipeline.extract.call_args.kwargs["ocr_priority"], "vlm")
        report = _format_md([result], {"model": "test"})
        self.assertIn("1/1 (100.0%)", report)
        self.assertIn("7 → 8", report)

    def test_pipeline_error_and_missing_labels_are_failed_cases(self):
        pipeline = SimpleNamespace(extract=Mock(return_value={"error": "failed"}))
        with patch("evaluate._load_labels", return_value={"Дата": ""}), patch.object(Path, "read_bytes", return_value=b"scan"):
            self.assertTrue(_evaluate_case(pipeline, Path("case.png"))["error"])
        with patch("evaluate._load_labels", side_effect=FileNotFoundError("missing")):
            self.assertTrue(_evaluate_case(pipeline, Path("case.png"))["error"])

    def test_normalization_keeps_passport_text_and_canonicalizes_phone_date(self):
        self.assertNotEqual(_norm_field("Паспорт", "4512 123456 МВД"), _norm_field("Паспорт", "4512 123456 РЖД"))
        self.assertEqual(_norm_field("Телефон", "89001234567"), _norm_field("Телефон", "+7 900 123 45 67"))
        self.assertEqual(_norm_field("Дата", "2024-02-29"), "29.02.2024")

    def test_evaluation_loads_saved_model_parameters_and_key(self):
        with patch("evaluate.load_settings", return_value={"host": "http://localhost", "model": "chosen",
                   "api_key": "test-key", "params": {"temperature": 0}}), patch("evaluate.CompilerPipeline") as factory:
            pipeline = _configured_pipeline()
        factory.assert_called_once_with(host="http://localhost", model_name="chosen")
        pipeline.extractor.set_api_key.assert_called_once_with("test-key")
        pipeline.extractor.apply_params.assert_called_once_with({"temperature": 0})
        self.assertEqual(pipeline.extractor.model_name, "chosen")

    def test_multipage_merge_keeps_skip_reason_and_request_total(self):
        pipeline = CompilerPipeline.__new__(CompilerPipeline)
        pipeline.extractor = SimpleNamespace(_postprocess_data=lambda data: data)
        field = "Телефон"
        page = {field: "9001234567", "_stats": {"requests": 1}, "_fields": {field: {
            "status": "needs_review", "verification": {"status": "skipped", "reason": "multipage_disabled", "attempted": 0, "completed": 0},
        }}}
        data = pipeline._merge_page_results([page, page])
        self.assertEqual(data["_stats"]["requests"], 2)
        self.assertEqual(data["_fields"][field]["verification"]["reason"], "multipage_disabled")


if __name__ == "__main__":
    unittest.main()
