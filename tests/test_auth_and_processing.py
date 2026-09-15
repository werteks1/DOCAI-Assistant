import base64
import importlib.util
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import openpyxl
import pymupdf
from fastapi.testclient import TestClient
from PIL import Image

import config
from compiler import auth_store
from compiler.jobs import DONE, ERROR, RUNNING, EXTRACTION_LOCK, Job, JobItem, JobStore
from compiler.models import BatchExtractRequest
from compiler.pipeline import CompilerPipeline, RecognitionError
from document_loader import DocumentLoader


def test_pipeline():
    pipeline = CompilerPipeline.__new__(CompilerPipeline)
    pipeline.extractor = SimpleNamespace(
        field_locator=SimpleNamespace(detector_name=config.PADDLE_DETECTOR, available=False),
        extract_from_image=Mock(), model_name="test", backend="test", host="http://localhost",
    )
    pipeline.available_models = []
    return pipeline


def image_payload():
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class LoginProtectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        for target, value in (("DOCIA_DB", Path(directory.name) / "auth.db"),):
            patcher = patch.object(config, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        failures = patch.object(auth_store, "_failures", {})
        failures.start()
        self.addCleanup(failures.stop)
        auth_store.init_db()

    def test_case_variants_share_lockout_including_cyrillic(self):
        auth_store.create_user("Оператор", "secret", "operator")
        for name in ("admin", "Оператор"):
            with self.subTest(username=name):
                variants = (name.lower(), name.upper(), name.title(), f" {name} ", name.swapcase())
                for variant in variants:
                    self.assertIsNone(auth_store.verify_login(variant, "wrong").user)
                for variant in variants:
                    self.assertTrue(auth_store.verify_login(variant, "admin" if name == "admin" else "secret").locked)

    def test_success_and_password_reset_clear_casefolded_failures(self):
        user = auth_store.create_user("Оператор", "secret", "operator")
        for _ in range(4):
            auth_store.verify_login("ОПЕРАТОР", "wrong")
        self.assertIsNotNone(auth_store.verify_login("оператор", "secret").user)
        for _ in range(5):
            auth_store.verify_login("оператор", "wrong")
        self.assertTrue(auth_store.verify_login("Оператор", "secret").locked)
        auth_store.set_password(user["id"], "changed")
        self.assertIsNotNone(auth_store.verify_login("ОПЕРАТОР", "changed").user)


class ProcessingAndAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Изолированный модуль API: без чтения рабочих настроек, БД и сети.
        spec = importlib.util.spec_from_file_location(
            "compiler._regression_api", Path(__file__).resolve().parents[1] / "compiler" / "api.py"
        )
        cls.api = importlib.util.module_from_spec(spec)
        with patch("compiler.settings_store.load_settings", return_value={}), \
                patch("compiler.pipeline.CompilerPipeline", return_value=test_pipeline()), \
                patch("compiler.auth_store.init_db"):
            spec.loader.exec_module(cls.api)

    def setUp(self):
        self.pipeline = test_pipeline()
        self.api.pipeline = self.pipeline
        self.api.job_store = JobStore()
        self.user = {"id": 1, "username": "owner", "role": "operator", "must_change": False}
        self.api.app.dependency_overrides[self.api.enforce_change] = lambda: self.user
        self.addCleanup(self.api.app.dependency_overrides.clear)
        self.client = TestClient(self.api.app)
        self.addCleanup(self.client.close)
        for patcher in (
            patch.object(DocumentLoader, "preprocess_for_handwriting", return_value=b"page"),
            patch.object(auth_store, "log_activity"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.payload = image_payload()

    def test_job_owner_can_read_preview_and_cancel(self):
        job = Job(job_id="owned", owner_id=1, total=1, status=DONE,
                  items=[JobItem(filename="test.png", status=DONE, data={"field": "value"})],
                  previews={0: b"preview"})
        self.api.job_store._jobs[job.job_id] = job
        self.assertEqual(self.client.get("/api/batch/owned").json()["items"][0]["data"], {"field": "value"})
        self.assertEqual(self.client.get("/api/batch/owned/0/preview").content, b"preview")
        job.status = RUNNING
        self.assertEqual(self.client.post("/api/batch/owned/cancel").json()["status"], "cancelling")
        self.assertTrue(job.cancelled)

    def test_other_accounts_cannot_read_or_cancel_even_with_same_username(self):
        job = Job(job_id="private", owner_id=2, username="owner", total=1, status=DONE,
                  items=[JobItem(filename="private.png", data={"field": "private"})], previews={0: b"private"})
        self.api.job_store._jobs[job.job_id] = job
        for role in ("operator", "admin"):
            self.user["role"] = role
            for method, path in (("GET", "/api/batch/private"), ("GET", "/api/batch/private/0/preview"),
                                 ("POST", "/api/batch/private/cancel")):
                with self.subTest(role=role, path=path):
                    response = self.client.request(method, path)
                    self.assertEqual(response.status_code, 404)
                    self.assertEqual(response.json(), {"detail": "Джоб не найден"})
        job.status = RUNNING
        self.assertEqual(self.client.post("/api/batch/private/cancel").status_code, 404)
        self.assertFalse(job.cancelled)

    def test_start_assigns_authenticated_owner_not_client_value(self):
        with patch("compiler.jobs.make_preview", return_value=None), patch("compiler.jobs.threading.Thread"):
            response = self.client.post("/api/batch", json={
                "documents": [{"image_base64": self.payload}], "owner_id": 999,
            })
        self.assertEqual(response.status_code, 202)
        self.addCleanup(EXTRACTION_LOCK.release)
        job = self.api.job_store.get(response.json()["job_id"])
        self.assertEqual(job.owner_id, self.user["id"])

    def test_operator_cannot_select_model_in_single_or_batch_requests(self):
        for model in ("other", "test", ""):
            for endpoint, body in (("/api/extract", {"image_base64": self.payload}),
                                   ("/api/batch", {"documents": [{"image_base64": self.payload}]})):
                with self.subTest(model=model, endpoint=endpoint):
                    response = self.client.post(endpoint, json={**body, "model": model})
                    self.assertEqual(response.status_code, 403)
        self.assertEqual(self.pipeline.extractor.model_name, "test")
        self.pipeline.extractor.extract_from_image.assert_not_called()

    def test_admin_can_select_model_but_cannot_change_it_during_extraction(self):
        self.user["role"] = "admin"
        self.pipeline.available_models = ["test", "other"]
        self.pipeline.extractor.extract_from_image.return_value = {"field": "value"}
        response = self.client.post("/api/extract", json={"image_base64": self.payload, "model": "other"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.pipeline.extractor.model_name, "other")
        with EXTRACTION_LOCK:
            self.assertEqual(self.client.post("/api/model", json={"model": "test"}).status_code, 409)
        self.assertEqual(self.pipeline.extractor.model_name, "other")

    def test_cancellation_is_owned_and_releases_lock_before_retry(self):
        entered = threading.Event()
        def wait_for_cancel(*args, **kwargs):
            entered.set()
            token = self.pipeline.extractor.cancellation
            while True:
                token.check()
                time.sleep(0.01)
        self.pipeline.extractor.extract_from_image.side_effect = wait_for_cancel
        responses = []
        worker = threading.Thread(target=lambda: responses.append(self.client.post("/api/extract", json={
            "image_base64": self.payload, "request_id": "cancel-test",
        })))
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            self.user = {**self.user, "id": 2}
            self.assertEqual(self.client.post("/api/extract/cancel-test/cancel").status_code, 404)
            self.user = {**self.user, "id": 1}
            self.assertEqual(self.client.post("/api/extract/cancel-test/cancel").status_code, 200)
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(responses[0].status_code, 409)
            self.assertIn("отменено", responses[0].json()["detail"])
            self.assertEqual(self.client.post("/api/extract/cancel-test/cancel").status_code, 404)
            self.pipeline.extractor.extract_from_image.side_effect = None
            self.pipeline.extractor.extract_from_image.return_value = {"field": "retry"}
            self.assertEqual(self.client.post("/api/extract", json={"image_base64": self.payload}).status_code, 200)
        finally:
            self.api.operation_store.cancel("cancel-test", 1)
            worker.join(2)

    def test_recognition_failure_returns_http_error_and_failed_activity(self):
        for invalid in ({"error": "Не удалось распарсить ответ модели"}, None, [], {}):
            with self.subTest(result=invalid):
                self.pipeline.extractor.extract_from_image.return_value = invalid
                response = self.client.post("/api/extract", json={"image_base64": self.payload})
                self.assertEqual(response.status_code, 422)
                self.assertIn("detail", response.json())
                self.assertFalse(auth_store.log_activity.call_args.kwargs["ok"])

    def test_successful_recognition_remains_exportable(self):
        data = {"Контактный телефон": "9001234567"}
        self.pipeline.extractor.extract_from_image.return_value = data
        response = self.client.post("/api/extract", json={"image_base64": self.payload})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"], data)
        self.assertTrue(auth_store.log_activity.call_args.kwargs["ok"])

    def test_pdf_page_error_cannot_be_merged_into_partial_success(self):
        good = {"field": "value", "_fields": {"field": {"status": "read"}}}
        for pages, results, page_number in (
            ([(None, b"page")], [{"error": "invalid JSON"}], None),
            ([(None, b"page")] * 2, [good, {"error": "invalid JSON"}], 2),
            ([(None, b"page")] * 2, [good, RuntimeError("timeout")], 2),
        ):
            with self.subTest(pages=len(pages), results=results):
                self.pipeline.extractor.extract_from_image.side_effect = results
                with patch.object(self.pipeline, "_rasterize_pdf_pages", return_value=pages):
                    with self.assertRaises(RecognitionError) as error:
                        self.pipeline.extract(self.payload, filename="test.pdf")
                if page_number:
                    self.assertIn(f"страницу {page_number}", str(error.exception))

    def test_preview_selects_pdf_page_and_rejects_missing(self):
        with pymupdf.open() as document:
            document.new_page(width=40, height=40)
            second_page = document.new_page(width=40, height=40)
            second_page.draw_rect(pymupdf.Rect(4, 4, 36, 36), color=(0, 0, 0), fill=(0, 0, 0))
            payload = base64.b64encode(document.tobytes()).decode()
        first = self.client.post("/api/preview", json={"image_base64": payload, "page": 0})
        second = self.client.post("/api/preview", json={"image_base64": payload, "page": 1})
        self.assertEqual((first.status_code, second.status_code), (200, 200))
        self.assertNotEqual(first.content, second.content)
        self.assertEqual(self.client.post("/api/preview", json={"image_base64": payload, "page": 5}).status_code, 422)
        self.assertEqual(self.client.post("/api/preview", json={"image_base64": payload, "page": 99}).status_code, 422)

    def test_export_status_colors_follow_value(self):
        records = [
            {"Имя файла источника": "checked.png", "Статус проверки": "Проверено"},
            {"Имя файла источника": "duplicate.png", "Статус проверки": "Проверено · возможный дубль"},
            {"Имя файла источника": "pending.png", "Статус проверки": "Требует проверки"},
        ]
        response = self.client.post("/api/export/excel", json={"records": records})
        self.assertEqual(response.status_code, 200)
        sheet = openpyxl.load_workbook(io.BytesIO(response.content)).active
        column = [cell.value for cell in sheet[1]].index("Статус проверки") + 1
        fills = [sheet.cell(row=row, column=column).fill.start_color.rgb for row in (2, 3, 4)]
        self.assertEqual(fills, ["00E8F5E9", "00FFF4D6", "00FFFFFF"])

    def test_batch_keeps_errors_out_of_results_and_continues_other_files(self):
        for all_failed in (False, True):
            with self.subTest(all_failed=all_failed):
                request = BatchExtractRequest(documents=[{"image_base64": self.payload}] * 2)
                job = Job(job_id="batch", total=2, owner_id=1,
                          items=[JobItem(filename="first.png"), JobItem(filename="second.png")])
                self.api.job_store._jobs = {job.job_id: job}
                error = {"error": "invalid JSON"}
                self.pipeline.extractor.extract_from_image.side_effect = [
                    error, error if all_failed else {"Контактный телефон": "9001234567"},
                ]
                self.api.job_store._run(job.job_id, request, self.pipeline)
                self.assertEqual(job.completed, 2)
                self.assertEqual(job.items[0].status, ERROR)
                self.assertIsNone(job.items[0].data)
                self.assertEqual(job.items[0].error, "invalid JSON")
                self.assertEqual(job.items[1].status, ERROR if all_failed else DONE)
                self.assertEqual(job.status, ERROR if all_failed else DONE)
                self.assertFalse(auth_store.log_activity.call_args.kwargs["ok"])


if __name__ == "__main__":
    unittest.main()
