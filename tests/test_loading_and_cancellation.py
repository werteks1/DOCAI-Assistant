import base64
import io
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pymupdf
import requests
from PIL import Image

import config
from cancellation import CancellationToken, ExtractionCancelled
from compiler.jobs import CANCELLED, DONE, EXTRACTION_LOCK, Job, JobBusy, JobStore, make_preview
from compiler.models import BatchExtractRequest
from compiler.pipeline import CompilerPipeline
from document_loader import DocumentLoader
from extractor import DocumentExtractor


class LoadingTests(unittest.TestCase):
    def test_exif_orientation_applies_to_pipeline_and_preview(self):
        image = Image.new("RGB", (80, 40), "white")
        exif = Image.Exif()
        exif[274] = 6
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", exif=exif)
        payload = base64.b64encode(buffer.getvalue()).decode()
        pipeline = CompilerPipeline.__new__(CompilerPipeline)
        pipeline.extractor = SimpleNamespace(field_locator=SimpleNamespace(detector_name=config.PADDLE_DETECTOR),
                                             extract_from_image=Mock(return_value={"field": "value"}))
        with patch.object(DocumentLoader, "preprocess_for_handwriting", return_value=b"page"):
            pipeline.extract(payload, filename="phone.jpg")
        self.assertEqual(pipeline.extractor.extract_from_image.call_args.kwargs["pil_image"].size, (40, 80))
        with Image.open(io.BytesIO(make_preview(payload))) as preview:
            self.assertEqual(preview.size, (40, 80))

    def test_oversized_image_is_rejected_before_pixel_decode(self):
        buffer = io.BytesIO()
        Image.new("RGB", (80, 40)).save(buffer, format="PNG")
        with patch.object(config, "MAX_IMAGE_PIXELS", 100), \
                patch("PIL.PngImagePlugin.PngImageFile.load", side_effect=AssertionError("decoded")):
            with self.assertRaises(ValueError):
                DocumentLoader.open_image(io.BytesIO(buffer.getvalue()))
            self.assertIsNone(make_preview(base64.b64encode(buffer.getvalue()).decode()))

    def test_pdf_size_checked_before_rasterization_including_preview(self):
        with pymupdf.open() as doc:
            doc.new_page(width=2000, height=2000)
            payload = doc.tobytes()
        pipeline = CompilerPipeline.__new__(CompilerPipeline)
        with patch.object(config, "MAX_IMAGE_PIXELS", 100), \
                patch.object(pymupdf.Page, "get_pixmap", side_effect=AssertionError("rasterized")):
            with self.assertRaises(ValueError):
                pipeline._rasterize_pdf_pages(payload)
            self.assertIsNone(make_preview(base64.b64encode(payload).decode()))

    def test_small_pdf_rasterizes_as_rgb(self):
        with pymupdf.open() as doc:
            page = doc.new_page(width=20, height=30)
            image = DocumentLoader.rasterize_page(page, pymupdf.Matrix(3, 3))
            self.assertEqual(image.size, (60, 90))
            self.assertEqual(image.mode, "RGB")

    def test_pdf_preview_selects_page(self):
        with pymupdf.open() as document:
            document.new_page(width=40, height=40)
            second_page = document.new_page(width=40, height=40)
            second_page.draw_rect(pymupdf.Rect(2, 2, 38, 38), color=(0, 0, 0), fill=(0, 0, 0))
            payload = base64.b64encode(document.tobytes()).decode()
        first, second = make_preview(payload, page=0), make_preview(payload, page=1)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)
        self.assertIsNone(make_preview(payload, page=5))


class QueueTests(unittest.TestCase):
    def test_cleanup_keeps_newest_even_when_uuid_sorts_first(self):
        store = JobStore()
        for index in range(20):
            job = Job(job_id=f"z{index}", status=DONE, created_at=index)
            store._jobs[job.job_id] = job
        newest = Job(job_id="000", status=DONE, created_at=21)
        store._jobs[newest.job_id] = newest
        store._update(newest)
        self.assertIs(store.get("000"), newest)
        self.assertIsNone(store.get("z0"))
        self.assertEqual(len(store._jobs), 20)

    def test_busy_start_does_not_create_failed_job(self):
        store = JobStore()
        request = BatchExtractRequest(documents=[{"image_base64": "a" * 16}])
        with EXTRACTION_LOCK:
            with self.assertRaises(JobBusy):
                store.start(request, Mock(), owner_id=1)
        self.assertEqual(store._jobs, {})

    def test_cancelled_job_stops_current_file_and_keeps_lock_until_exit(self):
        store = JobStore()
        request = BatchExtractRequest(documents=[{"image_base64": "a" * 16}] * 2)
        entered, stopping, release = threading.Event(), threading.Event(), threading.Event()
        def extract(*args, cancellation, **kwargs):
            entered.set()
            try:
                while True:
                    cancellation.check()
                    time.sleep(0.01)
            except ExtractionCancelled:
                stopping.set()
                release.wait(2)
                raise
        pipeline = SimpleNamespace(extract=Mock(side_effect=extract))
        with patch("compiler.jobs.make_preview", return_value=None), patch("compiler.auth_store.log_activity"):
            job_id = store.start(request, pipeline, owner_id=1)
            try:
                self.assertTrue(entered.wait(2))
                self.assertTrue(store.cancel(job_id))
                self.assertTrue(stopping.wait(2))
                with self.assertRaises(JobBusy):
                    store.start(request, pipeline, owner_id=1)
            finally:
                release.set()
            deadline = time.monotonic() + 2
            while EXTRACTION_LOCK.locked() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(EXTRACTION_LOCK.locked())
            job = store.get(job_id)
            self.assertEqual(job.status, CANCELLED)
            self.assertIsNone(job.items[0].data)
            self.assertEqual(job.items[1].status, "pending")
            self.assertEqual(pipeline.extract.call_count, 1)


class NetworkCancellationTests(unittest.TestCase):
    def test_real_network_wait_is_cancelled_for_both_backends(self):
        # Локальный HTTP-сервер имитирует зависание до заголовков или внутри потока.
        for backend in ("lmstudio", "ollama"):
            for phase in ("headers", "stream", "success"):
                with self.subTest(backend=backend, phase=phase):
                    self.run_network_case(backend, phase)

    def run_network_case(self, backend, phase):
        entered, release = threading.Event(), threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                if phase == "success":
                    self.send_response(200)
                    self.end_headers()
                    value = json.dumps({"value": "ok"})
                    body = "data: " + json.dumps({"choices": [{"delta": {"content": value}}]}) + "\n\ndata: [DONE]\n\n" if backend == "lmstudio" else json.dumps({"message": {"role": "assistant", "content": value}, "done": True}) + "\n"
                    self.wfile.write(body.encode())
                    return
                if phase == "headers":
                    entered.set()
                    release.wait(3)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream" if backend == "lmstudio" else "application/x-ndjson")
                self.end_headers()
                chunk = 'data: {"choices":[{"delta":{"content":"{"}}]}\n\n' if backend == "lmstudio" else json.dumps({"message": {"role": "assistant", "content": "{"}, "done": False}) + "\n"
                try:
                    self.wfile.write(chunk.encode())
                    self.wfile.flush()
                    entered.set()
                    release.wait(3)
                except OSError:
                    pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        extractor = DocumentExtractor.__new__(DocumentExtractor)
        extractor.host = f"http://127.0.0.1:{server.server_port}"
        extractor.backend = backend
        extractor.model_name = "test"
        extractor.temperature = 0
        extractor.top_p = None
        extractor.max_tokens = 20
        extractor.timeout_seconds = 30
        extractor.is_cancelled = False
        extractor.cancellation = CancellationToken()
        extractor._session = requests.Session()
        extractor._postprocess_data = lambda data: data
        errors = []
        results = []
        def run():
            try:
                method = extractor._call_ollama if backend == "ollama" else extractor._call_openai_compatible
                results.append(method("system", "prompt", b"image"))
            except Exception as exc:
                errors.append(exc)
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            if phase == "success":
                worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(results[0]["value"], "ok")
                return
            self.assertTrue(entered.wait(2))
            extractor.cancellation.cancel()
            worker.join(1)
            self.assertFalse(worker.is_alive(), "network operation did not stop")
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], ExtractionCancelled)
        finally:
            release.set()
            worker.join(3)
            extractor._session.close()
            server.shutdown()
            server.server_close()
            serving.join()


if __name__ == "__main__":
    unittest.main()
