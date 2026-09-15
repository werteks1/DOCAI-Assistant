import unittest
from unittest.mock import MagicMock, Mock, patch

from extractor import DocumentExtractor


class SiliconFlowTests(unittest.TestCase):
    def setUp(self):
        self.extractor = DocumentExtractor.__new__(DocumentExtractor)
        self.extractor.host = "https://api.siliconflow.com"
        self.extractor.backend = "siliconflow"
        self.extractor.model_name = "Qwen/Qwen3-VL-32B-Instruct"
        self.extractor.temperature = 0.0
        self.extractor.top_p = 1.0
        self.extractor.max_tokens = 512
        self.extractor.timeout_seconds = 30
        self.extractor.is_cancelled = False
        self.extractor.api_key = ""
        self.extractor._session = MagicMock()
        self.extractor._postprocess_data = lambda data: data

    def test_any_ai_host_is_allowed(self):
        self.assertEqual(
            DocumentExtractor._clean_host("https://api.siliconflow.com/v1"),
            "https://api.siliconflow.com",
        )
        self.assertEqual(
            DocumentExtractor._clean_host("http://api.siliconflow.com"),
            "http://api.siliconflow.com",
        )
        self.assertEqual(
            DocumentExtractor._clean_host("https://example.com"),
            "https://example.com",
        )
        self.assertEqual(
            DocumentExtractor._clean_host("http://seekai.cc"),
            "http://seekai.cc",
        )
        self.assertEqual(
            DocumentExtractor._clean_host("https://seekai.cc/v1"),
            "https://seekai.cc",
        )
        self.assertEqual(
            DocumentExtractor._clean_host("ai.example.com:8080"),
            "http://ai.example.com:8080",
        )
        with self.assertRaises(ValueError):
            DocumentExtractor._clean_host("https://")

    def test_endpoint_paths_are_stripped(self):
        self.assertEqual(
            DocumentExtractor._clean_host("https://seekai.cc/v1/chat/completions"),
            "https://seekai.cc",
        )
        self.assertEqual(
            DocumentExtractor._clean_host("https://seekai.cc/v1/models"),
            "https://seekai.cc",
        )
        self.assertEqual(
            DocumentExtractor._clean_host("http://192.168.0.19:1234/v1/chat/completions"),
            "http://192.168.0.19:1234",
        )

    def test_siliconflow_request_supports_two_images(self):
        response = Mock(status_code=200)
        response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"{\\"value\\":\\"test\\"}"}}]}',
            b'data: [DONE]',
        ]
        self.extractor._session.post.return_value = response

        result = self.extractor._call_openai_compatible(
            "system", "prompt", [b"\x89PNG-a", b"\x89PNG-b"]
        )

        payload = self.extractor._session.post.call_args.kwargs["json"]
        content = payload["messages"][1]["content"]
        self.assertEqual(sum(part["type"] == "image_url" for part in content), 2)
        image_parts = [part for part in content if part["type"] == "image_url"]
        self.assertTrue(all(part["image_url"]["detail"] == "high" for part in image_parts))
        self.assertEqual(result["value"], "test")

    def test_siliconflow_selector_keeps_only_vision_instruct_models(self):
        self.assertTrue(self.extractor._is_vision_instruct_model("Qwen/Qwen3-VL-32B-Instruct"))
        self.assertTrue(self.extractor._is_vision_instruct_model("zai-org/GLM-4.6V"))
        self.assertFalse(self.extractor._is_vision_instruct_model("Qwen/Qwen3-VL-32B-Thinking"))
        self.assertFalse(self.extractor._is_vision_instruct_model("Qwen/Qwen3-32B"))

    def test_backend_label_names_known_clouds_and_keeps_others_neutral(self):
        self.extractor.backend = "lmstudio"
        self.extractor.host = "https://openrouter.ai/api"
        self.assertEqual(self.extractor._backend_label(), "OpenRouter")
        self.extractor.host = "https://ai.example.com"
        self.assertEqual(self.extractor._backend_label(), "Сервер ИИ")
        self.extractor.host = "http://127.0.0.1:1234"
        self.assertEqual(self.extractor._backend_label(), "LM Studio")
        self.extractor.host = "http://192.168.0.19:1234"
        self.assertEqual(self.extractor._backend_label(), "LM Studio")
        self.extractor.backend = "ollama"
        self.assertEqual(self.extractor._backend_label(), "Ollama")
        self.extractor.backend = "siliconflow"
        self.assertEqual(self.extractor._backend_label(), "SiliconFlow")

    def test_empty_stream_is_retried_once_then_reported_with_server_name(self):
        empty = Mock(status_code=200)
        empty.iter_lines.return_value = [
            b'data: {"choices":[{"delta":null}]}',
            b'data: {"choices":[{"delta":{}}]}',
            b"data: [DONE]",
        ]
        self.extractor._session.post.return_value = empty
        self.extractor.backend = "lmstudio"
        self.extractor.host = "https://openrouter.ai/api"

        result = self.extractor._call_openai_compatible("system", "prompt", b"\x89PNG-x")

        self.assertEqual(self.extractor._session.post.call_count, 2, "пустой поток повторяем один раз")
        self.assertIn("OpenRouter вернул пустой ответ", result["error"])
        self.assertNotIn("LM Studio", result["error"])

    def test_parse_failure_names_the_actual_server(self):
        garbage = Mock(status_code=200)
        garbage.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"\\u041f\\u0440\\u0438\\u0432\\u0435\\u0442"}}]}',
            b"data: [DONE]",
        ]
        self.extractor._session.post.return_value = garbage
        self.extractor.backend = "lmstudio"
        self.extractor.host = "https://openrouter.ai/api"

        result = self.extractor._call_openai_compatible("system", "prompt", b"\x89PNG-x")

        self.assertEqual(result["error"], "Не удалось распарсить ответ OpenRouter")
        self.assertEqual(self.extractor._session.post.call_count, 1, "повтор бывает только при пустом ответе")

    def test_google_host_is_normalized_to_clean_base(self):
        self.assertEqual(
            DocumentExtractor._clean_host("https://generativelanguage.googleapis.com/v1beta/openai"),
            "https://generativelanguage.googleapis.com",
        )
        self.assertEqual(
            DocumentExtractor._clean_host("generativelanguage.googleapis.com/v1beta"),
            "https://generativelanguage.googleapis.com",
        )

    def test_google_models_are_listed_and_filtered(self):
        response = Mock(status_code=200)
        response.json.return_value = {"models": [
            {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent", "countTokens"]},
            {"name": "models/gemini-pro-latest", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-2.5-flash-preview-tts", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-2.5-flash-image", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/nano-banana-pro-preview", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
        ]}
        self.extractor._session.get.return_value = response
        self.extractor.host = "https://generativelanguage.googleapis.com"
        self.extractor.set_api_key("secret")

        models = self.extractor.get_available_models()

        self.assertEqual(self.extractor.backend, "google")
        self.assertEqual(models, ["gemini-pro-latest", "gemini-2.5-flash"])
        self.assertEqual(self.extractor._session.get.call_args.args[0],
                         "https://generativelanguage.googleapis.com/v1beta/models")
        self.extractor._session.headers.__setitem__.assert_any_call("x-goog-api-key", "secret")
        self.extractor._session.headers.pop.assert_any_call("Authorization", None)
        self.assertEqual(self.extractor._backend_label(), "Google AI Studio")

    def test_authorization_header_switches_with_host(self):
        self.extractor.set_api_key("secret")
        self.extractor._session.headers.__setitem__.assert_any_call("Authorization", "Bearer secret")
        self.extractor.set_host("https://generativelanguage.googleapis.com")
        self.extractor._session.headers.__setitem__.assert_any_call("x-goog-api-key", "secret")
        self.extractor.set_host("https://openrouter.ai/api")
        self.extractor._session.headers.__setitem__.assert_any_call("Authorization", "Bearer secret")

    def test_google_chat_uses_native_gemini_endpoint(self):
        response = Mock(status_code=200)
        response.iter_lines.return_value = [
            b'data: {"candidates":[{"content":{"parts":[{"thought":true,"text":"thinking"}]}}]}',
            b'data: {"candidates":[{"content":{"parts":[{"text":"{\\"value\\":\\"ok\\"}"}]}}],'
            b'"usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":4,"totalTokenCount":15}}',
        ]
        self.extractor._session.post.return_value = response
        self.extractor.set_host("https://generativelanguage.googleapis.com")
        self.extractor.backend = "google"
        self.extractor.model_name = "gemini-2.5-flash"
        self.extractor.set_api_key("secret")
        self.extractor._session.headers.__setitem__.reset_mock()

        result = self.extractor._call_google("system", "prompt", b"\x89PNG-x")

        self.assertEqual(
            self.extractor._session.post.call_args.args[0],
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
        )
        self.assertNotIn("headers", self.extractor._session.post.call_args.kwargs,
                         "ключ берётся из заголовков сессии, а не аргументом запроса")
        payload = self.extractor._session.post.call_args.kwargs["json"]
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "system")
        self.assertTrue(payload["contents"][0]["parts"][1]["inlineData"]["data"])
        self.assertEqual(result["value"], "ok")
        self.assertEqual(result["_stats"]["prompt_tokens"], 11)

    def test_rate_limit_is_retried_after_pause(self):
        rate_limited = Mock(status_code=429, headers={"Retry-After": "1"})
        rate_limited.json.return_value = {"error": {"message": "Provider returned error"}}
        ok = Mock(status_code=200)
        ok.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"{\\"value\\":\\"ok\\"}"}}]}',
            b"data: [DONE]",
        ]
        self.extractor._session.post.side_effect = [rate_limited, ok]
        self.extractor.backend = "lmstudio"
        self.extractor.host = "https://openrouter.ai"

        with patch("time.sleep") as sleep:
            result = self.extractor._call_openai_compatible("system", "prompt", b"\x89PNG-x")

        self.assertEqual(self.extractor._session.post.call_count, 2)
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 1.0)
        self.assertEqual(result["value"], "ok")

    def test_rate_limit_error_reports_provider_details(self):
        rate_limited = Mock(status_code=429, headers={})
        rate_limited.json.return_value = {
            "error": {
                "message": "Provider returned error",
                "metadata": {"raw": "rate limit exceeded", "provider_name": "openrouter"},
            }
        }
        self.extractor._session.post.return_value = rate_limited
        self.extractor.backend = "lmstudio"
        self.extractor.host = "https://openrouter.ai"

        with patch("time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                self.extractor._call_openai_compatible("system", "prompt", b"\x89PNG-x")

        message = str(ctx.exception)
        self.assertIn("OpenRouter вернул ошибку (429)", message)
        self.assertIn("rate limit exceeded", message)
        self.assertIn("openrouter", message)


if __name__ == "__main__":
    unittest.main()
