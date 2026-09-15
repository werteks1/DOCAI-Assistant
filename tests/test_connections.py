import copy
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

import config
from compiler import connections
from compiler.pipeline import CompilerPipeline
from extractor import DocumentExtractor


def memory_settings(initial=None):
    """Подменяет settings.json: чистый словарь вместо файла."""
    data = copy.deepcopy(initial or {})

    def load():
        return copy.deepcopy(data)

    def save(payload, drop=()):
        data.update(copy.deepcopy(payload))
        for key in drop:
            data.pop(key, None)
        return copy.deepcopy(data)

    def remove():
        data.clear()

    return data, Mock(side_effect=load), Mock(side_effect=save), Mock(side_effect=remove)


def fake_pipeline(unreachable=()):
    """Пайплайн с управляемым «сервером»: адреса из unreachable не отвечают."""
    pipeline = CompilerPipeline.__new__(CompilerPipeline)
    extractor = SimpleNamespace(
        field_locator=SimpleNamespace(detector_name=config.PADDLE_DETECTOR, available=False),
        extract_from_image=Mock(),
        model_name=config.DEFAULT_MODEL,
        backend="lmstudio",
        host=config.OLLAMA_HOST,
        api_key="",
        _clean_host=DocumentExtractor._clean_host,
        set_host=lambda host: setattr(extractor, "host", DocumentExtractor._clean_host(host)),
        set_api_key=lambda key: setattr(extractor, "api_key", (key or "").strip()),
        set_model=lambda model: setattr(extractor, "model_name", model) or model,
        apply_default_params=Mock(),
        params_snapshot=lambda: {"temperature": 0.0},
    )
    pipeline.extractor = extractor
    pipeline.available_models = []
    pipeline.connect_calls = []
    unreachable = set(unreachable)

    def connect(host, api_key=None):
        clean = DocumentExtractor._clean_host(host)
        pipeline.connect_calls.append((clean, api_key or ""))
        if clean in unreachable:
            return {"host": extractor.host, "backend": extractor.backend,
                    "model": extractor.model_name, "models": [], "connected": False}
        extractor.host = clean
        extractor.api_key = (api_key or "").strip()
        extractor.backend = "lmstudio"
        pipeline.available_models = ["qwen2.5-vl-7b", "llava-v1.6"]
        if extractor.model_name not in pipeline.available_models:
            extractor.model_name = "qwen2.5-vl-7b"
        return {"host": clean, "backend": "lmstudio", "model": extractor.model_name,
                "models": list(pipeline.available_models), "connected": True}

    def probe(host, api_key=None):
        snapshot = (extractor.host, extractor.backend, extractor.model_name,
                    extractor.api_key, list(pipeline.available_models))
        try:
            return connect(host, api_key)
        finally:
            (extractor.host, extractor.backend, extractor.model_name,
             extractor.api_key) = snapshot[:4]
            pipeline.available_models = snapshot[4]

    pipeline.connect = connect
    pipeline.probe = probe
    return pipeline


class ConnectionsStoreTests(unittest.TestCase):
    def test_legacy_single_host_is_migrated_into_active_connection(self):
        items, active_id = connections.from_saved(
            {"host": "http://127.0.0.1:1234", "api_key": "k", "model": "qwen"}
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(active_id, items[0]["id"])
        self.assertEqual(items[0]["host"], "http://127.0.0.1:1234")
        self.assertEqual(items[0]["api_key"], "k")
        self.assertEqual(items[0]["model"], "qwen")
        self.assertEqual(items[0]["name"], "LM Studio")

    def test_modern_list_wins_over_legacy_fields_and_skips_broken_entries(self):
        items, active_id = connections.from_saved({
            "host": "http://legacy:1",
            "connections": [
                {"id": "conn-a", "name": "A", "host": "https://api.siliconflow.com", "api_key": "k1", "model": "m1"},
                {"name": "без адреса"},
                {"id": "conn-a", "host": "https://seekai.cc", "api_key": "k2"},
            ],
            "active_connection_id": "conn-missing",
        })
        self.assertEqual([c["id"] for c in items], ["conn-a", items[1]["id"]])
        self.assertTrue(items[1]["id"].startswith("conn-"))
        self.assertEqual(items[0]["name"], "A")
        self.assertEqual(items[1]["name"], "SeekAI")
        self.assertEqual(active_id, "conn-a")

    def test_active_falls_back_to_first_and_names_are_guessed(self):
        items, active_id = connections.from_saved({
            "connections": [{"id": "conn-b", "host": "http://127.0.0.1:11434"}],
            "active_connection_id": "",
        })
        self.assertEqual(active_id, "conn-b")
        self.assertEqual(items[0]["name"], "Ollama")
        self.assertEqual(connections.guess_name("http://192.168.0.19:1234"), "LM Studio")
        self.assertEqual(connections.guess_name("https://generativelanguage.googleapis.com"), "Google AI Studio")
        self.assertEqual(connections.guess_name("https://example.com"), connections.DEFAULT_NAME)

    def test_new_id_never_collides(self):
        existing = {"conn-1"}
        for _ in range(20):
            candidate = connections.new_id(existing)
            self.assertNotIn(candidate, existing)
            existing.add(candidate)


class ConnectionsApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "compiler._connections_api",
            Path(__file__).resolve().parents[1] / "compiler" / "api.py",
        )
        cls.api = importlib.util.module_from_spec(spec)
        with patch("compiler.settings_store.load_settings", return_value={}), \
                patch("compiler.pipeline.CompilerPipeline", return_value=fake_pipeline()), \
                patch("compiler.auth_store.init_db"):
            spec.loader.exec_module(cls.api)

    def setUp(self):
        self.data, load, save, remove = memory_settings()
        for name, mock in (("load_settings", load), ("save_settings", save),
                           ("remove_settings", remove)):
            patcher = patch.object(self.api.settings_store, name, mock)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.log = patch.object(self.api.auth_store, "log_activity").start()
        self.addCleanup(self.log.stop)
        self.pipeline = fake_pipeline()
        self.api.pipeline = self.pipeline
        self.admin = {"id": 1, "username": "boss", "role": "admin", "must_change": False}
        self.api.app.dependency_overrides[self.api.require_admin] = lambda: self.admin
        self.addCleanup(self.api.app.dependency_overrides.clear)
        self.client = TestClient(self.api.app)
        self.addCleanup(self.client.close)

    def add(self, host, **payload):
        body = {"name": "", "host": host, "api_key": None, "activate": True, **payload}
        return self.client.post("/api/connections", json=body)

    def test_add_reachable_connection_becomes_active_and_masks_key(self):
        response = self.add("127.0.0.1:1234/v1/chat/completions", name="Кабинет 12", api_key="secret-key")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        saved = body["saved"]
        self.assertEqual(len(saved["connections"]), 1)
        conn = saved["connections"][0]
        self.assertEqual(conn["host"], "http://127.0.0.1:1234")
        self.assertEqual(conn["name"], "Кабинет 12")
        self.assertTrue(conn["active"])
        self.assertTrue(conn["api_key_set"])
        self.assertEqual(conn["model"], "qwen2.5-vl-7b")
        self.assertNotIn("secret-key", json.dumps(body, ensure_ascii=False))
        stored = self.data["connections"][0]
        self.assertEqual(stored["api_key"], "secret-key")
        self.assertEqual(self.data["active_connection_id"], stored["id"])

    def test_add_without_activate_keeps_previous_active(self):
        self.add("http://127.0.0.1:1234", api_key="")
        first_id = self.data["connections"][0]["id"]
        self.add("http://127.0.0.1:11434", activate=False)
        self.assertEqual(self.data["active_connection_id"], first_id)

    def test_add_reachable_with_model_choice_applies_it(self):
        response = self.add("http://127.0.0.1:1234", api_key="", model="llava-v1.6")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.data["connections"][0]["model"], "llava-v1.6")
        self.assertEqual(self.pipeline.extractor.model_name, "llava-v1.6")

    def test_add_model_missing_on_server_falls_back_to_server_default(self):
        response = self.add("http://127.0.0.1:1234", api_key="", model="stale-model")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.data["connections"][0]["model"], "qwen2.5-vl-7b")

    def test_update_changed_host_with_model_choice_applies_it(self):
        self.add("http://127.0.0.1:1234", api_key="")
        conn_id = self.data["connections"][0]["id"]
        response = self.client.post(
            f"/api/connections/{conn_id}", json={"host": "127.0.0.1:11434", "model": "llava-v1.6"}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.data["connections"][0]["model"], "llava-v1.6")
        self.assertEqual(self.pipeline.extractor.model_name, "llava-v1.6")

    def test_update_active_model_missing_on_server_is_rejected(self):
        self.add("http://127.0.0.1:1234", api_key="")
        conn_id = self.data["connections"][0]["id"]
        response = self.client.post(f"/api/connections/{conn_id}", json={"model": "bad-model"})
        self.assertEqual(response.status_code, 422)
        self.assertIn("не найдена", response.json()["detail"])

    def test_unreachable_add_is_saved_and_first_becomes_active_with_warning(self):
        self.pipeline = fake_pipeline(unreachable={"http://10.0.0.9:1234"})
        self.api.pipeline = self.pipeline
        response = self.add("http://10.0.0.9:1234", api_key="")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(len(body["saved"]["connections"]), 1)
        self.assertTrue(body["saved"]["connections"][0]["active"])
        self.assertIn("не ответил", body["warning"])

    def test_unreachable_second_add_keeps_previous_active(self):
        self.add("http://127.0.0.1:1234", api_key="")
        self.pipeline = fake_pipeline(unreachable={"http://10.0.0.9:1234"})
        self.api.pipeline = self.pipeline
        response = self.add("http://10.0.0.9:1234", api_key="key")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        active = [c for c in body["saved"]["connections"] if c["active"]]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["host"], "http://127.0.0.1:1234")

    def test_duplicate_host_and_limit_are_rejected(self):
        self.add("http://127.0.0.1:1234", api_key="")
        response = self.add("http://127.0.0.1:1234/v1", api_key="")
        self.assertEqual(response.status_code, 422)
        self.assertIn("уже есть", response.json()["detail"])
        self.data["connections"] = [
            {"id": f"conn-{i}", "name": f"n{i}", "host": f"http://10.0.0.{i}:1234",
             "api_key": "", "model": ""} for i in range(connections.MAX_CONNECTIONS)
        ]
        response = self.add("http://127.0.0.1:9999", api_key="")
        self.assertEqual(response.status_code, 422)
        self.assertIn("лимит", response.json()["detail"].lower())

    def test_activate_unreachable_is_rejected_and_keeps_active(self):
        self.add("http://127.0.0.1:1234", api_key="")
        first_id = self.data["connections"][0]["id"]
        self.add("http://127.0.0.1:11434", activate=False)
        second_id = self.data["connections"][1]["id"]
        self.api.pipeline = fake_pipeline(unreachable={"http://127.0.0.1:11434"})
        response = self.client.post(f"/api/connections/{second_id}/activate")
        self.assertEqual(response.status_code, 422)
        self.assertIn("не ответил", response.json()["detail"])
        self.assertEqual(self.data["active_connection_id"], first_id)

    def test_update_keeps_clears_and_renames(self):
        self.add("http://127.0.0.1:1234", name="Старое", api_key="old-key")
        conn_id = self.data["connections"][0]["id"]
        response = self.client.post(f"/api/connections/{conn_id}", json={"name": "Новое"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.data["connections"][0]["name"], "Новое")
        self.assertEqual(self.data["connections"][0]["api_key"], "old-key")
        response = self.client.post(f"/api/connections/{conn_id}", json={"api_key": ""})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.data["connections"][0]["api_key"], "")
        self.assertFalse(response.json()["saved"]["connections"][0]["api_key_set"])

    def test_update_active_connection_reconnects_and_saves_model(self):
        self.add("http://127.0.0.1:1234", api_key="")
        conn_id = self.data["connections"][0]["id"]
        response = self.client.post(f"/api/connections/{conn_id}", json={"host": "127.0.0.1:11434"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.data["connections"][0]["host"], "http://127.0.0.1:11434")
        self.assertEqual(self.pipeline.extractor.host, "http://127.0.0.1:11434")

    def test_check_uses_stored_key_and_does_not_switch(self):
        self.add("http://127.0.0.1:1234", api_key="stored-key")
        self.add("http://127.0.0.1:11434", api_key="other", activate=False)
        active_host = self.pipeline.extractor.host
        conn_id = self.data["connections"][0]["id"]
        response = self.client.post("/api/connections/check", json={"id": conn_id, "host": "http://127.0.0.1:1234"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["connected"])
        self.assertIn(("http://127.0.0.1:1234", "stored-key"), self.pipeline.connect_calls)
        self.assertEqual(self.pipeline.extractor.host, active_host)

    def test_delete_active_switches_to_reachable_remaining(self):
        self.add("http://127.0.0.1:1234", api_key="")
        self.add("http://127.0.0.1:11434", api_key="")
        first_id, second_id = (c["id"] for c in self.data["connections"])
        response = self.client.delete(f"/api/connections/{second_id}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.data["active_connection_id"], first_id)
        self.assertEqual(len(self.data["connections"]), 1)

    def test_delete_last_connection_returns_to_defaults(self):
        self.add("http://127.0.0.1:1234", api_key="")
        conn_id = self.data["connections"][0]["id"]
        response = self.client.delete(f"/api/connections/{conn_id}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.data["connections"], [])
        self.assertEqual(self.pipeline.extractor.host, config.OLLAMA_HOST)
        self.assertEqual(self.pipeline.extractor.model_name, config.DEFAULT_MODEL)

    def test_model_choice_is_persisted_in_active_connection(self):
        self.add("http://127.0.0.1:1234", api_key="")
        response = self.client.post("/api/model", json={"model": "llava-v1.6"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.data["connections"][0]["model"], "llava-v1.6")

    def test_legacy_settings_migrate_on_read_and_are_replaced_on_write(self):
        self.data.update({"host": "http://127.0.0.1:1234", "api_key": "legacy-key", "model": "qwen2.5-vl-7b"})
        response = self.client.get("/api/settings")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(len(body["saved"]["connections"]), 1)
        self.assertNotIn("legacy-key", json.dumps(body, ensure_ascii=False))
        self.add("http://127.0.0.1:11434", api_key="")
        self.assertNotIn("host", self.data)
        self.assertNotIn("api_key", self.data)
        self.assertNotIn("model", self.data)
        self.assertEqual(len(self.data["connections"]), 2)

    def test_non_admin_cannot_manage_connections(self):
        self.api.app.dependency_overrides[self.api.require_admin] = lambda: (_ for _ in ()).throw(
            HTTPException(status_code=403, detail="Недостаточно прав: требуется администратор")
        )
        self.assertEqual(self.client.get("/api/connections").status_code, 403)
        self.assertEqual(self.add("http://127.0.0.1:1234").status_code, 403)
        self.assertEqual(self.data, {})


if __name__ == "__main__":
    unittest.main()
