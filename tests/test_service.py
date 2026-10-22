import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from local_ai.huggingface import installed_models
from local_ai.providers import Provider
from local_ai.server import create_server
from local_ai.settings import Settings


class FakeProvider(Provider):
    def complete(self, settings, messages, max_tokens, temperature, stream):
        self.last_request = (settings, messages, max_tokens, temperature, stream)
        if stream:
            return iter(["Hello", " world"])
        return "Hello world"


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.provider = FakeProvider()
        self.server = create_server(
            port=0,
            settings_path=Path(self.temporary_directory.name) / "settings.json",
            providers={"mlx": self.provider},
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary_directory.cleanup()

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, response.headers, response.read()

    def test_health_and_loopback_api(self):
        status, _, body = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"status": "ok"})
        with self.assertRaises(ValueError):
            create_server(host="0.0.0.0", port=0)

    def test_settings_are_persisted_and_validated(self):
        status, _, body = self.request(
            "/api/settings",
            {"provider": "mlx", "model": "org/test-model"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["model"], "org/test-model")
        status, _, body = self.request("/api/settings")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["provider"], "mlx")
        with self.assertRaises(ValueError):
            Settings.from_dict({"ollama_url": "https://models.example.com"})
        request = urllib.request.Request(
            self.base_url + "/api/settings",
            data=json.dumps({"provider": "unknown"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=3)
        self.assertEqual(error.exception.code, 400)
        error.exception.read()
        error.exception.close()

    def test_openai_non_streaming_chat_completion(self):
        self.request("/api/settings", {"model": "org/test-model"})
        status, headers, body = self.request(
            "/v1/chat/completions",
            {
                "model": "org/test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 20,
            },
        )
        response = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(headers.get_content_type(), "application/json")
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(self.provider.last_request[2], 20)

    def test_openai_streaming_chat_completion(self):
        self.request("/api/settings", {"model": "org/test-model"})
        status, headers, body = self.request(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Hi"}], "stream": True},
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers.get_content_type(), "text/event-stream")
        self.assertIn(b'"content": "Hello"', body)
        self.assertIn(b"data: [DONE]", body)

    def test_rejects_bad_completion_input(self):
        self.request("/api/settings", {"model": "org/test-model"})
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps({"messages": [], "max_tokens": 0}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=3)
        self.assertEqual(error.exception.code, 400)
        error.exception.read()
        error.exception.close()

    def test_lists_downloaded_hugging_face_models_by_repository_path(self):
        model_path = Path(self.temporary_directory.name) / "models" / "org-name" / "model-name"
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text("{}", encoding="utf-8")
        with patch("local_ai.huggingface.default_data_dir", return_value=Path(self.temporary_directory.name)):
            models = installed_models()
        self.assertEqual(
            models,
            [{"id": "org-name/model-name", "path": str(model_path)}],
        )


if __name__ == "__main__":
    unittest.main()
