import json
import types
import io
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from local_ai.huggingface import download_model, search_models
from local_ai.providers import LlamaCppProvider, MLXProvider, OllamaProvider
from local_ai.settings import Settings


class MockModelServer(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        self.requests.append((self.path, payload))
        is_stream = payload.get("stream", False)
        if self.path == "/api/chat":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            if is_stream:
                for content in ("Ollama ", "response"):
                    self.wfile.write(
                        json.dumps({"message": {"content": content}, "done": False}).encode() + b"\n"
                    )
            else:
                self.wfile.write(json.dumps({"message": {"content": "Ollama response"}}).encode())
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if is_stream:
            for content in ("llama.cpp ", "response"):
                chunk = {"choices": [{"delta": {"content": content}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            result = {"choices": [{"message": {"content": "llama.cpp response"}}]}
            self.wfile.write(json.dumps(result).encode())

    def log_message(self, format_string, *args):
        pass


class ProviderTests(unittest.TestCase):
    def setUp(self):
        MockModelServer.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), MockModelServer)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.messages = [{"role": "user", "content": "hello"}]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_ollama_provider_non_streaming(self):
        result = OllamaProvider().complete(
            Settings(provider="ollama", model="llama3.2", ollama_url=self.url),
            self.messages,
            32,
            0.4,
            False,
        )
        self.assertEqual(result, "Ollama response")
        path, request = MockModelServer.requests[-1]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(request["options"]["num_predict"], 32)
        self.assertFalse(request["stream"])

    def test_ollama_provider_streaming(self):
        result = OllamaProvider().complete(
            Settings(provider="ollama", model="llama3.2", ollama_url=self.url),
            self.messages,
            32,
            0.4,
            True,
        )
        self.assertEqual("".join(result), "Ollama response")

    def test_llama_cpp_provider_non_streaming_and_streaming(self):
        settings = Settings(
            provider="llama_cpp",
            model="local-model",
            llama_cpp_url=self.url,
        )
        provider = LlamaCppProvider()
        result = provider.complete(settings, self.messages, 32, 0.4, False)
        self.assertEqual(result, "llama.cpp response")
        streamed = provider.complete(settings, self.messages, 32, 0.4, True)
        self.assertEqual("".join(streamed), "llama.cpp response")
        self.assertEqual(
            [path for path, _ in MockModelServer.requests],
            ["/v1/chat/completions", "/v1/chat/completions"],
        )

    def test_mlx_provider_formats_chat_and_streams_tokens(self):
        loaded = []
        tokenizer = types.SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: "formatted prompt"
        )
        mlx_lm = types.ModuleType("mlx_lm")

        def load(model_name):
            loaded.append(model_name)
            return object(), tokenizer

        mlx_lm.load = load
        mlx_lm.generate = lambda model, tokenizer, **kwargs: "MLX response"
        mlx_lm.stream_generate = lambda model, tokenizer, prompt, **kwargs: [
            types.SimpleNamespace(text="MLX "),
            types.SimpleNamespace(text="response"),
        ]
        with patch.dict("sys.modules", {"mlx_lm": mlx_lm}):
            provider = MLXProvider()
            settings = Settings(model="/local/model")
            result = provider.complete(settings, self.messages, 32, 0.4, False)
            streamed = provider.complete(settings, self.messages, 32, 0.4, True)
        self.assertEqual(result, "MLX response")
        self.assertEqual("".join(streamed), "MLX response")
        self.assertEqual(loaded, ["/local/model"])

    def test_hugging_face_download_uses_local_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target_root = Path(directory)
            with (
                patch("local_ai.huggingface.default_data_dir", return_value=target_root),
                patch.dict(
                    "sys.modules",
                    {
                        "huggingface_hub": type(
                            "FakeHub",
                            (),
                            {
                                "snapshot_download": staticmethod(
                                    lambda repo_id, local_dir: Path(local_dir)
                                )
                            },
                        )()
                    },
                ),
            ):
                result = download_model("org-name/model-name")
            self.assertEqual(result, target_root / "models" / "org-name" / "model-name")
            self.assertTrue(result.parent.exists())

    def test_hugging_face_search_maps_public_api_results(self):
        response = io.BytesIO(
            json.dumps(
                [{"id": "org/model", "downloads": 123, "likes": 4, "pipeline_tag": "text-generation"}]
            ).encode()
        )
        with patch("local_ai.huggingface.urllib.request.urlopen", return_value=response):
            models = search_models("org")
        self.assertEqual(
            models,
            [
                {
                    "id": "org/model",
                    "downloads": 123,
                    "likes": 4,
                    "pipeline_tag": "text-generation",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
