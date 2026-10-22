import json
import logging
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .huggingface import HuggingFaceError, download_model, installed_models, search_models
from .providers import Provider, ProviderError, default_providers
from .settings import DEFAULT_PORT, SettingsStore

logger = logging.getLogger(__name__)
MAX_BODY_BYTES = 1024 * 1024
MAX_COMPLETION_TOKENS = 8192


def make_handler(
    settings: SettingsStore,
    providers: dict[str, Provider] | None = None,
) -> type[BaseHTTPRequestHandler]:
    provider_map = providers or default_providers()

    class Handler(BaseHTTPRequestHandler):
        server_version = "LocalAI/0.1"

        def log_message(self, format_string: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), format_string % args)

        def _send_json(self, status: int, value: Any) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Any:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("Content-Length must be an integer") from error
            if length < 0 or length > MAX_BODY_BYTES:
                raise ValueError(f"request body must not exceed {MAX_BODY_BYTES} bytes")
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("request body must contain valid UTF-8 JSON") from error

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/health":
                    self._send_json(200, {"status": "ok"})
                elif parsed.path == "/api/settings":
                    from dataclasses import asdict

                    self._send_json(200, asdict(settings.get()))
                elif parsed.path == "/api/models":
                    self._send_json(200, {"data": installed_models()})
                elif parsed.path == "/api/huggingface/search":
                    query = parse_qs(parsed.query).get("q", [""])[0]
                    self._send_json(200, {"data": search_models(query)})
                elif parsed.path == "/v1/models":
                    active = settings.get()
                    data = (
                        [{"id": active.model, "object": "model", "owned_by": active.provider}]
                        if active.model
                        else []
                    )
                    self._send_json(200, {"object": "list", "data": data})
                else:
                    self._send_json(404, {"error": {"message": "Not found", "type": "not_found"}})
            except (ValueError, HuggingFaceError) as error:
                self._send_json(400, {"error": {"message": str(error), "type": "invalid_request_error"}})
            except Exception as error:
                logger.exception("GET %s failed", parsed.path)
                self._send_json(500, {"error": {"message": str(error), "type": "internal_error"}})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                body = self._read_json()
                if parsed.path == "/api/settings":
                    from dataclasses import asdict

                    updated = settings.update(body)
                    self._send_json(200, asdict(updated))
                elif parsed.path == "/api/huggingface/download":
                    if not isinstance(body, dict) or not isinstance(body.get("repo_id"), str):
                        raise ValueError("repo_id is required")
                    path = download_model(body["repo_id"])
                    self._send_json(200, {"repo_id": body["repo_id"], "path": str(path)})
                elif parsed.path == "/v1/chat/completions":
                    self._chat(body)
                else:
                    self._send_json(404, {"error": {"message": "Not found", "type": "not_found"}})
            except (ValueError, HuggingFaceError) as error:
                self._send_json(400, {"error": {"message": str(error), "type": "invalid_request_error"}})
            except ProviderError as error:
                logger.warning("Provider request failed: %s", error)
                self._send_json(502, {"error": {"message": str(error), "type": "provider_error"}})
            except Exception as error:
                logger.exception("POST %s failed", parsed.path)
                self._send_json(500, {"error": {"message": str(error), "type": "internal_error"}})

        def _chat(self, body: Any) -> None:
            if not isinstance(body, dict):
                raise ValueError("request must be a JSON object")
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty array")
            normalized_messages: list[dict[str, str]] = []
            for message in messages:
                if (
                    not isinstance(message, dict)
                    or not isinstance(message.get("role"), str)
                    or not isinstance(message.get("content"), str)
                ):
                    raise ValueError("each message must have string role and content fields")
                normalized_messages.append(
                    {"role": message["role"], "content": message["content"]}
                )

            max_tokens = body.get("max_tokens", body.get("max_completion_tokens", 512))
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
                raise ValueError("max_tokens must be an integer")
            if not 1 <= max_tokens <= MAX_COMPLETION_TOKENS:
                raise ValueError(f"max_tokens must be between 1 and {MAX_COMPLETION_TOKENS}")
            temperature = body.get("temperature", 0.7)
            if (
                not isinstance(temperature, (int, float))
                or isinstance(temperature, bool)
                or not 0 <= temperature <= 2
            ):
                raise ValueError("temperature must be a number between 0 and 2")
            stream = body.get("stream", False)
            if not isinstance(stream, bool):
                raise ValueError("stream must be a boolean")

            active = settings.get()
            if not active.model:
                raise ProviderError("No model is selected. Configure a model in the Local AI app first.")
            provider = provider_map.get(active.provider)
            if provider is None:
                raise ProviderError(f"Provider {active.provider!r} is unavailable.")
            result = provider.complete(
                active,
                normalized_messages,
                max_tokens,
                float(temperature),
                stream,
            )
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    for text in result:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": active.model,
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                        self.wfile.flush()
                    final_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": active.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    self.wfile.write(f"data: {json.dumps(final_chunk)}\n\ndata: [DONE]\n\n".encode())
                except Exception as error:
                    logger.exception("Streaming completion failed")
                    event = {"error": {"message": str(error), "type": "provider_error"}}
                    self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode())
                return

            response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": active.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result},
                        "finish_reason": "stop",
                    }
                ],
            }
            self._send_json(200, response)

    return Handler


def create_server(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    settings_path: Path | None = None,
    providers: dict[str, Provider] | None = None,
) -> ThreadingHTTPServer:
    if host != "127.0.0.1":
        raise ValueError("The Local AI API must bind to loopback only.")
    store = SettingsStore(settings_path)
    return ThreadingHTTPServer((host, port), make_handler(store, providers))
