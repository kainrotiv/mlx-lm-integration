import json
import logging
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from .settings import Settings

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    pass


class Provider:
    def complete(
        self,
        settings: Settings,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> str | Iterator[str]:
        raise NotImplementedError


class MLXProvider(Provider):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded_model: str | None = None
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self, model_name: str) -> tuple[Any, Any]:
        if not model_name:
            raise ProviderError("Select or download an MLX-compatible model first.")
        if self._loaded_model != model_name:
            try:
                from mlx_lm import load
            except ImportError as error:
                raise ProviderError(
                    "MLX is available only on Apple Silicon Macs. Install this project with "
                    "`pip install -e '.[mlx]'` on macOS arm64."
                ) from error
            logger.info("Loading MLX model %s", model_name)
            try:
                self._model, self._tokenizer = load(model_name)
            except Exception as error:
                raise ProviderError(f"Could not load MLX model {model_name!r}: {error}") from error
            self._loaded_model = model_name
        return self._model, self._tokenizer

    @staticmethod
    def _prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (TypeError, ValueError):
                logger.info("Tokenizer chat template unavailable; using plain role formatting")
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"

    def complete(
        self,
        settings: Settings,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> str | Iterator[str]:
        with self._lock:
            model, tokenizer = self._load(settings.model)
            prompt = self._prompt(tokenizer, messages)
            try:
                if stream:
                    from mlx_lm import stream_generate

                    def generate() -> Iterator[str]:
                        with self._lock:
                            try:
                                for result in stream_generate(
                                    model,
                                    tokenizer,
                                    prompt,
                                    max_tokens=max_tokens,
                                    temp=temperature,
                                ):
                                    if result.text:
                                        yield result.text
                            except Exception as error:
                                raise ProviderError(f"MLX generation failed: {error}") from error

                    return generate()
                from mlx_lm import generate

                return generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                )
            except ImportError as error:
                raise ProviderError("The installed mlx-lm package does not expose the required API.") from error
            except ProviderError:
                raise
            except Exception as error:
                raise ProviderError(f"MLX generation failed: {error}") from error


def _http_json(url: str, payload: dict[str, Any], timeout: int = 300) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read(4096).decode("utf-8", errors="replace")
        raise ProviderError(f"Model server returned HTTP {error.code}: {body}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ProviderError(f"Could not reach model server at {url}: {error}") from error


class OllamaProvider(Provider):
    def complete(
        self,
        settings: Settings,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> str | Iterator[str]:
        if not settings.model:
            raise ProviderError("Enter an installed Ollama model name in Settings.")
        url = settings.ollama_url.rstrip("/") + "/api/chat"
        payload = {
            "model": settings.model,
            "messages": messages,
            "stream": stream,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        if not stream:
            result = _http_json(url, payload)
            message = result.get("message") if isinstance(result, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str):
                raise ProviderError("Ollama returned an invalid chat completion response.")
            return content

        def generate() -> Iterator[str]:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    for line in response:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        if item.get("error"):
                            raise ProviderError(f"Ollama generation failed: {item['error']}")
                        content = item.get("message", {}).get("content", "")
                        if not isinstance(content, str):
                            raise ProviderError("Ollama returned an invalid streaming response.")
                        if content:
                            yield content
            except urllib.error.HTTPError as error:
                body = error.read(4096).decode("utf-8", errors="replace")
                raise ProviderError(f"Ollama returned HTTP {error.code}: {body}") from error
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                raise ProviderError(f"Could not reach Ollama at {url}: {error}") from error

        return generate()


class LlamaCppProvider(Provider):
    def complete(
        self,
        settings: Settings,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> str | Iterator[str]:
        if not settings.model:
            raise ProviderError("Enter the model name served by llama.cpp in Settings.")
        url = settings.llama_cpp_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": settings.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if not stream:
            result = _http_json(url, payload)
            try:
                content = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise ProviderError("llama.cpp returned an invalid chat completion response.") from error
            if not isinstance(content, str):
                raise ProviderError("llama.cpp returned an invalid chat completion response.")
            return content

        def generate() -> Iterator[str]:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    for line in response:
                        decoded = line.decode("utf-8").strip()
                        if not decoded.startswith("data:"):
                            continue
                        data = decoded[5:].strip()
                        if data == "[DONE]":
                            return
                        item = json.loads(data)
                        choices = item.get("choices", [])
                        if choices:
                            content = choices[0].get("delta", {}).get("content", "")
                            if not isinstance(content, str):
                                raise ProviderError("llama.cpp returned an invalid streaming response.")
                            if content:
                                yield content
            except urllib.error.HTTPError as error:
                body = error.read(4096).decode("utf-8", errors="replace")
                raise ProviderError(f"llama.cpp returned HTTP {error.code}: {body}") from error
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                raise ProviderError(f"Could not reach llama.cpp at {url}: {error}") from error

        return generate()


def default_providers() -> dict[str, Provider]:
    return {
        "mlx": MLXProvider(),
        "ollama": OllamaProvider(),
        "llama_cpp": LlamaCppProvider(),
    }
