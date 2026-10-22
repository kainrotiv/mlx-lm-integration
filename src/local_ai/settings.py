import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import urlsplit


PROVIDERS = ("mlx", "ollama", "llama_cpp")
DEFAULT_PORT = 11435


def default_data_dir() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif os.uname().sysname == "Darwin":
        root = Path.home() / "Library/Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return root / "LocalAI"


@dataclass(frozen=True)
class Settings:
    provider: str = "mlx"
    model: str = ""
    ollama_url: str = "http://127.0.0.1:11434"
    llama_cpp_url: str = "http://127.0.0.1:8080"

    def validate(self) -> "Settings":
        if self.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of: {', '.join(PROVIDERS)}")
        if not isinstance(self.model, str) or len(self.model) > 2048:
            raise ValueError("model must be a string no longer than 2048 characters")
        for field, url in (
            ("ollama_url", self.ollama_url),
            ("llama_cpp_url", self.llama_cpp_url),
        ):
            if not isinstance(url, str) or len(url) > 2048:
                raise ValueError(f"{field} must be a URL no longer than 2048 characters")
            try:
                parsed = urlsplit(url)
                parsed.port
            except ValueError as error:
                raise ValueError(f"{field} is not a valid URL") from error
            if (
                parsed.scheme not in ("http", "https")
                or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise ValueError(f"{field} must use an HTTP(S) loopback address on this computer")
        return self

    @classmethod
    def from_dict(cls, value: Any) -> "Settings":
        if not isinstance(value, dict):
            raise ValueError("settings must be a JSON object")
        allowed = {"provider", "model", "ollama_url", "llama_cpp_url"}
        unexpected = set(value) - allowed
        if unexpected:
            raise ValueError(f"unknown setting(s): {', '.join(sorted(unexpected))}")
        settings = cls(**value)
        return settings.validate()


class SettingsStore:
    def __init__(self, path: Path | None = None):
        self.path = path or default_data_dir() / "settings.json"
        self._lock = RLock()

    def get(self) -> Settings:
        with self._lock:
            if not self.path.exists():
                return Settings()
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(f"Could not read settings at {self.path}: {error}") from error
            try:
                return Settings.from_dict(data)
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"Invalid settings at {self.path}: {error}") from error

    def update(self, changes: Any) -> Settings:
        if not isinstance(changes, dict):
            raise ValueError("settings must be a JSON object")
        allowed = {"provider", "model", "ollama_url", "llama_cpp_url"}
        unexpected = set(changes) - allowed
        if unexpected:
            raise ValueError(f"unknown setting(s): {', '.join(sorted(unexpected))}")
        with self._lock:
            current = self.get()
            updated = replace(current, **changes).validate()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.path.parent,
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    json.dump(asdict(updated), temporary, indent=2)
                    temporary.write("\n")
                temporary_path.replace(self.path)
            except OSError as error:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                raise RuntimeError(f"Could not save settings at {self.path}: {error}") from error
            return updated
