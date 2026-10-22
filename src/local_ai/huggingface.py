import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .settings import default_data_dir


MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


class HuggingFaceError(RuntimeError):
    pass


def search_models(query: str, limit: int = 20) -> list[dict[str, Any]]:
    query = query.strip()
    if not query or len(query) > 200:
        raise ValueError("search query must contain 1 to 200 characters")
    params = urllib.parse.urlencode(
        {"search": query, "limit": limit, "sort": "downloads", "direction": "-1"}
    )
    request = urllib.request.Request(
        f"https://huggingface.co/api/models?{params}",
        headers={"User-Agent": "LocalAI/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            results = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise HuggingFaceError(f"Hugging Face returned HTTP {error.code}.") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise HuggingFaceError(f"Could not search Hugging Face: {error}") from error
    if not isinstance(results, list):
        raise HuggingFaceError("Hugging Face returned an invalid model search response.")
    models = []
    for item in results:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise HuggingFaceError("Hugging Face returned an invalid model search result.")
        models.append(
            {
                "id": item["id"],
                "downloads": item.get("downloads") if isinstance(item.get("downloads"), int) else 0,
                "likes": item.get("likes") if isinstance(item.get("likes"), int) else 0,
                "pipeline_tag": item.get("pipeline_tag")
                if isinstance(item.get("pipeline_tag"), str)
                else None,
            }
        )
    return models


def download_model(repo_id: str) -> Path:
    repo_id = repo_id.strip()
    if not MODEL_ID.fullmatch(repo_id):
        raise ValueError("repo_id must be a Hugging Face model id such as `org/model-name`")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise HuggingFaceError(
            "Hugging Face downloads require `huggingface-hub`. Install with `pip install -e .`."
        ) from error
    target = default_data_dir() / "models" / repo_id
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target),
            )
        )
    except Exception as error:
        raise HuggingFaceError(f"Could not download {repo_id!r} from Hugging Face: {error}") from error


def installed_models() -> list[dict[str, str]]:
    root = default_data_dir() / "models"
    if not root.exists():
        return []
    return [
        {"id": config.parent.relative_to(root).as_posix(), "path": str(config.parent)}
        for config in sorted(root.rglob("config.json"))
        if config.parent.is_dir()
    ]
