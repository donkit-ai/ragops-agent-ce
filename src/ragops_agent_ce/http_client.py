from __future__ import annotations

from typing import Any

import httpx

from .config import Settings, load_settings


class RAGOpsClient:
    """Minimal HTTP client wrapper for RAGOps APIs."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self._client = httpx.Client(base_url=self.settings.api_url, headers=self._headers())

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"User-Agent": "ragops-agent-ce/0.1"}
        if self.settings.api_key:
            headers["X-API-Key"] = self.settings.api_key
        return headers

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self._client.get(path, **kwargs)

    def post(self, path: str, json: Any | None = None, **kwargs: Any) -> httpx.Response:
        return self._client.post(path, json=json, **kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RAGOpsClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()
