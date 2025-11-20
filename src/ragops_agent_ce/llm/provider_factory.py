from __future__ import annotations

import json

from ..config import Settings
from ..config import load_settings
from .base import LLMProvider
from .providers.anthropic import AnthropicProvider
from .providers.azure_openai import AzureOpenAIProvider
from .providers.donkit import DonkitProvider
from .providers.mock import MockProvider
from .providers.openai import OpenAIProvider
from .providers.vertex import VertexProvider

PROVIDERS_TYPES = [
    OpenAIProvider,
    AzureOpenAIProvider,
    AnthropicProvider,
    MockProvider,
    VertexProvider,
    DonkitProvider,
]

PROVIDER_TYPES: dict[str, type[LLMProvider]] = {p.name: p for p in PROVIDERS_TYPES}


def __get_vertex_credentials():
    credentials_path = load_settings().vertex_credentials
    if not credentials_path:
        raise ValueError("RAGOPS_VERTEX_CREDENTIALS is not set in .env file")

    import os
    from pathlib import Path

    # Expand user path
    credentials_path = os.path.expanduser(credentials_path)
    creds_file = Path(credentials_path)

    if not creds_file.exists():
        raise FileNotFoundError(
            f"Credentials file not found: {credentials_path}\n"
            "Please ensure RAGOPS_VERTEX_CREDENTIALS path is correctly set in .env file."
        )

    try:
        with open(creds_file) as f:
            credentials_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error reading credentials file: {e}")

    if not credentials_data:
        raise ValueError("Credentials file is empty or contains no data")

    return credentials_data


def get_provider(settings: Settings | None = None, llm_provider: str | None = None) -> LLMProvider:
    cfg = settings or load_settings()
    provider_key = (llm_provider or cfg.llm_provider or "mock").lower()
    cls = PROVIDER_TYPES.get(provider_key)
    if not cls:
        raise ValueError(f"Unknown LLM provider: {provider_key}")

    if provider_key == "vertex":
        credentials_data = __get_vertex_credentials()
        return cls(cfg, credentials_data=credentials_data)
    elif provider_key == "openrouter":
        # OpenRouter uses OpenAI-compatible API with custom base_url
        # Create a modified config with OpenRouter endpoint
        openrouter_cfg = cfg.model_copy(update={"openai_base_url": "https://openrouter.ai/api/v1"})
        return cls(openrouter_cfg)
    elif provider_key == "ollama":
        ollama_cfg = cfg.model_copy(
            update={"openai_base_url": settings.ollama_base_url, "openai_api_key": "ollama"}
        )
        return cls(ollama_cfg)
    return cls(cfg)
