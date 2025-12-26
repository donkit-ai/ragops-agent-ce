from __future__ import annotations

import json
import os
from pathlib import Path

from donkit.llm import LLMModelAbstract, ModelFactory

from ..config import Settings, load_settings
from .providers.mock import MockProvider


def _get_vertex_credentials(settings: Settings) -> dict:
    """Load Vertex AI credentials from file."""
    credentials_path = settings.vertex_credentials
    if not credentials_path:
        raise ValueError("RAGOPS_VERTEX_CREDENTIALS is not set in .env file")

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


def get_provider(
    settings: Settings | None = None, llm_provider: str | None = None, model_name: str | None = None
) -> LLMModelAbstract:
    """
    Create LLM provider using donkit-llm ModelFactory.

    Returns LLMModelAbstract instance configured for the specified provider.
    """
    cfg = settings or load_settings()
    provider_key = (llm_provider or cfg.llm_provider).lower()

    # Special case: mock provider
    if provider_key == "mock":
        return MockProvider(cfg)

    # Donkit provider - uses RagopsAPIGatewayClient
    if provider_key == "donkit":
        if not cfg.donkit_api_key:
            raise ValueError("DONKIT_API_KEY is not set")

        credentials = {
            "api_key": cfg.donkit_api_key,
            "base_url": cfg.donkit_base_url,
        }
        return ModelFactory.create_model(
            provider="donkit", model_name=None, credentials=credentials
        )

    model_name = model_name or cfg.llm_model or _get_default_model(provider_key)

    if provider_key == "openai":
        credentials = {
            "api_key": cfg.openai_api_key,
            "base_url": cfg.openai_base_url,
            "organization": cfg.openai_organization,
        }
        return ModelFactory.create_model("openai", model_name, credentials)

    elif provider_key == "azure_openai":
        credentials = {
            "api_key": cfg.azure_openai_api_key,
            "azure_endpoint": cfg.azure_openai_endpoint,
            "api_version": cfg.azure_openai_api_version or "2024-08-01-preview",
            "deployment_name": cfg.azure_openai_deployment,
        }
        return ModelFactory.create_model("azure_openai", model_name, credentials)

    elif provider_key == "anthropic" or provider_key == "claude":
        credentials = {
            "api_key": cfg.anthropic_api_key,
            "base_url": None,
        }
        return ModelFactory.create_model("claude", model_name, credentials)

    elif provider_key == "vertex":
        credentials_data = _get_vertex_credentials(cfg)
        credentials = {
            "project_id": credentials_data.get("project_id"),
            "location": "us-central1",
            "credentials": credentials_data,
        }
        return ModelFactory.create_model("vertex", model_name, credentials)

    elif provider_key == "gemini":
        credentials = {
            "api_key": cfg.gemini_api_key,
            "project_id": cfg.vertex_project_id,
            "location": cfg.vertex_location or "us-central1",
            "use_vertex": cfg.gemini_use_vertex or False,
        }
        return ModelFactory.create_model("gemini", model_name, credentials)

    elif provider_key == "ollama":
        ollama_url = cfg.ollama_base_url or "http://localhost:11434"
        if "/v1" not in ollama_url:
            ollama_url += "/v1"
        credentials = {
            "api_key": "ollama",
            "base_url": ollama_url,
        }
        return ModelFactory.create_model("openai", model_name, credentials)

    elif provider_key == "openrouter":
        credentials = {
            "api_key": cfg.openai_api_key,
            "base_url": "https://openrouter.ai/api/v1",
        }
        return ModelFactory.create_model("openai", model_name, credentials)

    else:
        raise ValueError(f"Unknown LLM provider: {provider_key}")


def _get_default_model(provider: str) -> str:
    """Get default model name for provider."""
    defaults = {
        "openai": "gpt-5.1-mini",
        "azure_openai": "gpt-5.1-mini",
        "anthropic": "claude-4-5-sonnet",
        "claude": "claude-4-5-sonnet",
        "vertex": "gemini-2.5-flash",
        "gemini": "gemini-2.5-flash",
        "ollama": "gpt-oss:20b",
        "openrouter": "openai/gpt-5.1-mini",
        "donkit": "gemini-2.5-flash",
    }
    return defaults.get(provider)
