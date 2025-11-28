"""
Shared credential checking utilities.

Provides unified logic for checking provider credentials across the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

from dotenv import dotenv_values


class ProviderConfig(TypedDict):
    """Configuration for a provider."""

    embedder_model: str
    generation_model: str


# Default models for each provider
PROVIDER_DEFAULTS: dict[str, ProviderConfig] = {
    "donkit": {
        "embedder_model": "default",
        "generation_model": "default",
    },
    "openai": {
        "embedder_model": "text-embedding-3-small",
        "generation_model": "gpt-4.1-mini",
    },
    "vertex": {
        "embedder_model": "text-multilingual-embedding-002",
        "generation_model": "gemini-2.5-flash",
    },
    "azure_openai": {
        "embedder_model": "text-embedding-ada-002",
        "generation_model": "gpt-4.1-mini",
    },
    "ollama": {
        "embedder_model": "embeddinggemma",
        "generation_model": "llama3.2",
    },
    "anthropic": {
        "embedder_model": None,
        "generation_model": "claude-4-0-sonnet",
    },
    "openrouter": {
        "embedder_model": None,
        "generation_model": "gpt-4.1-mini",
    },
}


def check_provider_credentials(provider: str, env_path: Path | None = None) -> bool:
    """
    Check if a provider has credentials configured in .env file.

    Args:
        provider: Provider name (e.g., "vertex", "openai")
        env_path: Optional path to .env file (defaults to current directory)

    Returns:
        True if credentials are configured, False otherwise
    """
    env_path = env_path or Path.cwd() / ".env"

    if not env_path.exists():
        return False

    try:
        config = dotenv_values(env_path)
    except Exception:
        return False

    if provider == "vertex":
        creds_path = config.get("RAGOPS_VERTEX_CREDENTIALS")
        if not creds_path:
            return False
        # Expand user path and check if file exists
        creds_path = os.path.expanduser(creds_path)
        return Path(creds_path).exists()

    elif provider == "openai":
        return bool(config.get("RAGOPS_OPENAI_API_KEY"))

    elif provider == "azure_openai":
        required = [
            "RAGOPS_AZURE_OPENAI_API_KEY",
            "RAGOPS_AZURE_OPENAI_ENDPOINT",
            "RAGOPS_AZURE_OPENAI_DEPLOYMENT",
        ]
        return all(config.get(k) for k in required)

    elif provider == "anthropic":
        return bool(config.get("RAGOPS_ANTHROPIC_API_KEY"))

    elif provider == "ollama":
        # Ollama requires base_url configuration (usually localhost:11434)
        base_url = config.get("RAGOPS_OLLAMA_BASE_URL", "")
        return bool(base_url)

    elif provider == "openrouter":
        # OpenRouter has its own API key, different from OpenAI
        # Check if custom base URL is set to OpenRouter
        base_url = config.get("RAGOPS_OPENAI_BASE_URL", "")
        is_openrouter_url = "openrouter.ai" in base_url
        has_api_key = bool(config.get("RAGOPS_OPENAI_API_KEY"))
        return has_api_key and is_openrouter_url

    elif provider == "donkit":
        base_url = config.get("RAGOPS_DONKIT_BASE_URL", "")
        # is_donkit_url = "donkit.ai" in base_url
        has_api_key = bool(config.get("RAGOPS_DONKIT_API_KEY"))
        return has_api_key  # and is_donkit_url

    return False


def get_available_providers(env_path: Path | None = None) -> dict[str, bool]:
    """
    Check which providers have credentials configured.

    Args:
        env_path: Optional path to .env file (defaults to current directory)

    Returns:
        Dictionary mapping provider names to availability status
    """
    providers = ["donkit", "openai", "vertex", "azure_openai", "ollama", "anthropic", "openrouter"]
    return {provider: check_provider_credentials(provider, env_path) for provider in providers}


def get_best_provider(available_providers: dict[str, bool], purpose: str = "generation") -> str:
    """
    Select the best available provider based on priority.

    Args:
        available_providers: Dictionary of provider availability
        purpose: Either "generation" or "embeddings"

    Returns:
        Best available provider name
    """
    if purpose == "embeddings":
        priority = ["donkit", "openai", "vertex", "azure_openai", "ollama"]
    else:
        priority = ["donkit", "openai", "vertex", "azure_openai", "openrouter", "ollama"]

    for provider in priority:
        if available_providers.get(provider, False):
            return provider

    # Fallback to openai (most common)
    return "openai"


def get_recommended_config(env_path: Path | None = None) -> dict[str, str]:
    """
    Get recommended configuration based on available providers.

    Args:
        env_path: Optional path to .env file (defaults to current directory)

    Returns:
        Dictionary with recommended embedder_provider, embedder_model,
        generation_provider, and generation_model
    """
    available = get_available_providers(env_path)

    embedder_provider = get_best_provider(available, "embeddings")
    generation_provider = get_best_provider(available, "generation")

    embedder_config = PROVIDER_DEFAULTS.get(embedder_provider, PROVIDER_DEFAULTS["openai"])
    generation_config = PROVIDER_DEFAULTS.get(generation_provider, PROVIDER_DEFAULTS["openai"])

    return {
        "embedder_provider": embedder_provider,
        "embedder_model": embedder_config["embedder_model"],
        "generation_provider": generation_provider,
        "generation_model": generation_config["generation_model"],
    }
