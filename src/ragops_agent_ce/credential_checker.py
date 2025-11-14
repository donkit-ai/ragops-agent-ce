"""
Shared credential checking utilities.

Provides unified logic for checking provider credentials across the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values


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

    return False
