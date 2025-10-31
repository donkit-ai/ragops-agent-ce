from __future__ import annotations

import importlib
import json

from ..config import Settings
from ..config import load_settings
from .base import LLMProvider

PROVIDER_PATHS: dict[str, tuple[str, str]] = {
    "openai": ("ragops_agent_ce.llm.providers.openai", "OpenAIProvider"),
    "azure_openai": ("ragops_agent_ce.llm.providers.azure_openai", "AzureOpenAIProvider"),
    "anthropic": ("ragops_agent_ce.llm.providers.anthropic", "AnthropicProvider"),
    "ollama": ("ragops_agent_ce.llm.providers.openai", "OpenAIProvider"),
    "openrouter": ("ragops_agent_ce.llm.providers.openai", "OpenAIProvider"),
    "mock": ("ragops_agent_ce.llm.providers.mock", "MockProvider"),
    "vertex": ("ragops_agent_ce.llm.providers.vertex", "VertexProvider"),
}


def __get_vertex_credentials():
    credentials_path = load_settings().vertex_credentials
    with open(credentials_path) as f:
        credentials_data = json.load(f)
    return credentials_data


def get_provider(settings: Settings | None = None, llm_provider: str | None = None) -> LLMProvider:
    cfg = settings or load_settings()
    provider_key = (llm_provider or cfg.llm_provider or "mock").lower()
    path = PROVIDER_PATHS.get(provider_key)
    if not path:
        raise ValueError(f"Unknown LLM provider: {provider_key}")
    module_name, class_name = path
    module = importlib.import_module(module_name)
    cls: type[LLMProvider] = getattr(module, class_name)

    if provider_key == "vertex":
        credentials_data = __get_vertex_credentials()
        return cls(cfg, credentials_data=credentials_data)
    elif provider_key == "openrouter":
        # OpenRouter uses OpenAI-compatible API with custom base_url
        # Create a modified config with OpenRouter endpoint
        openrouter_cfg = cfg.model_copy(update={"openai_base_url": "https://openrouter.ai/api/v1"})
        return cls(openrouter_cfg)
    elif provider_key == "ollama":
        # Ollama uses OpenAI-compatible API with custom base_url
        # Use ollama_base_url from config, API key not required
        ollama_cfg = cfg.model_copy(
            update={
                "openai_base_url": f"{cfg.ollama_base_url}/v1",
                "openai_api_key": "ollama",  # Ollama doesn't require real API key
            }
        )
        return cls(ollama_cfg)

    return cls(cfg)
