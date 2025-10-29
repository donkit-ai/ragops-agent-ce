from __future__ import annotations

import importlib
import json

from ..config import Settings, load_settings
from .base import LLMProvider

PROVIDER_PATHS: dict[str, tuple[str, str]] = {
    "openai": ("ragops_agent_ce.llm.providers.openai", "OpenAIProvider"),
    "azure_openai": ("ragops_agent_ce.llm.providers.azure_openai", "AzureOpenAIProvider"),
    "anthropic": ("ragops_agent_ce.llm.providers.anthropic", "AnthropicProvider"),
    "ollama": ("ragops_agent_ce.llm.providers.ollama", "OllamaProvider"),
    "mock": ("ragops_agent_ce.llm.providers.mock", "MockProvider"),
    "vertex": ("ragops_agent_ce.llm.providers.vertex", "VertexProvider"),
}


def __get_vertex_credentials():
    credentials_path = load_settings().vertex_credentials
    with open(credentials_path) as f:
        credentials_data = json.load(f)
    return credentials_data


def get_provider(settings: Settings | None = None) -> LLMProvider:
    cfg = settings or load_settings()
    provider_key = (cfg.llm_provider or "mock").lower()
    path = PROVIDER_PATHS.get(provider_key)
    if not path:
        raise ValueError(f"Unknown LLM provider: {provider_key}")
    module_name, class_name = path
    module = importlib.import_module(module_name)
    cls: type[LLMProvider] = getattr(module, class_name)
    if provider_key == "vertex":
        credentials_data = __get_vertex_credentials()
        return cls(cfg, credentials_data=credentials_data)
    return cls(cfg)
