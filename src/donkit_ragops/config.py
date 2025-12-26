from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for RAGOps Agent CE."""

    db_path: str = Field(default="./ragops-agent-ce/ragops.db")

    # LLM configuration
    llm_provider: str = Field(default="mock")
    llm_model: str | None = Field(default=None)

    # API keys and endpoints
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_api_key: str | None = Field(default=None)
    openai_organization: str | None = Field(default=None)
    # Azure OpenAI
    azure_openai_api_key: str | None = Field(default=None)
    azure_openai_endpoint: str | None = Field(default=None)
    azure_openai_api_version: str = Field(default="2024-02-15-preview")
    azure_openai_deployment: str | None = Field(default=None)
    azure_openai_embeddings_deployment: str | None = Field(default=None)
    # Donkit
    donkit_api_key: str | None = Field(default=None)
    donkit_base_url: str = Field(default="http://localhost:8080")  # https://api.donkit.ai
    # Other providers
    anthropic_api_key: str | None = Field(default=None)
    ollama_base_url: str = Field(default="http://localhost:11434")
    # Vertex AI
    vertex_credentials: str = Field(default="./vertex_service_account.json")

    # MCP (Model Context Protocol) client settings
    # Comma-separated list of commands to launch MCP servers (e.g., "ragops-planner,ragops-chunker")
    mcp_commands: str | None = Field(default=None)
    # Optional comma-separated allowlist of tool names to expose from MCP
    mcp_allowlist: str | None = Field(default=None)

    # Logging
    log_level: str = Field(default="ERROR")

    # pydantic-settings v2 configuration
    model_config = SettingsConfigDict(env_prefix="RAGOPS_", case_sensitive=False)


def load_settings() -> Settings:
    # Load environment variables from .env files.
    # Priority order per file location (earlier has higher priority):
    # 1) Project root alongside pyproject.toml (derived from this file path)
    #    - .env.local then .env
    # 2) Current working directory traversal (find_dotenv)

    # Prefer loading from project root (works even if CWD is different)
    try:
        _pkg_dir = Path(__file__).resolve().parent  # .../src/donkit_ragops
        _src_dir = _pkg_dir.parent  # .../src
        _project_root = _src_dir.parent  # .../ragops-agent-ce
        for _fname in (".env.local", ".env"):
            _p = _project_root / _fname
            if _p.exists():
                load_dotenv(_p, override=True)
    except Exception:
        pass

    # Additionally, load from CWD-based search as a fallback
    for _fname in (".env.local", ".env"):
        _path = find_dotenv(filename=_fname, usecwd=True)
        if _path:
            load_dotenv(_path, override=True)

    return Settings()
