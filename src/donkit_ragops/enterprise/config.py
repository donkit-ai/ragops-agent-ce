"""Enterprise-specific settings.

Note: API token is NOT stored here - it's managed via keyring in auth.py
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnterpriseSettings(BaseSettings):
    """Enterprise mode configuration."""

    # API Gateway URL (Cloud MCP Server)
    api_url: str = Field(
        default="https://api.donkit.ai",
        description="Donkit API Gateway URL",
    )

    # MCP endpoint path
    mcp_path: str = Field(
        default="/mcp",
        description="MCP endpoint path on API Gateway",
    )

    # Connection timeout
    timeout: int = Field(
        default=60,
        description="Connection timeout in seconds",
    )

    model_config = SettingsConfigDict(
        env_prefix="DONKIT_ENTERPRISE_",
        case_sensitive=False,
    )

    @property
    def mcp_url(self) -> str:
        """Full URL for MCP endpoint."""
        return f"{self.api_url.rstrip('/')}{self.mcp_path}"


def load_enterprise_settings() -> EnterpriseSettings:
    """Load enterprise settings from environment."""
    return EnterpriseSettings()
