"""Enterprise mode for Donkit RAGOps CE.

This module provides enterprise functionality including:
- CloudMCPClient: Native FastMCP client for connecting to cloud MCP server
- Token management: Keyring-based secure token storage
- Enterprise configuration

Enterprise mode connects to the Donkit cloud infrastructure while keeping
the LLM agent running locally.
"""

# Lazy imports to avoid circular dependencies and optional dependency issues
# Import these directly from submodules:
#   from ragops_agent_ce.enterprise.auth import get_token, save_token, delete_token
#   from ragops_agent_ce.enterprise.config import EnterpriseSettings
#   from ragops_agent_ce.enterprise.mcp_client import CloudMCPClient

__all__ = [
    "CloudMCPClient",
    "EnterpriseSettings",
    "get_token",
    "save_token",
    "delete_token",
]


def __getattr__(name: str):
    """Lazy import for enterprise components."""
    if name in ("get_token", "save_token", "delete_token", "has_token", "TokenService"):
        from ragops_agent_ce.enterprise import auth

        return getattr(auth, name)
    if name in ("EnterpriseSettings", "load_enterprise_settings"):
        from ragops_agent_ce.enterprise import config

        return getattr(config, name)
    if name == "CloudMCPClient":
        from ragops_agent_ce.enterprise import mcp_client

        return mcp_client.CloudMCPClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
