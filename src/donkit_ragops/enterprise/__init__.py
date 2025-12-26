"""Enterprise mode for Donkit RAGOps CE.

This module provides enterprise functionality including:
- WebSocketClient: Connect to API Gateway via WebSocket (server-side agent)
- Token management: Keyring-based secure token storage
- Enterprise configuration
- File upload: Upload files to cloud via presigned URLs
- File analyzer: Extract metadata from files before upload

Enterprise mode connects to the Donkit cloud infrastructure. The LLM agent
runs on the server side (not locally). History is saved automatically on
the server.
"""

# Lazy imports to avoid circular dependencies and optional dependency issues
# Import these directly from submodules:
#   from donkit_ragops.enterprise.auth import get_token, save_token, delete_token
#   from donkit_ragops.enterprise.config import EnterpriseSettings
#   from donkit_ragops.enterprise.ws_client import WebSocketClient
#   from donkit_ragops.enterprise.upload import FileUploader, upload_files_to_cloud
#   from donkit_ragops.enterprise.analyzer import FileAnalyzer

__all__ = [
    "WebSocketClient",
    "EnterpriseSettings",
    "get_token",
    "save_token",
    "delete_token",
    "FileUploader",
    "upload_files_to_cloud",
    "FileAnalyzer",
]


def __getattr__(name: str):
    """Lazy import for enterprise components."""
    if name in ("get_token", "save_token", "delete_token", "has_token", "TokenService"):
        from donkit_ragops.enterprise import auth

        return getattr(auth, name)
    if name in ("EnterpriseSettings", "load_enterprise_settings"):
        from donkit_ragops.enterprise import config

        return getattr(config, name)
    if name == "WebSocketClient":
        from donkit_ragops.enterprise import ws_client

        return ws_client.WebSocketClient
    if name in ("FileUploader", "upload_files_to_cloud"):
        from donkit_ragops.enterprise import upload

        return getattr(upload, name)
    if name == "FileAnalyzer":
        from donkit_ragops.enterprise import analyzer

        return analyzer.FileAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
