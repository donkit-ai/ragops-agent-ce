"""Token management for enterprise mode.

Uses keyring for secure token storage. Tokens are NEVER:
- Logged to console/files
- Passed to LLM prompts
- Stored in .env files
"""

from __future__ import annotations

import keyring
import keyring.errors

SERVICE_NAME = "donkit-ragops"
TOKEN_KEY = "api_token"


class TokenService:
    """Service for managing enterprise API tokens using keyring."""

    def __init__(self, service_name: str = SERVICE_NAME):
        self.service_name = service_name

    def save_token(self, token: str) -> None:
        """Save token to keyring.

        Args:
            token: The API token to save
        """
        keyring.set_password(self.service_name, TOKEN_KEY, token)

    def get_token(self) -> str | None:
        """Get token from keyring.

        Returns:
            The stored token, or None if not found
        """
        return keyring.get_password(self.service_name, TOKEN_KEY)

    def delete_token(self) -> None:
        """Delete token from keyring."""
        try:
            keyring.delete_password(self.service_name, TOKEN_KEY)
        except keyring.errors.PasswordDeleteError:
            pass

    def has_token(self) -> bool:
        """Check if a token exists.

        Returns:
            True if token exists, False otherwise
        """
        return self.get_token() is not None


# Module-level convenience functions using default service
_default_service = TokenService()


def save_token(token: str) -> None:
    """Save token to keyring."""
    _default_service.save_token(token)


def get_token() -> str | None:
    """Get token from keyring."""
    return _default_service.get_token()


def delete_token() -> None:
    """Delete token from keyring."""
    _default_service.delete_token()


def has_token() -> bool:
    """Check if a token exists."""
    return _default_service.has_token()
