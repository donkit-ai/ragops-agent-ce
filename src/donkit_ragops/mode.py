"""Mode detection for Donkit RAGOps CE.

Determines whether to run in local or enterprise mode based on:
1. Explicit flags (--local / --enterprise)
2. Presence of enterprise token in keyring
"""

from __future__ import annotations

from enum import StrEnum


class Mode(StrEnum):
    """Operating mode for the CLI."""

    LOCAL = "local"
    ENTERPRISE = "enterprise"


def detect_mode(
    force_local: bool = False,
    force_enterprise: bool = False,
) -> Mode:
    """Detect the operating mode.

    Args:
        force_local: If True, always use local mode
        force_enterprise: If True, always use enterprise mode

    Returns:
        The detected mode (LOCAL or ENTERPRISE)

    Raises:
        ValueError: If both force_local and force_enterprise are True
    """
    if force_local and force_enterprise:
        raise ValueError("Cannot force both local and enterprise mode")

    if force_local:
        return Mode.LOCAL

    if force_enterprise:
        return Mode.ENTERPRISE

    # Auto-detect based on token presence
    try:
        from donkit_ragops.enterprise.auth import get_token

        if get_token():
            return Mode.ENTERPRISE
    except ImportError:
        # Enterprise module not available
        pass

    return Mode.LOCAL
