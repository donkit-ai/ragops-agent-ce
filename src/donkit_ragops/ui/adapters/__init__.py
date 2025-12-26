"""UI adapters for different frameworks.

- RichUI: Full-featured Rich console implementation
- PlainUI: Simple text output for testing and pipes
"""

from __future__ import annotations

from donkit_ragops.ui.adapters.plain_adapter import PlainUI
from donkit_ragops.ui.adapters.rich_adapter import RichUI

__all__ = ["RichUI", "PlainUI"]
