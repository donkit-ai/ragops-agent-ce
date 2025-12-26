"""Base abstractions for REPL implementations.

Follows Dependency Inversion Principle - depends on abstractions, not concretions.
Uses ABC (Abstract Base Class) for nominal typing.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from donkit.llm import LLMModelAbstract, Message

from donkit_ragops.ui import UIAdapter, get_ui, set_ui_adapter
from donkit_ragops.ui.protocol import UI
from donkit_ragops.ui.styles import StyleName

if TYPE_CHECKING:
    from donkit_ragops.agent.agent import LLMAgent
    from donkit_ragops.display import ScreenRenderer
    from donkit_ragops.mcp.client import MCPClient
    from donkit_ragops.repl_helpers import MCPEventHandler, ReplRenderHelper
    from donkit_ragops.schemas.agent_schemas import AgentSettings


@dataclass
class ReplContext:
    """Context for REPL with all dependencies.

    Implements manual dependency injection through constructor.
    """

    # UI interface - single point for all user interaction
    set_ui_adapter(UIAdapter.RICH)
    ui: UI = get_ui()  # field(default_factory=get_ui)

    # Core state
    history: list[Message] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)

    # Settings
    agent_settings: AgentSettings | None = None
    session_started_at: float = field(default_factory=time.time)
    show_checklist: bool = True

    # Provider and agent
    provider: LLMModelAbstract | None = None
    provider_name: str | None = None
    model: str | None = None
    agent: LLMAgent | None = None

    # MCP clients
    mcp_clients: list[MCPClient] = field(default_factory=list)

    # Rendering
    renderer: ScreenRenderer | None = None
    render_helper: ReplRenderHelper | None = None
    mcp_handler: MCPEventHandler | None = None

    # System prompt
    system_prompt: str | None = None


class BaseREPL(ABC):
    """Base class for REPL implementations.

    Uses ABC for strict inheritance contract.
    Subclasses must implement all abstract methods.
    """

    def __init__(self, context: ReplContext) -> None:
        """Initialize REPL with context.

        Args:
            context: REPL context with all dependencies
        """
        self.context = context
        self._running = True

    @property
    def is_running(self) -> bool:
        """Check if REPL is still running."""
        return self._running

    def stop(self) -> None:
        """Signal REPL to stop."""
        self._running = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize REPL resources.

        Called before the main loop starts.
        Should set up agents, connections, etc.
        """
        ...

    @abstractmethod
    async def run(self) -> None:
        """Run the REPL main loop.

        This is the entry point for starting the REPL.
        """
        ...

    @abstractmethod
    async def handle_input(self, user_input: str) -> bool:
        """Handle user input.

        Args:
            user_input: Raw user input string

        Returns:
            False if REPL should exit, True to continue
        """
        ...

    @abstractmethod
    async def handle_message(self, message: str) -> None:
        """Handle a chat message (non-command input).

        Args:
            message: User's chat message
        """
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up REPL resources.

        Called when REPL is shutting down.
        Should close connections, save state, etc.
        """
        ...

    def print_welcome(self) -> None:
        """Print welcome message. Can be overridden by subclasses."""
        self.context.ui.print("Type your message and press Enter to start...", StyleName.DIM)
        self.context.ui.print("Commands: :help, :q, :clear, :provider, :model", StyleName.DIM)
        self.context.ui.newline()
