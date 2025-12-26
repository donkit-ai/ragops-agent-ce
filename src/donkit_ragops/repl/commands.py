"""REPL command system.

Follows Open/Closed Principle - commands can be added without modifying existing code.
Uses ABC for command interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from donkit_ragops import texts
from donkit_ragops.ui.styles import StyledText, StyleName, styled_text

if TYPE_CHECKING:
    from donkit_ragops.repl.base import ReplContext


@dataclass
class CommandResult:
    """Result of command execution."""

    should_continue: bool = True
    should_exit: bool = False
    messages: list[str] = field(default_factory=list)
    styled_messages: list[StyledText] = field(default_factory=list)


class ReplCommand(ABC):
    """Base class for REPL commands.

    Uses ABC for strict inheritance contract.
    New commands can be added by subclassing without modifying existing code.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Primary command name (without colon)."""
        ...

    @property
    @abstractmethod
    def aliases(self) -> list[str]:
        """Alternative names for the command."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description for help text."""
        ...

    def matches(self, input_str: str) -> bool:
        """Check if input matches this command."""
        normalized = input_str.strip().lower()
        all_names = [f":{self.name}"] + [f":{a}" for a in self.aliases]
        # Also match without colon for some commands
        all_names.extend([self.name] + self.aliases)
        return normalized in all_names

    @abstractmethod
    async def execute(self, context: ReplContext) -> CommandResult:
        """Execute the command.

        Args:
            context: REPL context with state and dependencies

        Returns:
            CommandResult indicating next action
        """
        ...


class HelpCommand(ReplCommand):
    """Show help message."""

    @property
    def name(self) -> str:
        return "help"

    @property
    def aliases(self) -> list[str]:
        return ["h", "?"]

    @property
    def description(self) -> str:
        return "Show available commands"

    async def execute(self, context: ReplContext) -> CommandResult:
        return CommandResult(styled_messages=texts.styled_help_commands())


class ClearCommand(ReplCommand):
    """Clear conversation history."""

    @property
    def name(self) -> str:
        return "clear"

    @property
    def aliases(self) -> list[str]:
        return []

    @property
    def description(self) -> str:
        return "Clear conversation history"

    async def execute(self, context: ReplContext) -> CommandResult:
        # Keep only system messages
        system_msgs = [m for m in context.history if m.role == "system"]
        context.history.clear()
        context.history.extend(system_msgs)
        context.transcript.clear()
        return CommandResult(styled_messages=[styled_text((StyleName.DIM, "History cleared"))])


class QuitCommand(ReplCommand):
    """Exit the REPL."""

    @property
    def name(self) -> str:
        return "quit"

    @property
    def aliases(self) -> list[str]:
        return ["q", "exit"]

    @property
    def description(self) -> str:
        return "Exit the application"

    def matches(self, input_str: str) -> bool:
        """Override to also match bare 'exit' and 'quit' without colon."""
        normalized = input_str.strip().lower()
        return normalized in {":quit", ":q", ":exit", "quit", "exit"}

    async def execute(self, context: ReplContext) -> CommandResult:
        context.transcript.append("[Bye]")
        return CommandResult(
            should_exit=True,
            styled_messages=[styled_text((StyleName.DIM, "Goodbye!"))],
        )


class CommandRegistry:
    """Registry for REPL commands.

    Implements Open/Closed Principle - new commands can be registered
    without modifying the registry itself.
    """

    def __init__(self) -> None:
        self._commands: list[ReplCommand] = []

    def register(self, command: ReplCommand) -> None:
        """Register a command.

        Args:
            command: Command instance to register
        """
        self._commands.append(command)

    def register_defaults(self) -> None:
        """Register default REPL commands."""
        self.register(HelpCommand())
        self.register(ClearCommand())
        self.register(QuitCommand())

    def get_command(self, input_str: str) -> ReplCommand | None:
        """Find command matching input.

        Args:
            input_str: User input string

        Returns:
            Matching command or None
        """
        for cmd in self._commands:
            if cmd.matches(input_str):
                return cmd
        return None

    def is_command(self, input_str: str) -> bool:
        """Check if input is a command."""
        return input_str.strip().startswith(":")

    def get_all_commands(self) -> list[ReplCommand]:
        """Get all registered commands."""
        return list(self._commands)


def create_default_registry() -> CommandRegistry:
    """Create registry with default commands."""
    registry = CommandRegistry()
    registry.register_defaults()
    return registry
