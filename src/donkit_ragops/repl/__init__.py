"""REPL module for RAGOps Agent CE.

Provides REPL implementations for local and enterprise modes.
"""

from donkit_ragops.repl.base import BaseREPL, ReplContext
from donkit_ragops.repl.commands import CommandRegistry, CommandResult, ReplCommand
from donkit_ragops.repl.enterprise_repl import EnterpriseREPL, EnterpriseWsHandler
from donkit_ragops.repl.local_repl import LocalREPL

__all__ = [
    "BaseREPL",
    "ReplContext",
    "ReplCommand",
    "CommandResult",
    "CommandRegistry",
    "LocalREPL",
    "EnterpriseREPL",
    "EnterpriseWsHandler",
]
