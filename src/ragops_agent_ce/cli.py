from __future__ import annotations

import os
import shlex
import time

import typer
from loguru import logger
from rich.console import Console
from rich.markup import escape

try:
    import readline

    # Advanced readline configuration for better UX
    readline.parse_and_bind("tab: complete")
    readline.parse_and_bind("set editing-mode emacs")
    readline.parse_and_bind("set completion-ignore-case on")
    readline.parse_and_bind("set show-all-if-ambiguous on")
    readline.parse_and_bind("set menu-complete-display-prefix on")

    # History management
    history_file = os.path.expanduser("~/.ragops_history")
    try:
        readline.read_history_file(history_file)
        readline.set_history_length(1000)
    except Exception:
        logger.warning("Failed to load readline history")
        pass

    # Save history on exit
    import atexit

    atexit.register(readline.write_history_file, history_file)

    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False

from . import __version__
from .agent.agent import LLMAgent, default_tools
from .agent.prompts import OPENAI_SYSTEM_PROMPT, VERTEX_SYSTEM_PROMPT
from .checklist_manager import ChecklistWatcherWithRenderer, get_active_checklist_text
from .config import load_settings
from .db import close, kv_all_by_prefix, open_db
from .display import ScreenRenderer
from .interactive_input import get_user_input
from .llm.provider_factory import get_provider
from .llm.types import Message
from .logging_config import setup_logging
from .mcp.client import MCPClient
from .prints import RAGOPS_LOGO_ART, RAGOPS_LOGO_TEXT
from .setup_wizard import run_setup_if_needed

app = typer.Typer(
    pretty_exceptions_enable=False,
)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"ragops-agent-ce {__version__}")
        raise typer.Exit(code=0)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    setup: bool = typer.Option(
        False,
        "--setup",
        help="Run setup wizard to configure the agent",
    ),
    system: str | None = typer.Option(
        None, "--system", "-s", help="System prompt to guide the agent"
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="LLM model to use (overrides settings)"
    ),
    provider: str | None = typer.Option(
        None, "--provider", "-p", help="LLM provider to use (overrides .env settings)"
    ),
    show_checklist: bool = typer.Option(
        True,
        "--show-checklist/--no-checklist",
        help="Render checklist panel at start and after each step",
    ),
) -> None:
    """RagOps Agent CE - LLM-powered CLI agent for building RAG pipelines."""
    # Setup logging according to .env / settings
    try:
        setup_logging(load_settings())
    except Exception:
        # Don't break CLI if logging setup fails
        pass

    # If no subcommand is provided, run the REPL
    if ctx.invoked_subcommand is None:
        # Run setup wizard if needed or forced
        if not run_setup_if_needed(force=setup):
            raise typer.Exit(code=1)

        # If --setup flag was used, exit after setup
        if setup:
            raise typer.Exit(code=0)

        _start_repl(
            system=system or VERTEX_SYSTEM_PROMPT
            if provider == "vertexai"
            else OPENAI_SYSTEM_PROMPT,
            model=model,
            provider=provider,
            mcp_commands=DEFAULT_MCP_COMMANDS,
            mcp_only=False,
            show_checklist=show_checklist,
        )


@app.command()
def ping() -> None:
    """Simple health command to verify the CLI is working."""
    console.print("pong")


@app.command()
def grep(
    pattern: str = typer.Argument(..., help="The regex pattern to search for."),
    include: str = typer.Option(
        "",
        "--include",
        "-i",
        help="File pattern to include in the search (e.g., '*.py', '*.{ts,tsx}').",
    ),
    path: str = typer.Option(
        ".",
        "--path",
        "-p",
        help="The directory to search in. Defaults to current working directory.",
    ),
) -> None:
    """Searches file contents using regular expressions with ripgrep (rg)."""
    from .agent.tools import tool_grep

    tool = tool_grep()
    args = {"pattern": pattern, "include": include, "path": path}
    result = tool.handler(args)
    console.print(result)


DEFAULT_MCP_COMMANDS = [
    "ragops-compose-manager",
    "ragops-rag-planner",
    "ragops-read-engine",
    "ragops-chunker",
    "ragops-vectorstore-loader",
    "ragops-checklist",
    "ragops-rag-query",
]


def _list_existing_projects() -> list[dict]:
    """Get list of existing projects from database."""
    import json

    db = open_db()
    try:
        all_projects_raw = kv_all_by_prefix(db, "project_")
        projects = [json.loads(value) for _, value in all_projects_raw]
        return projects
    finally:
        close(db)


def _format_projects_for_transcript(projects: list[dict]) -> list[str]:
    """Format projects as transcript lines."""
    lines = []

    if not projects:
        lines.append("[dim]No existing projects found. Start a new one![/dim]")
        return lines

    lines.append("[bold cyan]Existing Projects:[/bold cyan]")
    for i, project in enumerate(projects, 1):
        project_id = project.get("project_id", "unknown")
        goal = project.get("goal", "No goal set")
        status = project.get("status", "unknown")

        # Truncate long goals
        if len(goal) > 60:
            goal = goal[:57] + "..."

        status_color = (
            "green" if status == "completed" else "yellow" if status == "in_progress" else "white"
        )
        lines.append(
            f"  {i}. [bold]{project_id}[/bold] - {goal} [{status_color}]({status})[/{status_color}]"
        )

    lines.append("")
    lines.append("[dim]You can continue any project by mentioning its ID in your message.[/dim]")
    return lines


def _start_repl(
    *,
    system: str | None,
    model: str | None,
    provider: str | None,
    mcp_commands: list[str] | None,
    mcp_only: bool,
    show_checklist: bool,
) -> None:
    console.print(RAGOPS_LOGO_TEXT)
    console.print(RAGOPS_LOGO_ART)

    settings = load_settings()
    if provider:
        os.environ.setdefault("RAGOPS_LLM_PROVIDER", provider)
        settings = settings.model_copy(update={"llm_provider": provider})
    prov = get_provider(settings)

    tools = [] if mcp_only else default_tools()
    mcp_clients = []
    commands = mcp_commands if mcp_commands is not None else []
    if commands:
        for cmd_str in commands:
            cmd_parts = shlex.split(cmd_str)
            logger.debug(f"Starting MCP client: {cmd_parts}")
            mcp_clients.append(MCPClient(cmd_parts[0], cmd_parts[1:]))

    agent = LLMAgent(prov, tools=tools, mcp_clients=mcp_clients)

    session_started_at = time.time()

    history: list[Message] = []
    if system:
        history.append(Message(role="system", content=system))

    renderer = ScreenRenderer()

    # Sanitize transcript from any legacy checklist lines (we now render checklist separately)
    def _sanitize_transcript(trans: list[str]) -> None:
        markers = {
            "[dim]--- Checklist Created ---[/dim]",
        }
        # Remove any lines that exactly match known markers or start with the checklist header
        i = 0
        while i < len(trans):
            line = trans[i].strip()
            if line in markers or line.startswith("[white on blue]"):  # checklist header style
                trans.pop(i)
                # Do not increment i, continue checking at same index after pop
                continue
            i += 1

    def _get_session_checklist() -> str | None:
        return get_active_checklist_text(session_started_at)

    transcript: list[str] = []
    renderer.render_startup_screen()

    transcript.append(
        "Current provider: "
        + f"{settings.llm_provider}"
        + (f", model override: {model}" if model else "")
    )
    transcript.append("[enhanced input enabled]" if READLINE_AVAILABLE else "[basic input mode]")
    transcript.append(
        "[bold green] RagOps Agent> [/bold green]Hello! "
        "I'm Donkit - RagOps Agent, your assistant for building RAG pipelines. "
        "How can I help you today?"
    )
    watcher = None
    if show_checklist:
        watcher = ChecklistWatcherWithRenderer(
            transcript,
            renderer,
            session_start_mtime=session_started_at,
        )
        watcher.start()

    while True:
        try:
            cl_text = _get_session_checklist()
            if cl_text:
                renderer.render_conversation_and_checklist(
                    transcript, cl_text, show_input_space=True
                )
            else:
                renderer.render_conversation_screen(transcript, show_input_space=True)
            user_input = get_user_input()
        except (EOFError, KeyboardInterrupt):
            transcript.append("[Exiting REPL]")
            if watcher:
                watcher.stop()
            _sanitize_transcript(transcript)
            cl_text = _get_session_checklist()
            if cl_text:
                renderer.render_conversation_and_checklist(transcript, cl_text)
            else:
                renderer.render_conversation_screen(transcript)
            renderer.render_goodbye_screen()
            break

        if not user_input:
            continue

        if user_input in {":q", ":quit", ":exit", "exit", "quit"}:
            transcript.append("[Bye]")
            if watcher:
                watcher.stop()
            _sanitize_transcript(transcript)
            cl_text = _get_session_checklist()
            if cl_text:
                renderer.render_conversation_and_checklist(transcript, cl_text)
            else:
                renderer.render_conversation_screen(transcript)
            renderer.render_goodbye_screen()
            break

        transcript.append(f"[bold blue]you>[/bold blue] {escape(user_input)}")
        _sanitize_transcript(transcript)
        cl_text = _get_session_checklist()
        if cl_text:
            renderer.render_conversation_and_checklist(transcript, cl_text)
        else:
            renderer.render_conversation_screen(transcript)

        try:
            history.append(Message(role="user", content=user_input))
            reply = agent.respond(history, model=model)
            history.append(Message(role="assistant", content=reply))
            transcript.append(f"[bold green]RagOps Agent>[/bold green] {reply}")
        except Exception as e:
            transcript.append(f"[bold red]Error:[/bold red] {str(e)}")
