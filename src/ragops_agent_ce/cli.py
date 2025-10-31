from __future__ import annotations

import os
import re
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


def _time_str() -> str:
    """Get current time string for transcript."""
    return "[dim]" + time.strftime("[%H:%M]", time.localtime()) + "[/]"


def _render_markdown_to_rich(text: str) -> str:
    """Convert markdown text to simple Rich markup without breaking formatting."""
    # Simple markdown to rich markup conversion without full rendering
    # This preserves the transcript panel formatting

    # Bold: **text** -> [bold]text[/bold]
    result = re.sub(r"\*\*(.+?)\*\*", r"[bold]\1[/bold]", text)

    # Italic: *text* -> [italic]text[/italic] (but don't match list items)
    result = re.sub(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)", r"[italic]\1[/italic]", result)

    # Inline code: `text` -> [cyan]text[/cyan]
    result = re.sub(r"`(.+?)`", r"[cyan]\1[/cyan]", result)

    # Headers: ## text -> [bold cyan]text[/bold cyan]
    result = re.sub(r"^#+\s+(.+)$", r"[bold cyan]\1[/bold cyan]", result, flags=re.MULTILINE)

    # List items: - text or * text -> • text (with proper indentation preserved)
    result = re.sub(r"^(\s*)[*-]\s+", r"\1• ", result, flags=re.MULTILINE)

    # Numbered lists: 1. text -> 1. text (keep as is)

    return result


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

    # Create agent
    agent = LLMAgent(
        prov,
        tools=tools,
        mcp_clients=mcp_clients,
    )
    renderer.render_startup_screen()

    transcript.append(
        "Current provider: "
        + f"{settings.llm_provider}"
        + (f", model override: {model}" if model else "")
    )
    transcript.append("[enhanced input enabled]" if READLINE_AVAILABLE else "[basic input mode]")

    # Render welcome message as markdown
    welcome_msg = (
        "Hello! I'm **Donkit - RagOps Agent**, your assistant for building RAG pipelines. "
        "How can I help you today?"
    )
    rendered_welcome = _render_markdown_to_rich(welcome_msg)
    transcript.append(f"{_time_str()} [bold green]RagOps Agent>[/bold green] {rendered_welcome}")
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

        transcript.append(f"{_time_str()} [bold blue]you>[/bold blue] {escape(user_input)}")
        _sanitize_transcript(transcript)
        cl_text = _get_session_checklist()
        if cl_text:
            renderer.render_conversation_and_checklist(transcript, cl_text)
        else:
            renderer.render_conversation_screen(transcript)

        try:
            history.append(Message(role="user", content=user_input))
            # Use streaming if provider supports it
            if prov.supports_streaming():
                reply = ""
                interrupted = False

                # Add placeholder to transcript
                transcript.append(f"{_time_str()} [bold green]RagOps Agent>[/bold green] ")
                response_index = len(transcript) - 1

                try:
                    # Accumulate everything in display order
                    display_content = ""
                    # Temporary message shown only during execution
                    temp_executing = ""

                    # Stream events - process content and tool calls
                    for event in agent.respond_stream(history, model=model):
                        if event.type == "content":
                            # Accumulate content for history
                            reply += event.content or ""
                            # Add content to display only if not already there
                            content_chunk = event.content or ""
                            if content_chunk and content_chunk not in display_content:
                                display_content += content_chunk
                        elif event.type == "tool_call_start":
                            # Show temporary executing message (not added to display_content)
                            args_str = ", ".join(event.tool_args.keys()) if event.tool_args else ""
                            temp_executing = (
                                f"\n\n[dim]🔧 Executing tool:[/dim] "
                                f"[yellow]{event.tool_name}[/yellow]({args_str})"
                            )
                        elif event.type == "tool_call_end":
                            # Clear temp message and add success to permanent content
                            temp_executing = ""
                            msg = f"\n[dim]✓ Tool:[/dim] [green]{event.tool_name}[/green]\n"
                            display_content += msg
                        elif event.type == "tool_call_error":
                            # Clear temp message and add error to permanent content
                            temp_executing = ""
                            msg = (
                                f"\n[dim]✗ Tool failed:[/dim] "
                                f"\[red]{event.tool_name}[/red] - {event.error}\n"
                            )
                            display_content += msg

                        # Update transcript with permanent content + temporary executing message
                        transcript[response_index] = (
                            f"{_time_str()} [bold green]RagOps Agent>[/bold green] "
                            f"{display_content}{temp_executing}"
                        )

                        # Re-render screen after each event
                        cl_text = _get_session_checklist()
                        if cl_text:
                            renderer.render_conversation_and_checklist(
                                transcript, cl_text, show_input_space=False
                            )
                        else:
                            renderer.render_conversation_screen(transcript, show_input_space=False)
                except KeyboardInterrupt:
                    interrupted = True
                    transcript[response_index] = (
                        f"{_time_str()} [yellow]⚠ Generation interrupted by user[/yellow]"
                    )

                # Add to history if we got a response
                if reply and not interrupted:
                    history.append(Message(role="assistant", content=reply))
            else:
                # Fall back to non-streaming mode
                # Add placeholder
                transcript.append(f"{_time_str()} [bold green]RagOps Agent>[/bold green] ")
                response_index = len(transcript) - 1

                try:
                    reply = agent.respond(history, model=model)
                    history.append(Message(role="assistant", content=reply))

                    # Replace placeholder with actual response
                    rendered_reply = _render_markdown_to_rich(reply)
                    transcript[response_index] = (
                        f"{_time_str()} [bold green]RagOps Agent>[/bold green] {rendered_reply}"
                    )
                except KeyboardInterrupt:
                    console.print("\n[yellow]⚠ Generation interrupted by user[/yellow]")
                    transcript[response_index] = (
                        f"{_time_str()} [yellow]Generation interrupted[/yellow]"
                    )
        except Exception as e:
            transcript.append(f"{_time_str()} [bold red]Error:[/bold red] {str(e)}")
