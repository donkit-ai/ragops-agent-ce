from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import time
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.markup import escape

from ragops_agent_ce.schemas.agent_schemas import AgentSettings

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
from .agent.agent import LLMAgent
from .agent.agent import default_tools
from .agent.prompts import OPENAI_SYSTEM_PROMPT
from .agent.prompts import VERTEX_SYSTEM_PROMPT
from .checklist_manager import ChecklistWatcherWithRenderer
from .checklist_manager import active_checklist
from .checklist_manager import get_active_checklist_text
from .config import load_settings
from .display import ScreenRenderer
from .interactive_input import get_user_input
from .llm.provider_factory import get_provider
from .llm.types import Message
from .logging_config import setup_logging
from .mcp.client import MCPClient
from .model_selector import save_model_selection
from .model_selector import select_model_at_startup
from .prints import RAGOPS_LOGO_ART
from .prints import RAGOPS_LOGO_TEXT
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
    """RAGOps Agent CE - LLM-powered CLI agent for building RAG pipelines."""
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

        # Model selection at startup (mandatory)
        # Only skip if provider is explicitly provided via CLI flag
        if provider is None:
            model_selection = select_model_at_startup()
            if model_selection is None:
                console.print("[red]Model selection cancelled. Exiting.[/red]")
                raise typer.Exit(code=1)
            provider, model_from_selection = model_selection
            # Override model from selection if not provided via CLI
            if model is None:
                model = model_from_selection
        else:
            # Provider provided via CLI, save it to KV database as latest
            save_model_selection(provider, model)

        asyncio.run(
            _astart_repl(
                system=system or VERTEX_SYSTEM_PROMPT
                if provider == "vertex" or provider == "vertexai"
                else OPENAI_SYSTEM_PROMPT,
                model=model,
                provider=provider,
                mcp_commands=DEFAULT_MCP_COMMANDS,
                mcp_only=False,
                show_checklist=show_checklist,
            )
        )


@app.command()
def ping() -> None:
    """Simple health command to verify the CLI is working."""
    console.print("pong")


DEFAULT_MCP_COMMANDS = ["donkit-ragops-mcp"]


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


async def _astart_repl(
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

    # Try to get provider and validate credentials
    try:
        prov = get_provider(settings, llm_provider=provider)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        console.print(f"[red]Error initializing provider '{provider}':[/red] {e}")
        console.print("[yellow]Please ensure credentials are configured correctly.[/yellow]")
        raise typer.Exit(code=1)

    tools = [] if mcp_only else default_tools()

    session_started_at = time.time()

    history: list[Message] = []
    if system:
        history.append(Message(role="system", content=system))

    # Transcript buffer must be available for helper functions below
    transcript: list[str] = []
    agent_settings = AgentSettings(llm_provider=prov, model=model)
    renderer = ScreenRenderer()

    # Progress state for MCP tools
    progress_line_index: int | None = None

    def mcp_progress_callback(progress: float, total: float | None, message: str | None) -> None:
        """Callback for MCP tool progress updates."""
        nonlocal progress_line_index

        if total is not None:
            percentage = (progress / total) * 100
            progress_text = f"[dim]⏳ Progress: {percentage:.1f}% - {message or ''}[/dim]"
        else:
            progress_text = f"[dim]⏳ Progress: {progress} - {message or ''}[/dim]"

        if progress_line_index is None:
            # First progress update - add new line
            transcript.append(progress_text)
            progress_line_index = len(transcript) - 1
        else:
            # Update existing progress line
            transcript[progress_line_index] = progress_text

        # Re-render to show progress
        cl_text = _get_session_checklist() if show_checklist else None
        renderer.render_project(
            transcript, cl_text, agent_settings=agent_settings, show_input_space=False
        )

    # Create MCP clients with progress callback
    mcp_clients = []
    commands = mcp_commands if mcp_commands is not None else []
    if commands:
        for cmd_str in commands:
            cmd_parts = shlex.split(cmd_str)
            logger.debug(f"Starting MCP client: {cmd_parts}")
            mcp_clients.append(
                MCPClient(cmd_parts[0], cmd_parts[1:], progress_callback=mcp_progress_callback)
            )

    # Helpers for rendering and transcript updates
    def _render_current_screen(show_input_space: bool) -> None:
        cl_text = _get_session_checklist()
        renderer.render_project(
            transcript, cl_text, agent_settings=agent_settings, show_input_space=show_input_space
        )

    def _append_user_line(text: str) -> None:
        transcript.append(f"\n\n{_time_str()} [bold blue]you>[/bold blue] {escape(text)}")

    def _start_agent_placeholder() -> int:
        transcript.append(f"\n{_time_str()} [bold green]RAGOps Agent>[/bold green] ")
        return len(transcript) - 1

    def _set_agent_line(index: int, display_content: str, temp_executing: str) -> None:
        transcript[index] = (
            f"\n{_time_str()} [bold green]RAGOps Agent>[/bold green] {display_content}{temp_executing}"  # noqa
        )

    # Formatting helpers for tool execution messages
    def _tool_executing_message(tool_name: str, tool_args: dict | None) -> str:
        args_str = ", ".join(tool_args.keys()) if tool_args else ""
        return f"\n[dim]🔧 Executing tool:[/dim] [yellow]{escape(tool_name)}[/yellow]({args_str})"

    def _tool_done_message(tool_name: str) -> str:
        return f"\n[dim]✓ Tool:[/dim] [green]{escape(tool_name)}[/green]\n"

    def _tool_error_message(tool_name: str, error: str) -> str:
        return f"\n[dim]✗ Tool failed:[/dim] [red]{escape(tool_name)}[/red] - {escape(error)}\n"

    # Stream event handler: returns updated (reply, display_content, temp_executing)
    def _process_stream_event(
        event, reply: str, display_content: str, temp_executing: str
    ) -> tuple[str, str, str]:
        nonlocal progress_line_index
        et = getattr(event, "type", None)
        if et == "content":
            content_chunk = event.content or ""
            reply = reply + content_chunk
            display_content = display_content + content_chunk
            return reply, display_content, temp_executing
        if et == "tool_call_start":
            return reply, display_content, _tool_executing_message(event.tool_name, event.tool_args)
        if et == "tool_call_end":
            if getattr(event, "tool_name", None) == "get_checklist":
                # Refresh checklist after get_checklist tool
                # Safely parse tool result - it might be JSON or plain text error message
                try:
                    tool_result = history[-1].content or "{}"
                    checklist_content = json.loads(tool_result)
                    if checklist_content.get("name"):
                        active_checklist.name = checklist_content.get("name") + ".json"
                except (json.JSONDecodeError, ValueError):
                    # Tool returned plain text (e.g., "Checklist '...' not found."), not JSON
                    # Skip checklist update in this case - checklist doesn't exist yet
                    pass
                cl_text = _get_session_checklist()
                renderer.render_project(
                    transcript,
                    cl_text,
                    agent_settings=agent_settings,
                    show_input_space=False,
                )
            if getattr(event, "tool_name", None) in ("create_checklist", "update_checklist_item"):
                # Refresh checklist after create/update operations
                try:
                    tool_result = history[-1].content or "{}"
                    checklist_content = json.loads(tool_result)
                    if checklist_content.get("name"):
                        active_checklist.name = checklist_content.get("name") + ".json"
                except (json.JSONDecodeError, ValueError):
                    # Tool returned plain text error, skip checklist update
                    pass
                cl_text = _get_session_checklist()
                renderer.render_project(
                    transcript,
                    cl_text,
                    agent_settings=agent_settings,
                    show_input_space=False,
                )
            # Remove progress line and reset for next tool execution
            if progress_line_index is not None:
                transcript.pop(progress_line_index)
            progress_line_index = None
            return reply, display_content + _tool_done_message(event.tool_name), ""
        if et == "tool_call_error":
            # Remove progress line and reset for next tool execution
            if progress_line_index is not None:
                transcript.pop(progress_line_index)
            progress_line_index = None
            return (
                reply,
                display_content + _tool_error_message(event.tool_name, event.error or ""),
                "",
            )
        return reply, display_content, temp_executing

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

    # Create agent
    agent = LLMAgent(
        prov,
        tools=tools,
        mcp_clients=mcp_clients,
    )
    # Initialize MCP tools asynchronously
    await agent._ainit_mcp_tools()
    renderer.render_startup_screen()

    # Render welcome message as markdown
    welcome_msg = (
        "Hello! I'm **Donkit - RAGOps Agent**, your assistant for building RAG pipelines. "
        "How can I help you today?"
    )
    rendered_welcome = _render_markdown_to_rich(welcome_msg)
    transcript.append(f"{_time_str()} [bold green]RAGOps Agent>[/bold green] {rendered_welcome}")
    watcher = None
    if show_checklist:
        watcher = ChecklistWatcherWithRenderer(
            transcript,
            agent_settings,
            renderer,
            session_start_mtime=session_started_at,
        )
        watcher.start()

    while True:
        try:
            _render_current_screen(show_input_space=True)
            user_input = get_user_input()
        except (EOFError, KeyboardInterrupt):
            transcript.append("[Exiting REPL]")
            if watcher:
                watcher.stop()
            _sanitize_transcript(transcript)
            _render_current_screen(show_input_space=False)
            renderer.render_goodbye_screen()
            break

        if not user_input:
            continue

        if user_input == ":help":
            transcript += [
                "",
                "  [yellow]Available commands:[/yellow]",
                "  [bold]:help[/bold] - Show this help message",
                "  [bold]:q[/bold] or [bold]:quit[/bold] - Exit the agent",
                "  [bold]:clear[/bold] - Clear the conversation transcript",
                "  [bold]:provider[/bold] - Select LLM provider",
                "  [bold]:model[/bold] - Select LLM model",
            ]
            continue

        if user_input == ":clear":
            transcript = []
            continue

        if user_input == ":provider":
            # Select provider interactively
            _render_current_screen(show_input_space=False)
            from .credential_checker import check_provider_credentials
            from .interactive_input import interactive_select
            from .model_selector import PROVIDERS

            # Build list of providers with status indicators
            choices = []
            provider_map = {}
            current_provider_name = provider or settings.llm_provider or None

            for idx, (prov_key, prov_info) in enumerate(PROVIDERS.items()):
                has_creds = check_provider_credentials(prov_key)
                choice_text = ""
                if has_creds:
                    choice_text += "[bold green]✓[/bold green] "
                else:
                    choice_text += "[bold yellow]⚠[/bold yellow] "
                choice_text += prov_info["display"]
                if has_creds:
                    choice_text += " [bold green][Ready][/bold green]"
                else:
                    choice_text += " [yellow][Setup Required][/yellow]"
                if prov_key == current_provider_name:
                    choice_text += " [bold cyan]← Current[/bold cyan]"

                choices.append(choice_text)
                provider_map[idx] = prov_key

            title = "Select LLM Provider"
            selected_choice = interactive_select(choices, title=title)

            if selected_choice is None:
                _render_current_screen(show_input_space=True)
                continue

            selected_idx = choices.index(selected_choice)
            new_provider = provider_map[selected_idx]

            # Check if credentials are configured
            has_creds = check_provider_credentials(new_provider)
            if not has_creds:
                # Ask user to configure credentials
                _render_current_screen(show_input_space=False)
                from rich.console import Console as RichConsole
                from rich.prompt import Prompt

                from .interactive_input import interactive_confirm

                setup_console = RichConsole()
                setup_console.print(
                    f"\n[bold yellow]⚠ Provider not configured[/bold yellow]\n"
                    f"Credentials required for "
                    f"[cyan]{PROVIDERS[new_provider]['display']}[/cyan]\n"
                )

                configure_now = interactive_confirm("Configure credentials now?", default=True)

                if not configure_now:
                    transcript.append(
                        "[bold yellow]Provider selection cancelled. "
                        "Credentials required.[/bold yellow]"
                    )
                    _render_current_screen(show_input_space=True)
                    continue

                # Configure credentials interactively
                config = {}
                try:
                    if new_provider == "openai":
                        setup_console.print(
                            "[dim]Get your API key at: https://platform.openai.com/api-keys[/dim]\n"
                        )
                        api_key = Prompt.ask("Enter OpenAI API key", password=True)
                        if not api_key:
                            transcript.append("[bold red]Error:[/bold red] API key is required")
                            _render_current_screen(show_input_space=True)
                            continue
                        config["RAGOPS_OPENAI_API_KEY"] = api_key

                    elif new_provider == "azure_openai":
                        setup_console.print(
                            "[dim]You need credentials from Azure OpenAI service.[/dim]\n"
                        )
                        api_key = Prompt.ask("Enter Azure OpenAI API key", password=True)
                        endpoint = Prompt.ask("Enter Azure OpenAI endpoint", default="")
                        deployment = Prompt.ask("Enter deployment name", default="")
                        if not api_key or not endpoint or not deployment:
                            transcript.append("[bold red]Error:[/bold red] All fields are required")
                            _render_current_screen(show_input_space=True)
                            continue
                        config["RAGOPS_AZURE_OPENAI_API_KEY"] = api_key
                        config["RAGOPS_AZURE_OPENAI_ENDPOINT"] = endpoint
                        config["RAGOPS_AZURE_OPENAI_DEPLOYMENT"] = deployment
                        config["RAGOPS_AZURE_OPENAI_API_VERSION"] = "2024-02-15-preview"

                    elif new_provider == "anthropic":
                        setup_console.print(
                            "[dim]Get your API key at: https://console.anthropic.com/[/dim]\n"
                        )
                        api_key = Prompt.ask("Enter Anthropic API key", password=True)
                        if not api_key:
                            transcript.append("[bold red]Error:[/bold red] API key is required")
                            _render_current_screen(show_input_space=True)
                            continue
                        config["RAGOPS_ANTHROPIC_API_KEY"] = api_key

                    elif new_provider == "vertex":
                        setup_console.print(
                            "[dim]You need a service account key file from Google Cloud.[/dim]\n"
                        )
                        path = Prompt.ask("Enter path to service account JSON file")
                        path = os.path.expanduser(path)
                        if not Path(path).exists():
                            transcript.append(f"[bold red]Error:[/bold red] File not found: {path}")
                            _render_current_screen(show_input_space=True)
                            continue
                        config["RAGOPS_VERTEX_CREDENTIALS"] = path

                    elif new_provider == "ollama":
                        base_url = Prompt.ask(
                            "Ollama base URL", default="http://localhost:11434/api/v1"
                        )
                        config["RAGOPS_OPENAI_API_KEY"] = "ollama"
                        config["RAGOPS_OPENAI_BASE_URL"] = base_url

                    elif new_provider == "openrouter":
                        setup_console.print(
                            "[dim]Get your API key at: https://openrouter.ai/keys[/dim]\n"
                        )
                        api_key = Prompt.ask("Enter OpenRouter API key", password=True)
                        if not api_key:
                            transcript.append("[bold red]Error:[/bold red] API key is required")
                            _render_current_screen(show_input_space=True)
                            continue
                        config["RAGOPS_OPENAI_API_KEY"] = api_key
                        config["RAGOPS_OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

                    # Save to .env file
                    from dotenv import dotenv_values

                    env_path = Path.cwd() / ".env"
                    existing_config = {}
                    if env_path.exists():
                        try:
                            existing_config = dict(dotenv_values(env_path))
                        except Exception:
                            pass

                    # Merge configs
                    merged_config = {**existing_config, **config}
                    merged_config["RAGOPS_LLM_PROVIDER"] = new_provider

                    # Write to .env
                    lines = []
                    if not env_path.exists():
                        lines.extend(
                            [
                                "# RAGOps Agent CE Configuration",
                                "",
                            ]
                        )

                    # Add provider setting
                    lines.append(f"RAGOPS_LLM_PROVIDER={new_provider}")
                    lines.append("")

                    # Add provider-specific settings
                    lines.append(f"# {new_provider.upper()} settings")
                    for key, value in config.items():
                        lines.append(f"{key}={value}")
                    lines.append("")

                    # Add any existing settings not overwritten
                    for key, value in existing_config.items():
                        if key not in merged_config or key == "RAGOPS_LLM_PROVIDER":
                            continue
                        if key not in config:
                            lines.append(f"{key}={value}")

                    env_path.write_text("\n".join(lines))
                    transcript.append(
                        "[bold green]✓ Credentials configured and saved to .env[/bold green]"
                    )

                except Exception as e:
                    transcript.append(
                        f"[bold red]Error:[/bold red] Failed to configure credentials: {str(e)}"
                    )
                    _render_current_screen(show_input_space=True)
                    continue

                # Reload settings after saving
                settings = load_settings()

            # Update provider
            os.environ["RAGOPS_LLM_PROVIDER"] = new_provider
            settings = settings.model_copy(update={"llm_provider": new_provider})
            try:
                prov = get_provider(settings, llm_provider=new_provider)
                agent_settings.llm_provider = prov
                provider = new_provider
                # Reset model when provider changes (model might not be compatible)
                model = None
                agent_settings.model = None
                agent = LLMAgent(
                    prov,
                    tools=tools,
                    mcp_clients=mcp_clients,
                )
                await agent._ainit_mcp_tools()
                transcript.append(
                    "[bold cyan]Provider updated:[/bold cyan] "
                    f"[yellow]{PROVIDERS[new_provider]['display']}[/yellow]"
                )

                # Automatically prompt for model selection after provider change
                _render_current_screen(show_input_space=False)
                from .interactive_input import interactive_select

                # Try to get chat models from provider dynamically
                models = []
                try:
                    if prov and hasattr(prov, "list_chat_models"):
                        models = prov.list_chat_models()
                    elif prov and hasattr(prov, "list_models"):
                        models = prov.list_models()
                except Exception as e:
                    logger.warning(f"Failed to get models from provider: {e}")

                # Fallback to common models if provider doesn't support listing or failed
                if not models:
                    common_models = {
                        "openai": [
                            "gpt-4o",
                            "gpt-4o-mini",
                            "gpt-4-turbo",
                            "gpt-4",
                            "gpt-3.5-turbo",
                            "o1-preview",
                            "o1-mini",
                        ],
                        "azure_openai": [
                            "gpt-4o",
                            "gpt-4o-mini",
                            "gpt-4-turbo",
                            "gpt-4",
                            "gpt-3.5-turbo",
                        ],
                        "vertex": [
                            "gemini-2.5-flash",
                            "gemini-2.0-flash-exp",
                            "gemini-1.5-pro",
                            "gemini-1.5-flash",
                            "gemini-pro",
                        ],
                        "ollama": [
                            "llama3.1",
                            "llama3",
                            "mistral",
                            "mixtral",
                            "phi3",
                            "codellama",
                        ],
                        "openrouter": [
                            "openai/gpt-4o",
                            "openai/gpt-4-turbo",
                            "anthropic/claude-3.5-sonnet",
                            "google/gemini-pro-1.5",
                            "meta-llama/llama-3.1-70b-instruct",
                        ],
                        "anthropic": [
                            "claude-3-5-sonnet-20241022",
                            "claude-3-opus-20240229",
                            "claude-3-sonnet-20240229",
                            "claude-3-haiku-20240307",
                        ],
                    }
                    models = common_models.get(new_provider, [])

                if models:
                    # Build choices list
                    choices = []
                    for model_name in models:
                        choice = model_name
                        if model_name == (agent_settings.model or model):
                            choice += " [bold cyan]← Current[/bold cyan]"
                        choices.append(choice)

                    # Add "Skip" option
                    choices.append("Skip (use default)")

                    title = f"Select Model for {PROVIDERS[new_provider]['display']}"
                    selected_choice = interactive_select(choices, title=title)

                    if selected_choice and selected_choice != "Skip (use default)":
                        # Extract model name (remove "← Current" if present)
                        new_model = selected_choice.split(" [")[0].strip()

                        # Validate model by trying to use it
                        try:
                            # Test if model is actually available by making a test request
                            test_messages = [Message(role="user", content="test")]
                            try:
                                # Try to generate with the model (with minimal tokens)
                                prov.generate(
                                    test_messages,
                                    model=new_model,
                                    max_tokens=1,
                                )
                                # If successful, model is available
                                agent_settings.model = new_model
                                model = new_model
                                transcript.append(
                                    "[bold cyan]Model selected:[/bold cyan] "
                                    f"[yellow]{new_model}[/yellow]"
                                )
                                save_model_selection(new_provider, new_model)
                            except Exception as model_error:
                                # Model is not available
                                error_msg = str(model_error)
                                # Extract more user-friendly error message
                                if "model" in error_msg.lower() and (
                                    "not found" in error_msg.lower()
                                    or "does not exist" in error_msg.lower()
                                    or "not available" in error_msg.lower()
                                ):
                                    friendly_msg = (
                                        f"Model '{new_model}' is not available or not "
                                        "accessible with your API key."
                                    )
                                else:
                                    friendly_msg = (
                                        f"Model '{new_model}' is not available: {error_msg}"
                                    )
                                transcript.append(
                                    f"[bold red]Error:[/bold red] {friendly_msg}\n"
                                    "[yellow]Please select a different model.[/yellow]"
                                )
                                # Don't save the invalid model
                        except Exception as e:
                            # If validation itself fails, still try to set the model
                            # but warn the user
                            logger.warning(f"Model validation failed: {e}")
                            agent_settings.model = new_model
                            model = new_model
                            transcript.append(
                                "[bold cyan]Model selected:[/bold cyan] "
                                f"[yellow]{new_model}[/yellow]\n"
                                "[dim yellow]Note: Could not validate model "
                                "availability[/dim yellow]"
                            )
                            save_model_selection(new_provider, new_model)
                    elif selected_choice == "Skip (use default)":
                        transcript.append("[dim]Using default model for this provider[/dim]")
                    # If None (cancelled), just continue without setting model
                else:
                    transcript.append(
                        "[bold yellow]No models available for selection. "
                        "Please specify model name in .env file or use CLI flag.[/bold yellow]"
                    )

            except Exception as e:
                transcript.append(
                    f"[bold red]Error:[/bold red] Failed to initialize provider: {str(e)}"
                )
            _render_current_screen(show_input_space=True)
            continue

        if user_input == ":model":
            # Select model interactively
            _render_current_screen(show_input_space=False)
            from .interactive_input import interactive_select

            current_provider_name = provider or settings.llm_provider or "openai"
            current_model_name = agent_settings.model or model

            # Try to get chat models from provider dynamically (for agent use)
            models = []
            try:
                if prov and hasattr(prov, "list_chat_models"):
                    models = prov.list_chat_models()
                elif prov and hasattr(prov, "list_models"):
                    # Fallback to all models if list_chat_models not available
                    models = prov.list_models()
            except Exception as e:
                logger.warning(f"Failed to get models from provider: {e}")

            # Fallback to common models if provider doesn't support listing or failed
            if not models:
                common_models = {
                    "openai": [
                        "gpt-4o",
                        "gpt-4o-mini",
                        "gpt-4-turbo",
                        "gpt-4",
                        "gpt-3.5-turbo",
                        "o1-preview",
                        "o1-mini",
                    ],
                    "azure_openai": [
                        "gpt-4o",
                        "gpt-4o-mini",
                        "gpt-4-turbo",
                        "gpt-4",
                        "gpt-3.5-turbo",
                    ],
                    "vertex": [
                        "gemini-2.5-flash",
                        "gemini-2.0-flash-exp",
                        "gemini-1.5-pro",
                        "gemini-1.5-flash",
                        "gemini-pro",
                        "gemini-pro-vision",
                    ],
                    "ollama": [
                        "llama3.1",
                        "llama3",
                        "mistral",
                        "mixtral",
                        "phi3",
                        "codellama",
                    ],
                    "openrouter": [
                        "openai/gpt-4o",
                        "openai/gpt-4-turbo",
                        "anthropic/claude-3.5-sonnet",
                        "google/gemini-pro-1.5",
                        "meta-llama/llama-3.1-70b-instruct",
                    ],
                    "anthropic": [
                        "claude-3-5-sonnet-20241022",
                        "claude-3-opus-20240229",
                        "claude-3-sonnet-20240229",
                        "claude-3-haiku-20240307",
                    ],
                }
                models = common_models.get(current_provider_name, [])

            if not models:
                # If no predefined models, ask user to input
                transcript.append(
                    "[bold yellow]No predefined models for this provider.[/bold yellow]\n"
                    f"Current model: [cyan]{current_model_name or 'not set'}[/cyan]\n"
                    "Please specify model name in .env file or use CLI flag."
                )
                _render_current_screen(show_input_space=True)
                continue

            # Build choices list
            choices = []
            for model_name in models:
                choice = model_name
                if model_name == current_model_name:
                    choice += " [bold cyan]← Current[/bold cyan]"
                choices.append(choice)

            # Add "Custom" option
            choices.append("Custom (enter manually)")

            title = f"Select Model for {current_provider_name}"
            selected_choice = interactive_select(choices, title=title)

            if selected_choice is None:
                _render_current_screen(show_input_space=True)
                continue

            if selected_choice == "Custom (enter manually)":
                transcript.append(
                    "[bold yellow]Please specify model name in .env file or "
                    "use CLI flag.[/bold yellow]"
                )
                _render_current_screen(show_input_space=True)
                continue

            # Extract model name (remove "← Current" if present)
            new_model = selected_choice.split(" [")[0].strip()

            # Validate model by trying to use it
            try:
                # Test if model is actually available
                test_messages = [Message(role="user", content="test")]
                try:
                    prov.generate(
                        test_messages,
                        model=new_model,
                        max_tokens=1,
                    )
                    # If successful, model is available
                    agent_settings.model = new_model
                    model = new_model
                    transcript.append(
                        "[bold cyan]Model updated:[/bold cyan] " f"[yellow]{new_model}[/yellow]"
                    )
                    save_model_selection(current_provider_name, new_model)
                except Exception as model_error:
                    # Model is not available
                    error_msg = str(model_error)
                    if "model" in error_msg.lower() and (
                        "not found" in error_msg.lower()
                        or "does not exist" in error_msg.lower()
                        or "not available" in error_msg.lower()
                    ):
                        friendly_msg = (
                            f"Model '{new_model}' is not available or not "
                            "accessible with your API key."
                        )
                    else:
                        friendly_msg = f"Model '{new_model}' is not available: {error_msg}"
                    transcript.append(
                        f"[bold red]Error:[/bold red] {friendly_msg}\n"
                        "[yellow]Please select a different model using :model command.[/yellow]"
                    )
                    # Don't save the invalid model
            except Exception as e:
                # If validation itself fails, still try to set the model but warn
                logger.warning(f"Model validation failed: {e}")
                agent_settings.model = new_model
                model = new_model
                transcript.append(
                    "[bold cyan]Model updated:[/bold cyan] "
                    f"[yellow]{new_model}[/yellow]\n"
                    "[dim yellow]Note: Could not validate model availability[/dim yellow]"
                )
                save_model_selection(current_provider_name, new_model)
            _render_current_screen(show_input_space=True)
            continue

        if user_input in {":q", ":quit", ":exit", "exit", "quit"}:
            transcript.append("[Bye]")
            if watcher:
                watcher.stop()
            _sanitize_transcript(transcript)
            _render_current_screen(show_input_space=False)
            renderer.render_goodbye_screen()
            break

        _append_user_line(user_input)
        _sanitize_transcript(transcript)
        _render_current_screen(show_input_space=False)

        try:
            history.append(Message(role="user", content=user_input))
            # Use streaming if provider supports it
            if prov.supports_streaming():
                reply = ""
                interrupted = False

                # Add placeholder to transcript
                response_index = _start_agent_placeholder()

                try:
                    # Accumulate everything in display order
                    display_content = ""
                    # Temporary message shown only during execution
                    temp_executing = ""

                    # Stream events - process content and tool calls
                    async for event in agent.arespond_stream(history, model=model):
                        reply, display_content, temp_executing = _process_stream_event(
                            event, reply, display_content, temp_executing
                        )

                        # Update transcript with permanent content + temporary executing message
                        _set_agent_line(response_index, display_content, temp_executing)

                        # Re-render screen after each event
                        _render_current_screen(show_input_space=False)
                except KeyboardInterrupt:
                    interrupted = True
                    transcript[response_index] = (
                        f"{_time_str()} [yellow]⚠ Generation interrupted by user[/yellow]"
                    )
                except Exception as e:
                    # Show error to user
                    error_msg = f"{_time_str()} [bold red]Error:[/bold red] {str(e)}"
                    if response_index is not None:
                        transcript[response_index] = error_msg
                    else:
                        transcript.append(error_msg)
                    logger.error(f"Error during streaming: {e}", exc_info=True)
                    _render_current_screen(show_input_space=False)

                # Add to history if we got a response
                if reply and not interrupted:
                    history.append(Message(role="assistant", content=reply))
                elif not interrupted and not reply:
                    # No reply but no interruption - likely an error was swallowed
                    error_msg = (
                        f"{_time_str()} [bold red]Error:[/bold red] "
                        "No response from agent. Check logs for details."
                    )
                    if response_index is not None:
                        transcript[response_index] = error_msg
                    else:
                        transcript.append(error_msg)
                    _render_current_screen(show_input_space=False)
            else:
                # Fall back to non-streaming mode
                # Add placeholder
                response_index = _start_agent_placeholder()

                try:
                    reply = await agent.arespond(history, model=model)
                    if reply:
                        history.append(Message(role="assistant", content=reply))
                        # Replace placeholder with actual response
                        rendered_reply = _render_markdown_to_rich(reply)
                        _set_agent_line(response_index, rendered_reply, "")
                    else:
                        # No reply - likely an error
                        transcript[response_index] = (
                            f"{_time_str()} [bold red]Error:[/bold red] "
                            "No response from agent. Check logs for details."
                        )
                except KeyboardInterrupt:
                    console.print("\n[yellow]⚠ Generation interrupted by user[/yellow]")
                    transcript[response_index] = (
                        f"{_time_str()} [yellow]Generation interrupted[/yellow]"
                    )
                except Exception as e:
                    error_msg = f"{_time_str()} [bold red]Error:[/bold red] {str(e)}"
                    transcript[response_index] = error_msg
                    logger.error(f"Error during agent response: {e}", exc_info=True)
                _render_current_screen(show_input_space=False)
        except Exception as e:
            transcript.append(f"{_time_str()} [bold red]Error:[/bold red] {str(e)}")
            logger.error(f"Error in main loop: {e}", exc_info=True)
            _render_current_screen(show_input_space=False)
