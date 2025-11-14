from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import time

import typer
from loguru import logger
from rich.console import Console

from ragops_agent_ce import __version__
from ragops_agent_ce import texts
from ragops_agent_ce.agent.agent import LLMAgent
from ragops_agent_ce.agent.agent import default_tools
from ragops_agent_ce.agent.prompts import OPENAI_SYSTEM_PROMPT
from ragops_agent_ce.agent.prompts import VERTEX_SYSTEM_PROMPT
from ragops_agent_ce.cli_helpers import format_model_choices
from ragops_agent_ce.cli_helpers import get_available_models
from ragops_agent_ce.cli_helpers import select_provider_interactively
from ragops_agent_ce.cli_helpers import validate_model_choice
from ragops_agent_ce.config import load_settings
from ragops_agent_ce.display import ScreenRenderer
from ragops_agent_ce.interactive_input import get_user_input
from ragops_agent_ce.interactive_input import interactive_select
from ragops_agent_ce.llm.provider_factory import get_provider
from ragops_agent_ce.llm.types import Message
from ragops_agent_ce.logging_config import setup_logging
from ragops_agent_ce.mcp.client import MCPClient
from ragops_agent_ce.model_selector import PROVIDERS
from ragops_agent_ce.model_selector import save_model_selection
from ragops_agent_ce.model_selector import select_model_at_startup
from ragops_agent_ce.prints import RAGOPS_LOGO_ART
from ragops_agent_ce.prints import RAGOPS_LOGO_TEXT
from ragops_agent_ce.repl_helpers import MCPEventHandler
from ragops_agent_ce.repl_helpers import build_stream_render_helper
from ragops_agent_ce.repl_helpers import format_timestamp
from ragops_agent_ce.schemas.agent_schemas import AgentSettings
from ragops_agent_ce.setup_wizard import run_setup_if_needed

app = typer.Typer(
    pretty_exceptions_enable=False,
)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"ragops-agent-ce {__version__}")
        raise typer.Exit()


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
            raise typer.Exit()

        # Model selection at startup (mandatory)
        # Only skip if provider is explicitly provided via CLI flag
        if provider is None:
            model_selection = select_model_at_startup()
            if model_selection is None:
                console.print(texts.ERROR_MODEL_SELECTION_CANCELLED)
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
    """Convert Markdown text to simple Rich markup without breaking formatting."""
    # Simple markdown to rich markup conversion without full rendering
    # This preserves the transcript panel formatting

    # Bold: **text** -> [bold]text[/bold]
    result = re.sub(r"\*\*(.+?)\*\*", r"[bold]\1[/bold]", text)

    # Italic: *text* -> [italic]text[/italic] (but don't match list items)
    result = re.sub(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)", r"[italic]\1[/italic]", result)  # noqa

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
        console.print(texts.ERROR_PROVIDER_INIT.format(provider=provider, error=e))
        console.print(texts.ERROR_CREDENTIALS_REQUIRED)
        raise typer.Exit(code=1)

    tools = default_tools()

    session_started_at = time.time()

    history: list[Message] = []
    if system:
        history.append(Message(role="system", content=system))

    # Transcript buffer must be available for helper functions below
    transcript: list[str] = []
    agent_settings = AgentSettings(llm_provider=prov, model=model)
    renderer = ScreenRenderer()

    render_helper = build_stream_render_helper(
        transcript=transcript,
        renderer=renderer,
        agent_settings=agent_settings,
        session_started_at=session_started_at,
        show_checklist=show_checklist,
    )

    mcp_handler = MCPEventHandler(
        render_helper=render_helper,
        agent_settings=agent_settings,
        session_started_at=session_started_at,
        show_checklist=show_checklist,
    )

    # Create MCP clients with progress callback
    mcp_clients = []
    commands = mcp_commands if mcp_commands is not None else []
    if commands:
        for cmd_str in commands:
            cmd_parts = shlex.split(cmd_str)
            mcp_clients.append(
                MCPClient(
                    cmd_parts[0], cmd_parts[1:], progress_callback=mcp_handler.progress_callback
                )
            )

    # Create agent
    agent = LLMAgent(
        prov,
        tools=tools,
        mcp_clients=mcp_clients,
    )
    # Initialize MCP tools asynchronously
    await agent.ainit_mcp_tools()
    renderer.render_startup_screen()

    rendered_welcome = _render_markdown_to_rich(texts.WELCOME_MESSAGE)
    render_helper.append_agent_message(rendered_welcome)
    while True:
        render_helper.render_current_screen()
        user_input = get_user_input()
        if not user_input:
            continue

        if user_input == ":help":
            transcript += texts.HELP_COMMANDS
            continue

        if user_input == ":clear":
            transcript.clear()
            continue

        if user_input == ":provider":
            render_helper.render_current_screen()
            new_provider = await select_provider_interactively(
                current_provider=provider or settings.llm_provider,
                current_model=agent_settings.model or model,
                console=console,
                settings=settings,
                agent_settings=agent_settings,
                prov=prov,
                agent=agent,
                tools=tools,
                mcp_clients=mcp_clients,
            )

            if new_provider is None:
                render_helper.render_current_screen()
                continue

            transcript.extend(new_provider.messages)

            if new_provider.cancelled:
                render_helper.render_current_screen()
                continue

            provider = new_provider.provider
            prov = new_provider.prov
            agent = new_provider.agent
            settings = new_provider.settings
            agent_settings = new_provider.agent_settings
            model = new_provider.model
            mcp_handler.agent_settings = agent_settings
            mcp_handler.clear_progress()

            render_helper.render_current_screen()

            if new_provider.prompt_model_selection and provider:
                models = get_available_models(prov, provider)

                if models:
                    choices = format_model_choices(models, agent_settings.model or model)
                    choices.append(texts.MODEL_SELECT_SKIP)

                    title = texts.MODEL_SELECT_TITLE.format(provider=PROVIDERS[provider]["display"])
                    selected_choice = interactive_select(choices, title=title)

                    if selected_choice and selected_choice != texts.MODEL_SELECT_SKIP:
                        new_model = selected_choice.split(" [")[0].strip()
                        success, messages = validate_model_choice(
                            prov, provider, new_model, agent_settings
                        )
                        transcript.extend(messages)
                        if success:
                            model = new_model
                    elif selected_choice == texts.MODEL_SELECT_SKIP:
                        transcript.append(texts.MODEL_USING_DEFAULT)
                else:
                    transcript.append(texts.MODEL_NO_AVAILABLE)

            render_helper.render_current_screen()
            continue

        if user_input == ":model":
            # Select model interactively
            render_helper.render_current_screen()
            current_provider_name = provider or settings.llm_provider or "openai"
            current_model_name = agent_settings.model or model

            models = get_available_models(prov, current_provider_name)
            if not models:
                # If no predefined models, ask user to input
                transcript.append(
                    texts.MODEL_NO_PREDEFINED.format(
                        current_model_name=current_model_name or "not set"
                    )
                )
                render_helper.render_current_screen()
                continue

            # Build choices list
            choices = format_model_choices(models, current_model_name)

            # Add "Custom" option
            choices.append("Custom (enter manually)")

            title = f"Select Model for {current_provider_name}"
            selected_choice = interactive_select(choices, title=title)

            if selected_choice is None:
                render_helper.render_current_screen()
                continue

            if selected_choice == "Custom (enter manually)":
                transcript.append(
                    "[bold yellow]Please specify model name in .env file or "
                    "use CLI flag.[/bold yellow]"
                )
                render_helper.render_current_screen()
                continue

            # Extract model name (remove "← Current" if present)
            new_model = selected_choice.split(" [")[0].strip()

            success, messages = validate_model_choice(
                prov, current_provider_name, new_model, agent_settings
            )
            transcript.extend(messages)
            if success:
                model = new_model
            render_helper.render_current_screen()
            continue

        if user_input in {":q", ":quit", ":exit", "exit", "quit"}:
            transcript.append("[Bye]")
            render_helper.render_current_screen()
            renderer.render_goodbye_screen()
            break

        render_helper.append_user_line(user_input)
        render_helper.render_current_screen()

        try:
            history.append(Message(role="user", content=user_input))
            # Use streaming if provider supports it
            if prov.supports_streaming():
                reply = ""
                interrupted = False

                # Add placeholder to transcript
                response_index = render_helper.start_agent_placeholder()

                try:
                    # Accumulate everything in display order
                    display_content = ""
                    # Temporary message shown only during execution
                    temp_executing = ""
                    # Stream events - process content and tool calls
                    async for event in agent.arespond_stream(history, model=model):
                        reply, display_content, temp_executing = mcp_handler.process_stream_event(
                            event,
                            history,
                            reply,
                            display_content,
                            temp_executing,
                        )
                        # Update transcript with permanent content + temporary executing message
                        render_helper.set_agent_line(
                            response_index, display_content, temp_executing
                        )
                        render_helper.render_current_screen()
                except KeyboardInterrupt:
                    interrupted = True
                    transcript[response_index] = (
                        f"{format_timestamp()} [yellow]⚠ Generation interrupted by user[/yellow]"
                    )
                except Exception as e:
                    # Show error to user
                    error_msg = f"{format_timestamp()} [bold red]Error:[/bold red] {str(e)}"
                    if response_index is not None:
                        transcript[response_index] = error_msg
                    else:
                        transcript.append(error_msg)
                    logger.error(f"Error during streaming: {e}", exc_info=True)
                    render_helper.render_current_screen()
                    mcp_handler.clear_progress()

                # Add to history if we got a response
                if reply and not interrupted:
                    history.append(Message(role="assistant", content=reply))
                elif not interrupted and not reply:
                    # No reply but no interruption - likely an error was swallowed
                    error_msg = (
                        f"{format_timestamp()} [bold red]Error:[/bold red] No response from agent"
                    )
                    if response_index is not None:
                        transcript[response_index] = error_msg
                    else:
                        transcript.append(error_msg)
                    render_helper.render_current_screen()
                mcp_handler.clear_progress()
            else:
                # Fall back to non-streaming mode
                # Add placeholder
                response_index = render_helper.start_agent_placeholder()

                try:
                    reply = await agent.arespond(history, model=model)
                    if reply:
                        history.append(Message(role="assistant", content=reply))
                        # Replace placeholder with actual response
                        rendered_reply = _render_markdown_to_rich(reply)
                        render_helper.set_agent_line(response_index, rendered_reply, "")
                    else:
                        # No reply - likely an error
                        transcript[response_index] = (
                            f"{format_timestamp()} [bold red]Error:[/bold red] "
                            "No response from agent. Check logs for details."
                        )
                except KeyboardInterrupt:
                    console.print("\n[yellow]⚠ Generation interrupted by user[/yellow]")
                    transcript[response_index] = (
                        f"{format_timestamp()} [yellow]Generation interrupted[/yellow]"
                    )
                except Exception as e:
                    error_msg = f"{format_timestamp()} [bold red]Error:[/bold red] {str(e)}"
                    transcript[response_index] = error_msg
                    logger.error(f"Error during agent response: {e}", exc_info=True)
                render_helper.render_current_screen()
                mcp_handler.clear_progress()
        except Exception as e:
            render_helper.append_error(str(e))
            logger.error(f"Error in main loop: {e}", exc_info=True)
            render_helper.render_current_screen()
