"""CLI entry point for RAGOps Agent CE.

Follows Single Responsibility Principle - only handles CLI commands and routing.
REPL logic is delegated to dedicated REPL classes.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import time

import typer

from donkit_ragops import __version__
from donkit_ragops.agent.agent import LLMAgent, default_tools
from donkit_ragops.agent.prompts import get_prompt
from donkit_ragops.config import load_settings
from donkit_ragops.display import print_startup_header
from donkit_ragops.llm.provider_factory import get_provider
from donkit_ragops.logging_config import setup_logging
from donkit_ragops.mcp.client import MCPClient
from donkit_ragops.mode import Mode, detect_mode
from donkit_ragops.model_selector import save_model_selection, select_model_at_startup
from donkit_ragops.repl.base import ReplContext
from donkit_ragops.repl.local_repl import LocalREPL
from donkit_ragops.schemas.agent_schemas import AgentSettings
from donkit_ragops.setup_wizard import run_setup_if_needed
from donkit_ragops.ui import get_ui
from donkit_ragops.ui.styles import StyleName, styled_text

app = typer.Typer(
    pretty_exceptions_enable=False,
)

DEFAULT_MCP_COMMANDS = ["donkit-ragops-mcp"]

# Default models for providers
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "azure_openai": "gpt-4o-mini",
    "vertex": "gemini-2.5-flash",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
    "donkit": "donkit-fast",
    "anthropic": "claude-3-5-sonnet-20241022",
    "mock": "gpt-4o-mini",
}


def version_callback(value: bool) -> None:
    """Handle --version flag."""
    if value:
        ui = get_ui()
        ui.print(f"donkit-ragops {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
    setup: bool = typer.Option(
        False,
        "--setup",
        help="Run setup wizard to configure the agent",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Force local mode (overrides auto-detection)",
    ),
    enterprise: bool = typer.Option(
        False,
        "--enterprise",
        help="Force enterprise mode (requires authentication via 'login')",
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
    """donkit-ragops - LLM-powered CLI agent for building RAG pipelines.

    Start an interactive REPL session where the agent orchestrates built-in MCP servers
    to plan, chunk, read, evaluate, and load documents into vector stores.

    MODES:
      Local mode (default)    - Build RAG pipelines locally using Docker
      Enterprise mode         - Connect to Donkit cloud (requires 'login')

    MCP SERVERS (built-in):
      • rag-planner           - RAG configuration planning
      • chunker               - Document chunking strategies
      • read-engine           - Document parsing (PDF, Office, images)
      • vectorstore-loader    - Vector DB operations (Qdrant, Milvus, Chroma)
      • compose-manager       - Docker Compose orchestration
      • rag-query             - RAG querying
      • rag-evaluation        - RAG pipeline evaluation

    REPL COMMANDS:
      :help, :h, :?           - Show available commands
      :quit, :q, exit         - Exit the agent
      :clear                  - Clear conversation history
      :provider               - Select LLM provider
      :model                  - Select LLM model
      /                       - Open command palette (autocomplete)
      @/path                  - Attach files (enterprise mode only)

    EXAMPLES:
      donkit-ragops                              # Start in local mode
      donkit-ragops --enterprise                 # Start in enterprise mode
      donkit-ragops --setup                      # Run setup wizard
      donkit-ragops --provider openai -m gpt-4o  # Override provider & model
      donkit-ragops status                       # Check mode & auth status
    """
    # Setup logging
    try:
        setup_logging(load_settings())
    except Exception:
        pass

    if ctx.invoked_subcommand is None:
        ui = get_ui()
        # Validate mutual exclusion
        if local and enterprise:
            ui.print_error("Cannot use --local and --enterprise together")
            raise typer.Exit(code=1)

        # Run setup wizard if needed
        if not run_setup_if_needed(force=setup):
            raise typer.Exit(code=1)

        if setup:
            raise typer.Exit()

        # Detect mode
        try:
            mode = detect_mode(force_local=local, force_enterprise=enterprise)
        except ValueError as e:
            ui.print_error(str(e))
            raise typer.Exit(code=1)

        # Route to appropriate REPL
        if mode == Mode.ENTERPRISE:
            _run_enterprise_mode()
        else:
            _run_local_mode(system, model, provider, show_checklist)


def _run_enterprise_mode() -> None:
    """Run enterprise mode with Rich + prompt_toolkit REPL."""
    from donkit_ragops.enterprise.auth import get_token
    from donkit_ragops.enterprise.config import load_enterprise_settings
    from donkit_ragops.repl.base import ReplContext
    from donkit_ragops.repl.enterprise_repl import EnterpriseREPL

    ui = get_ui()

    # Import API client - it's installed as a dependency
    try:
        from donkit.ragops_api_gateway_client.client import RagopsAPIGatewayClient
    except ImportError:
        ui.print_error("Enterprise client not installed")
        raise typer.Exit(code=1)

    token = get_token()
    if not token:
        ui.print_error(
            "Enterprise mode requires authentication.\n" "Run 'donkit-ragops login' first."
        )
        raise typer.Exit(code=1)

    enterprise_settings = load_enterprise_settings()

    api_client = RagopsAPIGatewayClient(
        base_url=enterprise_settings.api_url,
        api_token=token,
    )

    # Create context and run REPL
    context = ReplContext(
        session_started_at=time.time(),
        show_checklist=False,
    )

    repl = EnterpriseREPL(
        context=context,
        api_client=api_client,
        token=token,
        enterprise_settings=enterprise_settings,
    )
    _run_repl_with_interrupt_handling(repl)


def _run_local_mode(
    system: str | None,
    model: str | None,
    provider: str | None,
    show_checklist: bool,
) -> None:
    """Run local mode REPL."""
    from donkit_ragops import texts

    ui = get_ui()
    ui.print("Mode: local", StyleName.DIM)

    # Model selection at startup
    if provider is None:
        model_selection = select_model_at_startup()
        if model_selection is None:
            ui.print_error(texts.ERROR_MODEL_SELECTION_CANCELLED)
            raise typer.Exit(code=1)
        provider, model_from_selection = model_selection
        if model is None:
            model = model_from_selection
    else:
        save_model_selection(provider, model)

    # Ensure model is set
    if model is None:
        model = DEFAULT_MODELS.get((provider or "").lower(), "gpt-4o-mini")
        ui.print_styled(
            styled_text(
                (StyleName.WARNING, "No model selected. Using default: "),
                (StyleName.INFO, model),
            )
        )

    # Initialize provider and agent
    settings = load_settings()
    if provider:
        settings = settings.model_copy(update={"llm_provider": provider})

    try:
        prov = get_provider(settings, llm_provider=provider, model_name=model)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        from donkit_ragops import texts

        ui.print_error(texts.ERROR_PROVIDER_INIT.format(provider=provider, error=e))
        ui.print_error(texts.ERROR_CREDENTIALS_REQUIRED)
        raise typer.Exit(code=1)

    tools = default_tools()

    # Create MCP clients
    mcp_clients = []
    for cmd_str in DEFAULT_MCP_COMMANDS:
        cmd_parts = shlex.split(cmd_str)
        mcp_clients.append(MCPClient(cmd_parts[0], cmd_parts[1:]))

    # Create agent
    agent = LLMAgent(prov, tools=tools, mcp_clients=mcp_clients)

    # Create agent settings
    agent_settings = AgentSettings(llm_provider=prov, model=model)

    # Get system prompt
    system_prompt = system or get_prompt(provider, debug=settings.log_level == "DEBUG")

    # Print startup header
    print_startup_header(provider=provider or "initializing", model=model)

    # Create context with dependencies
    context = ReplContext(
        session_started_at=time.time(),
        show_checklist=show_checklist,
        provider=prov,
        provider_name=provider,
        model=model,
        agent=agent,
        mcp_clients=mcp_clients,
        agent_settings=agent_settings,
        system_prompt=system_prompt,
    )

    repl = LocalREPL(context)
    _run_repl_with_interrupt_handling(repl)


def _run_repl_with_interrupt_handling(repl) -> None:
    """Run REPL with proper interrupt handling."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    repl_task = loop.create_task(repl.run())

    try:
        while not repl_task.done():
            try:
                loop.run_until_complete(repl_task)
            except KeyboardInterrupt:
                repl_task.cancel()
    finally:
        loop.close()


@app.command()
def ping() -> None:
    """Simple health command to verify the CLI is working."""
    ui = get_ui()
    ui.print("pong")


@app.command()
def login(
    token: str = typer.Option(
        ...,
        "--token",
        "-t",
        prompt="Enter your API token",
        hide_input=True,
        help="Your Donkit API token",
    ),
) -> None:
    """Login to Donkit cloud (enterprise mode)."""
    from donkit_ragops.enterprise.auth import save_token
    from donkit_ragops.enterprise.config import load_enterprise_settings

    ui = get_ui()

    # Import API client
    try:
        from donkit.ragops_api_gateway_client.client import RagopsAPIGatewayClient
    except ImportError:
        ui.print_error("Enterprise client not installed")
        raise typer.Exit(code=1)

    settings = load_enterprise_settings()

    ui.print("Validating token...", StyleName.DIM)
    try:
        client = RagopsAPIGatewayClient(
            base_url=settings.api_url,
            api_token=token,
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:

            async def validate():
                async with client:
                    await client.list_projects()

            loop.run_until_complete(validate())
        finally:
            loop.close()

        ui.print_success("Token validated successfully.")
    except Exception as e:
        ui.print_error(f"Token validation failed: {e}")
        ui.print_warning("Please check your token and try again.")
        raise typer.Exit(code=1)

    save_token(token)
    ui.print_success("Token saved successfully.")
    ui.print(f"API URL: {settings.api_url}", StyleName.DIM)
    ui.print("\nYou can now use enterprise mode:")
    ui.print_styled(styled_text((None, "  "), (StyleName.INFO, "donkit-ragops --enterprise")))


@app.command()
def logout() -> None:
    """Logout from Donkit cloud (remove stored token)."""
    from donkit_ragops.enterprise.auth import delete_token, has_token

    ui = get_ui()
    if has_token():
        delete_token()
        ui.print_success("Token removed successfully.")
    else:
        ui.print("No token found.", StyleName.DIM)


@app.command()
def status() -> None:
    """Show current mode and authentication status."""
    from donkit_ragops.enterprise.auth import has_token
    from donkit_ragops.enterprise.config import load_enterprise_settings

    ui = get_ui()
    mode = detect_mode()
    enterprise_settings = load_enterprise_settings()

    ui.print_styled(styled_text((StyleName.BOLD, "Mode: "), (None, mode.value)))
    ui.print_styled(styled_text((StyleName.BOLD, "Version: "), (None, __version__)))

    if mode.value == "local":
        try:
            local_settings = load_settings()
            ui.print_styled(
                styled_text(
                    (StyleName.BOLD, "Provider: "),
                    (None, local_settings.llm_provider or "not configured"),
                )
            )
            ui.print_styled(
                styled_text(
                    (StyleName.BOLD, "Model: "),
                    (None, local_settings.llm_model or "default"),
                )
            )
            ui.print_styled(
                styled_text(
                    (StyleName.BOLD, "MCP Servers: "),
                    (None, ", ".join(DEFAULT_MCP_COMMANDS)),
                )
            )
        except Exception:
            ui.print_styled(
                styled_text(
                    (StyleName.BOLD, "Provider: "),
                    (StyleName.DIM, "not configured"),
                )
            )

    if has_token():
        ui.print_styled(
            styled_text(
                (StyleName.BOLD, "Enterprise auth: "),
                (StyleName.SUCCESS, "authenticated"),
            )
        )
        ui.print_styled(
            styled_text(
                (StyleName.BOLD, "API URL: "),
                (None, enterprise_settings.api_url),
            )
        )
    else:
        ui.print_styled(
            styled_text(
                (StyleName.BOLD, "Enterprise auth: "),
                (StyleName.DIM, "not authenticated"),
            )
        )
        ui.print("Run 'donkit-ragops login' to authenticate", StyleName.DIM)
