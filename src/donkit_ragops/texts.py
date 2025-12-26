"""UI text constants for RAGOps Agent CE.

This module contains all user-facing text strings used in the CLI,
organized by feature/section for easy maintenance and localization.

Text constants are plain strings without framework-specific markup.
Use styled_* functions for styled output with the UI abstraction layer.
"""

from __future__ import annotations

from donkit_ragops.ui.styles import StyledText, StyleName, styled_text

# ============================================================================
# Help and Commands (plain text + styled versions)
# ============================================================================

HELP_COMMANDS_PLAIN = [
    "",
    "Available commands:",
    "  :help - Show this help message",
    "  :q or :quit - Exit the agent",
    "  :clear - Clear the conversation transcript",
    "  :provider - Select LLM provider",
    "  :model - Select LLM model",
    "",
    "Quick actions:",
    "  / - Open command palette (autocomplete commands)",
    "  @/path - Attach files (enterprise mode only)",
]


def styled_help_commands() -> list[StyledText]:
    """Return styled help commands."""
    return [
        styled_text((None, "")),
        styled_text((StyleName.WARNING, "Available commands:")),
        styled_text((StyleName.BOLD, "  :help"), (None, " - Show this help message")),
        styled_text(
            (StyleName.BOLD, "  :q"),
            (None, " or "),
            (StyleName.BOLD, ":quit"),
            (None, " - Exit the agent"),
        ),
        styled_text((StyleName.BOLD, "  :clear"), (None, " - Clear the conversation transcript")),
        styled_text((StyleName.BOLD, "  :provider"), (None, " - Select LLM provider")),
        styled_text((StyleName.BOLD, "  :model"), (None, " - Select LLM model")),
        styled_text((None, "")),
        styled_text((StyleName.WARNING, "Quick actions:")),
        styled_text(
            (StyleName.BOLD, "  /"), (None, " - Open command palette (autocomplete commands)")
        ),
        styled_text(
            (StyleName.BOLD, "  @/path"),
            (None, " - Attach files "),
            (StyleName.DIM, "(enterprise mode only)"),
        ),
    ]


# Legacy: Keep for backward compatibility (Rich markup)
HELP_COMMANDS = [
    "",
    "  [yellow]Available commands:[/yellow]",
    "  [bold]:help[/bold] - Show this help message",
    "  [bold]:q[/bold] or [bold]:quit[/bold] - Exit the agent",
    "  [bold]:clear[/bold] - Clear the conversation transcript",
    "  [bold]:provider[/bold] - Select LLM provider",
    "  [bold]:model[/bold] - Select LLM model",
    "",
    "  [yellow]Quick actions:[/yellow]",
    "  [bold]/[/bold] - Open command palette (autocomplete commands)",
    "  [bold]@/path[/bold] - Attach files [dim](enterprise mode only)[/dim]",
]

# ============================================================================
# Provider Selection
# ============================================================================

PROVIDER_SELECT_TITLE = "Select LLM Provider"
PROVIDER_CONFIGURE_PROMPT = "Configure credentials now?"

# Plain text versions
PROVIDER_STATUS_READY_PLAIN = "[Ready]"
PROVIDER_STATUS_SETUP_REQUIRED_PLAIN = "[Setup Required]"
PROVIDER_CURRENT_PLAIN = "<- Current"


def styled_provider_status_ready() -> StyledText:
    """Return styled 'Ready' status."""
    return styled_text((StyleName.SUCCESS, "[Ready]"))


def styled_provider_status_setup_required() -> StyledText:
    """Return styled 'Setup Required' status."""
    return styled_text((StyleName.WARNING, "[Setup Required]"))


def styled_provider_current() -> StyledText:
    """Return styled 'Current' marker."""
    return styled_text((StyleName.INFO, "<- Current"))


def styled_provider_not_configured(provider: str) -> StyledText:
    """Return styled 'Provider not configured' message."""
    return styled_text(
        (StyleName.WARNING, "Provider not configured\n"),
        (None, f"Credentials required for {provider}\n"),
    )


def styled_provider_selection_cancelled() -> StyledText:
    """Return styled 'Selection cancelled' message."""
    return styled_text((StyleName.WARNING, "Provider selection cancelled. Credentials required."))


def styled_provider_updated(provider: str) -> StyledText:
    """Return styled 'Provider updated' message."""
    return styled_text(
        (StyleName.INFO, "Provider updated: "),
        (StyleName.WARNING, provider),
    )


# Legacy Rich markup versions
PROVIDER_STATUS_READY = "[bold green][Ready][/bold green]"
PROVIDER_STATUS_SETUP_REQUIRED = "[yellow][Setup Required][/yellow]"
PROVIDER_CURRENT = "[bold cyan]<- Current[/bold cyan]"
PROVIDER_NOT_CONFIGURED = (
    "[bold yellow]Provider not configured[/bold yellow]\nCredentials required for {provider}\n"
)
PROVIDER_SELECTION_CANCELLED = (
    "[bold yellow]Provider selection cancelled. Credentials required.[/bold yellow]"
)
PROVIDER_UPDATED = "[bold cyan]Provider updated:[/bold cyan] [yellow]{provider}[/yellow]"

# ============================================================================
# Credentials Configuration
# ============================================================================

# Plain text help messages
OPENAI_CREDENTIALS_HELP_PLAIN = "Get your API key at: https://platform.openai.com/api-keys\n"
AZURE_OPENAI_CREDENTIALS_HELP_PLAIN = "You need credentials from Azure OpenAI service.\n"
ANTHROPIC_CREDENTIALS_HELP_PLAIN = "Get your API key at: https://console.anthropic.com/\n"
VERTEX_CREDENTIALS_HELP_PLAIN = "You need a service account key file from Google Cloud.\n"
OPENROUTER_CREDENTIALS_HELP_PLAIN = "Get your API key at: https://openrouter.ai/keys\n"

# Plain text error messages
CREDENTIALS_ERROR_REQUIRED_PLAIN = "Error: API key is required"
CREDENTIALS_ERROR_ALL_REQUIRED_PLAIN = "Error: All fields are required"
CREDENTIALS_SAVED_PLAIN = "Credentials configured and saved to .env"


def styled_credentials_help(url: str) -> StyledText:
    """Return styled credentials help message."""
    return styled_text((StyleName.DIM, f"Get your API key at: {url}\n"))


def styled_credentials_error(message: str) -> StyledText:
    """Return styled credentials error."""
    return styled_text((StyleName.ERROR, "Error: "), (None, message))


def styled_credentials_saved() -> StyledText:
    """Return styled 'Credentials saved' message."""
    return styled_text((StyleName.SUCCESS, "Credentials configured and saved to .env"))


def styled_credentials_file_not_found(path: str) -> StyledText:
    """Return styled 'File not found' error."""
    return styled_text((StyleName.ERROR, "Error: "), (None, f"File not found: {path}"))


# Legacy Rich markup versions
OPENAI_CREDENTIALS_HELP = "[dim]Get your API key at: https://platform.openai.com/api-keys[/dim]\n"
AZURE_OPENAI_CREDENTIALS_HELP = "[dim]You need credentials from Azure OpenAI service.[/dim]\n"
ANTHROPIC_CREDENTIALS_HELP = "[dim]Get your API key at: https://console.anthropic.com/[/dim]\n"
VERTEX_CREDENTIALS_HELP = "[dim]You need a service account key file from Google Cloud.[/dim]\n"
OPENROUTER_CREDENTIALS_HELP = "[dim]Get your API key at: https://openrouter.ai/keys[/dim]\n"
CREDENTIALS_ERROR_REQUIRED = "[bold red]Error:[/bold red] API key is required"
CREDENTIALS_ERROR_ALL_REQUIRED = "[bold red]Error:[/bold red] All fields are required"
CREDENTIALS_ERROR_FILE_NOT_FOUND = "[bold red]Error:[/bold red] File not found: {path}"
CREDENTIALS_SAVED = "[bold green]Credentials configured and saved to .env[/bold green]"
CREDENTIALS_CONFIG_ERROR = "[bold red]Error:[/bold red] Failed to configure credentials: {error}"

# ============================================================================
# Model Selection
# ============================================================================

MODEL_SELECT_TITLE = "Select Model for {provider}"
MODEL_SELECT_SKIP = "Skip (use default)"

# Plain text versions
MODEL_CURRENT_PLAIN = "<- Current"
MODEL_NOT_AVAILABLE_FRIENDLY = (
    "Model '{model}' is not available or not accessible with your API key."
)
MODEL_NOT_AVAILABLE_WITH_ERROR = "Model '{model}' is not available: {error}"


def styled_model_current() -> StyledText:
    """Return styled 'Current' marker."""
    return styled_text((StyleName.INFO, "<- Current"))


def styled_model_updated(model: str) -> StyledText:
    """Return styled 'Model updated' message."""
    return styled_text(
        (StyleName.INFO, "Model updated: "),
        (StyleName.WARNING, model),
    )


def styled_model_selected(model: str) -> StyledText:
    """Return styled 'Model selected' message."""
    return styled_text(
        (StyleName.INFO, "Model selected: "),
        (StyleName.WARNING, model),
    )


def styled_model_not_available(error: str) -> StyledText:
    """Return styled 'Model not available' error."""
    return styled_text(
        (StyleName.ERROR, "Error: "),
        (None, error),
        (None, "\n"),
        (StyleName.WARNING, "Please select a different model."),
    )


def styled_model_no_predefined(current_model: str) -> StyledText:
    """Return styled 'No predefined models' message."""
    return styled_text(
        (StyleName.WARNING, "No predefined models for this provider.\n"),
        (None, "Current model: "),
        (StyleName.INFO, current_model),
        (None, "\nPlease specify model name in .env file or use CLI flag."),
    )


def styled_model_no_available() -> StyledText:
    """Return styled 'No models available' message."""
    return styled_text(
        (StyleName.WARNING, "No models available for selection. "),
        (None, "Please specify model name in .env file or use CLI flag."),
    )


# Legacy Rich markup versions
MODEL_CURRENT = "[bold cyan]<- Current[/bold cyan]"
MODEL_UPDATED = "[bold cyan]Model updated:[/bold cyan] [yellow]{model}[/yellow]"
MODEL_SELECTED = "[bold cyan]Model selected:[/bold cyan] [yellow]{model}[/yellow]"
MODEL_NOT_AVAILABLE = (
    "[bold red]Error:[/bold red] {error}\n[yellow]Please select a different model.[/yellow]"
)
MODEL_VALIDATION_ERROR = (
    "[bold red]Error:[/bold red] Could not validate model availability\n"
    "[yellow]Note: Could not validate model availability[/yellow]"
)
MODEL_NO_PREDEFINED = (
    "[bold yellow]No predefined models for this provider.[/bold yellow]\n"
    "Current model: [cyan]{current_model_name}[/cyan]\n"
    "Please specify model name in .env file or use CLI flag."
)
MODEL_NO_AVAILABLE = (
    "[bold yellow]No models available for selection. "
    "Please specify model name in .env file or use CLI flag.[/bold yellow]"
)
MODEL_USING_DEFAULT = "[dim]Using default model for this provider[/dim]"

# ============================================================================
# Tool Execution Messages
# ============================================================================

# Plain text versions
TOOL_EXECUTING_PLAIN = "Executing tool: {tool}({args})"
TOOL_DONE_PLAIN = "Tool done: {tool}"
TOOL_ERROR_PLAIN = "Tool failed: {tool} - {error}"


def styled_tool_executing(tool: str, args: str) -> StyledText:
    """Return styled 'Executing tool' message."""
    return styled_text(
        (StyleName.DIM, "Executing tool: "),
        (StyleName.TOOL_NAME, tool),
        (StyleName.TOOL_ARGS, f"({args})"),
    )


def styled_tool_done(tool: str) -> StyledText:
    """Return styled 'Tool done' message."""
    return styled_text(
        (StyleName.DIM, "Tool done: "),
        (StyleName.SUCCESS, tool),
    )


def styled_tool_error(tool: str, error: str) -> StyledText:
    """Return styled 'Tool error' message."""
    return styled_text(
        (StyleName.DIM, "Tool failed: "),
        (StyleName.ERROR, tool),
        (None, f" - {error}"),
    )


# Legacy Rich markup versions
TOOL_EXECUTING = "[dim]Executing tool:[/dim] [yellow]{tool}[/yellow]({args})"
TOOL_DONE = "[dim]Tool:[/dim] [green]{tool}[/green]"
TOOL_ERROR = "[dim]Tool failed:[/dim] [red]{tool}[/red] - {error}"

# ============================================================================
# Progress Messages
# ============================================================================

PROGRESS_PERCENTAGE_PLAIN = "Progress: {percentage:.1f}% - {message}"
PROGRESS_GENERIC_PLAIN = "Progress: {progress} - {message}"


def styled_progress(percentage: float, message: str) -> StyledText:
    """Return styled progress message."""
    return styled_text((StyleName.DIM, f"Progress: {percentage:.1f}% - {message}"))


# Legacy Rich markup versions
PROGRESS_PERCENTAGE = "[dim]Progress: {percentage:.1f}% - {message}[/dim]"
PROGRESS_GENERIC = "[dim]Progress: {progress} - {message}[/dim]"

# ============================================================================
# Checklist Messages
# ============================================================================

CHECKLIST_CREATED_MARKER_PLAIN = "--- Checklist Created ---"
CHECKLIST_CREATED_MARKER = "[dim]--- Checklist Created ---[/dim]"
CHECKLIST_CREATED = "[dim]--- Checklist Created ---[/dim]"

# ============================================================================
# User Interface
# ============================================================================

# Plain text versions
USER_PREFIX_PLAIN = ">"
AGENT_PREFIX_PLAIN = "*"


def styled_user_prefix() -> StyledText:
    """Return styled user prefix."""
    return styled_text((StyleName.BOLD, ">"))


def styled_agent_prefix() -> StyledText:
    """Return styled agent prefix."""
    return styled_text((StyleName.AGENT_PREFIX, "*"))


# Legacy Rich markup versions
USER_PREFIX = "[bold]>[/bold]"
AGENT_PREFIX = "[bold green]*[/bold green]"

# Welcome message (markdown, no Rich markup needed)
WELCOME_MESSAGE = (
    "Hello! I'm **Donkit - RAGOps Agent**, your assistant for building RAG pipelines. "
    "How can I help you today?"
)

EXITING_MESSAGE = "[Exiting REPL]"
EXITING_REPL = "[Exiting REPL]"

# ============================================================================
# Error Messages
# ============================================================================

# Plain text versions
ERROR_PROVIDER_INIT_PLAIN = "Error initializing provider '{provider}': {error}"
ERROR_CREDENTIALS_REQUIRED_PLAIN = "Please ensure credentials are configured correctly."
ERROR_MODEL_SELECTION_CANCELLED_PLAIN = "Model selection cancelled. Exiting."


def styled_error_provider_init(provider: str, error: str) -> StyledText:
    """Return styled provider init error."""
    return styled_text(
        (StyleName.ERROR, f"Error initializing provider '{provider}': "),
        (None, str(error)),
    )


def styled_error_credentials_required() -> StyledText:
    """Return styled credentials required message."""
    return styled_text((StyleName.WARNING, "Please ensure credentials are configured correctly."))


def styled_error_model_selection_cancelled() -> StyledText:
    """Return styled model selection cancelled message."""
    return styled_text((StyleName.ERROR, "Model selection cancelled. Exiting."))


# Legacy Rich markup versions
ERROR_PROVIDER_INIT = "[red]Error initializing provider '{provider}':[/red] {error}"
ERROR_CREDENTIALS_REQUIRED = "[yellow]Please ensure credentials are configured correctly.[/yellow]"
ERROR_MODEL_SELECTION_CANCELLED = "[red]Model selection cancelled. Exiting.[/red]"
ERROR_API_KEY_REQUIRED = "[bold red]Error:[/bold red] API key is required"
ERROR_ALL_FIELDS_REQUIRED = "[bold red]Error:[/bold red] All fields are required"
ERROR_FILE_NOT_FOUND = "[bold red]Error:[/bold red] File not found: {path}"
ERROR_PROVIDER_INIT_FAILED = "[bold red]Error:[/bold red] Failed to initialize provider: {error}"

# ============================================================================
# Credentials Help Messages (Legacy duplicates - kept for compatibility)
# ============================================================================

CREDENTIALS_OPENAI_HELP = "[dim]Get your API key at: https://platform.openai.com/api-keys[/dim]\n"
CREDENTIALS_AZURE_HELP = "[dim]You need credentials from Azure OpenAI service.[/dim]\n"
CREDENTIALS_ANTHROPIC_HELP = "[dim]Get your API key at: https://console.anthropic.com/[/dim]\n"
CREDENTIALS_VERTEX_HELP = "[dim]You need a service account key file from Google Cloud.[/dim]\n"
CREDENTIALS_OPENROUTER_HELP = "[dim]Get your API key at: https://openrouter.ai/keys[/dim]\n"
CREDENTIALS_SAVED_SUCCESS = "[bold green]Credentials configured and saved to .env[/bold green]"

# ============================================================================
# Welcome Messages (Legacy)
# ============================================================================

WELCOME_MESSAGE_RENDERED = "{time} [bold green]RAGOps Agent>[/bold green] {message}"

# ============================================================================
# Thinking/Spinner Messages
# ============================================================================

THINKING_MESSAGE_PLAIN = "Thinking..."
THINKING_MESSAGE = "[dim]Thinking...[/dim]"


def styled_thinking() -> StyledText:
    """Return styled thinking message."""
    return styled_text((StyleName.DIM, "Thinking..."))
