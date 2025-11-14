"""UI text constants for RAGOps Agent CE.

This module contains all user-facing text strings used in the CLI,
organized by feature/section for easy maintenance and localization.
"""

# ============================================================================
# Help and Commands
# ============================================================================

HELP_COMMANDS = [
    "",
    "  [yellow]Available commands:[/yellow]",
    "  [bold]:help[/bold] - Show this help message",
    "  [bold]:q[/bold] or [bold]:quit[/bold] - Exit the agent",
    "  [bold]:clear[/bold] - Clear the conversation transcript",
    "  [bold]:provider[/bold] - Select LLM provider",
    "  [bold]:model[/bold] - Select LLM model",
]

# ============================================================================
# Provider Selection
# ============================================================================

PROVIDER_SELECT_TITLE = "Select LLM Provider"

PROVIDER_STATUS_READY = "[bold green][Ready][/bold green]"
PROVIDER_STATUS_SETUP_REQUIRED = "[yellow][Setup Required][/yellow]"
PROVIDER_CURRENT = "[bold cyan]← Current[/bold cyan]"

PROVIDER_NOT_CONFIGURED = (
    "[bold yellow]⚠ Provider not configured[/bold yellow]\nCredentials required for {provider}\n"
)

PROVIDER_CONFIGURE_PROMPT = "Configure credentials now?"

PROVIDER_SELECTION_CANCELLED = (
    "[bold yellow]Provider selection cancelled. Credentials required.[/bold yellow]"
)

PROVIDER_UPDATED = "[bold cyan]Provider updated:[/bold cyan] [yellow]{provider}[/yellow]"

# ============================================================================
# Credentials Configuration
# ============================================================================

OPENAI_CREDENTIALS_HELP = "[dim]Get your API key at: https://platform.openai.com/api-keys[/dim]\n"

AZURE_OPENAI_CREDENTIALS_HELP = "[dim]You need credentials from Azure OpenAI service.[/dim]\n"

ANTHROPIC_CREDENTIALS_HELP = "[dim]Get your API key at: https://console.anthropic.com/[/dim]\n"

VERTEX_CREDENTIALS_HELP = "[dim]You need a service account key file from Google Cloud.[/dim]\n"

OPENROUTER_CREDENTIALS_HELP = "[dim]Get your API key at: https://openrouter.ai/keys[/dim]\n"

CREDENTIALS_ERROR_REQUIRED = "[bold red]Error:[/bold red] API key is required"
CREDENTIALS_ERROR_ALL_REQUIRED = "[bold red]Error:[/bold red] All fields are required"
CREDENTIALS_ERROR_FILE_NOT_FOUND = "[bold red]Error:[/bold red] File not found: {path}"

CREDENTIALS_SAVED = "[bold green]✓ Credentials configured and saved to .env[/bold green]"

CREDENTIALS_CONFIG_ERROR = "[bold red]Error:[/bold red] Failed to configure credentials: {error}"

# ============================================================================
# Model Selection
# ============================================================================

MODEL_SELECT_TITLE = "Select Model for {provider}"
MODEL_SELECT_SKIP = "Skip (use default)"
MODEL_CURRENT = "[bold cyan]← Current[/bold cyan]"

MODEL_UPDATED = "[bold cyan]Model updated:[/bold cyan] [yellow]{model}[/yellow]"
MODEL_SELECTED = "[bold cyan]Model selected:[/bold cyan] [yellow]{model}[/yellow]"

MODEL_NOT_AVAILABLE = (
    "[bold red]Error:[/bold red] {error}\n[yellow]Please select a different model.[/yellow]"
)

MODEL_NOT_AVAILABLE_FRIENDLY = (
    "Model '{model}' is not available or not accessible with your API key."
)

MODEL_NOT_AVAILABLE_WITH_ERROR = "Model '{model}' is not available: {error}"

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

# ============================================================================
# Tool Execution Messages
# ============================================================================

TOOL_EXECUTING = "[dim]🔧 Executing tool:[/dim] [yellow]{tool}[/yellow]({args})"
TOOL_DONE = "[dim]✓ Tool:[/dim] [green]{tool}[/green]"
TOOL_ERROR = "[dim]✗ Tool failed:[/dim] [red]{tool}[/red] - {error}"

# ============================================================================
# Progress Messages
# ============================================================================

PROGRESS_PERCENTAGE = "[dim]⏳ Progress: {percentage:.1f}% - {message}[/dim]"
PROGRESS_GENERIC = "[dim]⏳ Progress: {progress} - {message}[/dim]"

# ============================================================================
# Checklist Messages
# ============================================================================

CHECKLIST_CREATED_MARKER = "[dim]--- Checklist Created ---[/dim]"

# ============================================================================
# User Interface
# ============================================================================

USER_PREFIX = "[bold blue]you>[/bold blue]"
AGENT_PREFIX = "[bold green]RAGOps Agent>[/bold green]"

WELCOME_MESSAGE = (
    "Hello! I'm **Donkit - RAGOps Agent**, your assistant for building RAG pipelines. "
    "How can I help you today?"
)

EXITING_MESSAGE = "[Exiting REPL]"

# ============================================================================
# Error Messages
# ============================================================================

ERROR_PROVIDER_INIT = "[red]Error initializing provider '{provider}':[/red] {error}"

ERROR_CREDENTIALS_REQUIRED = "[yellow]Please ensure credentials are configured correctly.[/yellow]"

ERROR_MODEL_SELECTION_CANCELLED = "[red]Model selection cancelled. Exiting.[/red]"

ERROR_API_KEY_REQUIRED = "[bold red]Error:[/bold red] API key is required"
ERROR_ALL_FIELDS_REQUIRED = "[bold red]Error:[/bold red] All fields are required"
ERROR_FILE_NOT_FOUND = "[bold red]Error:[/bold red] File not found: {path}"
ERROR_PROVIDER_INIT_FAILED = "[bold red]Error:[/bold red] Failed to initialize provider: {error}"

# ============================================================================
# Credentials Help Messages
# ============================================================================

CREDENTIALS_OPENAI_HELP = "[dim]Get your API key at: https://platform.openai.com/api-keys[/dim]\n"

CREDENTIALS_AZURE_HELP = "[dim]You need credentials from Azure OpenAI service.[/dim]\n"

CREDENTIALS_ANTHROPIC_HELP = "[dim]Get your API key at: https://console.anthropic.com/[/dim]\n"

CREDENTIALS_VERTEX_HELP = "[dim]You need a service account key file from Google Cloud.[/dim]\n"

CREDENTIALS_OPENROUTER_HELP = "[dim]Get your API key at: https://openrouter.ai/keys[/dim]\n"

CREDENTIALS_SAVED_SUCCESS = "[bold green]✓ Credentials configured and saved to .env[/bold green]"

# ============================================================================
# Checklist Messages
# ============================================================================

CHECKLIST_CREATED = "[dim]--- Checklist Created ---[/dim]"

# ============================================================================
# Exit Messages
# ============================================================================

EXITING_REPL = "[Exiting REPL]"

# ============================================================================
# Welcome Messages
# ============================================================================

WELCOME_MESSAGE_RENDERED = "{time} [bold green]RAGOps Agent>[/bold green] {message}"

# ============================================================================
# Model Selection Messages
# ============================================================================

MODEL_USING_DEFAULT = "[dim]Using default model for this provider[/dim]"
