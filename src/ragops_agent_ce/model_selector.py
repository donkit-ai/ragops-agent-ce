"""
Model selection module for RagOps Agent CE.

Provides model selection screen at startup with visual indicators
for configured credentials and automatic resume of latest selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ragops_agent_ce.credential_checker import check_provider_credentials
from ragops_agent_ce.db import kv_get
from ragops_agent_ce.db import kv_set
from ragops_agent_ce.db import open_db

if TYPE_CHECKING:
    pass

# KV storage key for latest model selection
LATEST_MODEL_KEY = "latest_model"

# Provider definitions matching setup_wizard
PROVIDERS = {
    "vertex": {
        "display": "Vertex AI (Google Cloud)",
        "description": "Google's Gemini models via Vertex AI",
    },
    "openai": {
        "display": "OpenAI",
        "description": "ChatGPT API and compatible providers",
    },
    "azure_openai": {
        "display": "Azure OpenAI",
        "description": "OpenAI models via Azure",
    },
    "ollama": {
        "display": "Ollama (Local)",
        "description": "Local LLM server (OpenAI-compatible)",
    },
    "openrouter": {
        "display": "OpenRouter",
        "description": "Access 100+ models via OpenRouter API",
    },
}


def get_latest_model_selection() -> tuple[str, str | None] | None:
    """
    Retrieve the latest model selection from KV database.

    Returns:
        Tuple of (provider, model) or None if no selection found
    """
    db = open_db()
    try:
        latest_str = kv_get(db, LATEST_MODEL_KEY)
        if not latest_str:
            return None

        data = json.loads(latest_str)
        provider = data.get("provider")
        model = data.get("model")
        return (provider, model) if provider else None
    except (json.JSONDecodeError, KeyError):
        return None
    finally:
        # Properly close database connection
        if hasattr(db, "_engine") and db._engine:
            db._engine.dispose()


def save_model_selection(provider: str, model: str | None = None) -> None:
    """
    Save model selection to KV database.

    Args:
        provider: Provider name
        model: Optional model name
    """
    db = open_db()
    try:
        data = {"provider": provider, "model": model}
        kv_set(db, LATEST_MODEL_KEY, json.dumps(data))
    finally:
        # Properly close database connection
        if hasattr(db, "_engine") and db._engine:
            db._engine.dispose()


def select_model_at_startup(
    env_path: Path | None = None, max_retries: int = 3
) -> tuple[str, str | None] | None:
    """
    Show model selection screen at startup.

    Displays all available providers with visual indicators:
    - ✓ (green) for providers with configured credentials
    - ⚠ (yellow) for providers without credentials
    - Highlights latest selected model

    Args:
        env_path: Optional path to .env file
        max_retries: Maximum number of configuration retry attempts

    Returns:
        Tuple of (provider, model) or None if cancelled
    """
    from rich.console import Console
    from rich.text import Text

    from ragops_agent_ce.interactive_input import interactive_select

    console = Console()
    env_path = env_path or Path.cwd() / ".env"

    retry_count = 0
    while retry_count < max_retries:
        # Get latest selection
        latest_selection = get_latest_model_selection()
        latest_provider = latest_selection[0] if latest_selection else None

        # Build provider list with indicators
        choices = []
        provider_map = {}  # Map choice index to provider name

        # Sort providers: latest first, then alphabetically by display name
        sorted_providers = []
        other_providers = []

        for provider, info in PROVIDERS.items():
            if provider == latest_provider:
                # Latest provider goes first
                sorted_providers.insert(0, (provider, info))
            else:
                other_providers.append((provider, info))

        # Sort other providers alphabetically by display name
        other_providers.sort(key=lambda x: x[1]["display"])
        sorted_providers.extend(other_providers)

        configured_count = 0
        for idx, (provider, info) in enumerate(sorted_providers):
            has_creds = check_provider_credentials(provider, env_path)
            if has_creds:
                configured_count += 1

            # Build choice string with visual indicators
            choice_text = Text()

            # Status indicator with better styling
            if has_creds:
                choice_text.append("✓", style="bold green")
            else:
                choice_text.append("⚠", style="bold yellow")

            # Spacing
            choice_text.append("  ", style="")

            # Add provider display name with better styling
            if has_creds:
                choice_text.append(info["display"], style="bold green")
            else:
                choice_text.append(info["display"], style="white")

            choice_text.append(" ", style="")

            # Status badge
            if has_creds:
                choice_text.append("[Ready]", style="bold green")
            else:
                choice_text.append("[Setup Required]", style="yellow")

            # Mark latest selection
            if provider == latest_provider:
                choice_text.append(" ", style="")
                choice_text.append("← Last used", style="bold cyan")

            choices.append(choice_text.markup)
            provider_map[idx] = provider

        # Build title with configured count and visual styling
        title = f"🚀 Select LLM Model Provider · {configured_count}/{len(PROVIDERS)} Ready"

        console.print()
        # Default to first item (0) which is the latest selection if it exists
        default_index = 0 if latest_provider else 0
        selected_choice = interactive_select(choices, title=title, default_index=default_index)

        if selected_choice is None:
            return None

        # Find selected provider
        selected_idx = choices.index(selected_choice)
        selected_provider = provider_map[selected_idx]

        # Check if selected provider has credentials configured
        has_creds = check_provider_credentials(selected_provider, env_path)
        if not has_creds:
            provider_display = PROVIDERS[selected_provider]["display"]
            console.print(f"\n[yellow]⚠ {provider_display} is not configured.[/yellow]")
            console.print("[dim]Credentials are required for this provider.[/dim]\n")

            # Ask if user wants to configure now
            from ragops_agent_ce.interactive_input import interactive_confirm

            configure_now = interactive_confirm("Configure credentials now?", default=True)

            if configure_now:
                # Run setup wizard for this specific provider
                from ragops_agent_ce.setup_wizard import SetupWizard

                wizard = SetupWizard(env_path)

                # Set provider in config and run configuration
                wizard.config["RAGOPS_LLM_PROVIDER"] = selected_provider
                if wizard._configure_provider(selected_provider):
                    # Save configuration
                    if wizard._save_config():
                        provider_display = PROVIDERS[selected_provider]["display"]
                        console.print(f"\n✓ {provider_display} configured!\n")
                        # Re-check credentials after configuration
                        has_creds = check_provider_credentials(selected_provider, env_path)
                        if not has_creds:
                            console.print(
                                "\n[red]Configuration saved but credentials check " "failed.[/red]"
                            )
                            retry_count += 1
                            if retry_count >= max_retries:
                                console.print(
                                    f"[red]Maximum retry attempts ({max_retries}) " "reached.[/red]"
                                )
                                return None
                            console.print("[yellow]Please try again.[/yellow]\n")
                            continue
                    else:
                        console.print("\n[red]Error saving configuration.[/red]")
                        retry_count += 1
                        if retry_count >= max_retries:
                            console.print(
                                f"[red]Maximum retry attempts ({max_retries}) " "reached.[/red]"
                            )
                            return None
                        console.print("[yellow]Please try again.[/yellow]\n")
                        continue
                else:
                    console.print("\n[red]Configuration cancelled or failed.[/red]")
                    retry_count += 1
                    if retry_count >= max_retries:
                        console.print(
                            f"[red]Maximum retry attempts ({max_retries}) " "reached.[/red]"
                        )
                        return None
                    console.print("[yellow]Please try again.[/yellow]\n")
                    continue
            else:
                # User chose "No" - return to model selection screen
                console.print()
                retry_count += 1
                if retry_count >= max_retries:
                    console.print(
                        f"[yellow]Maximum retry attempts ({max_retries}) " "reached.[/yellow]"
                    )
                    return None
                continue

        # If we get here, provider has credentials configured
        # For now, we don't ask for specific model - just return provider
        # Model can be specified later via CLI flags or in .env
        model = None

        # Save selection
        save_model_selection(selected_provider, model)

        console.print(f"\n✓ Selected: [green]{PROVIDERS[selected_provider]['display']}[/green]\n")

        return (selected_provider, model)

    # Should not reach here, but added for safety
    return None
