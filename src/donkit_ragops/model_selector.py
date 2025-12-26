"""
Model selection module for RAGOps Agent CE.

Provides model selection screen at startup with visual indicators
for configured credentials and automatic resume of latest selection.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from donkit.llm import GenerateRequest, Message
from rich.text import Text

from donkit_ragops.config import load_settings
from donkit_ragops.credential_checker import check_provider_credentials
from donkit_ragops.db import kv_get, kv_set, open_db
from donkit_ragops.interactive_input import interactive_select
from donkit_ragops.llm.provider_factory import get_provider
from donkit_ragops.supported_models import SUPPORTED_MODELS
from donkit_ragops.ui import get_ui
from donkit_ragops.ui.styles import StyleName, styled_text

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
    "donkit": {
        "display": "Donkit",
        "description": "Donkit default model",
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

    ui = get_ui()
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
        title = f"Select LLM Model Provider ({configured_count}/{len(PROVIDERS)} Ready)"

        ui.newline()
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
            ui.newline()
            ui.print_warning(f"{provider_display} is not configured.")
            ui.print("Credentials are required for this provider.", StyleName.DIM)
            ui.newline()

            # Ask if user wants to configure now
            from donkit_ragops.interactive_input import interactive_confirm

            configure_now = interactive_confirm("Configure credentials now?", default=True)

            if configure_now:
                # Run setup wizard for this specific provider
                from donkit_ragops.setup_wizard import SetupWizard

                wizard = SetupWizard(env_path)

                # Set provider in config and run configuration
                wizard.config["RAGOPS_LLM_PROVIDER"] = selected_provider
                if wizard.configure_provider(selected_provider):
                    # Save configuration
                    if wizard.save_config():
                        provider_display = PROVIDERS[selected_provider]["display"]
                        ui.newline()
                        ui.print_success(f"{provider_display} configured!")
                        ui.newline()
                        # Re-check credentials after configuration
                        has_creds = check_provider_credentials(selected_provider, env_path)
                        if not has_creds:
                            ui.newline()
                            ui.print_error("Configuration saved but credentials check failed.")
                            retry_count += 1
                            if retry_count >= max_retries:
                                ui.print_error(f"Maximum retry attempts ({max_retries}) reached.")
                                return None
                            ui.print_warning("Please try again.")
                            ui.newline()
                            continue
                    else:
                        ui.newline()
                        ui.print_error("Error saving configuration.")
                        retry_count += 1
                        if retry_count >= max_retries:
                            ui.print_error(f"Maximum retry attempts ({max_retries}) reached.")
                            return None
                        ui.print_warning("Please try again.")
                        ui.newline()
                        continue
                else:
                    ui.newline()
                    ui.print_error("Configuration cancelled or failed.")
                    retry_count += 1
                    if retry_count >= max_retries:
                        ui.print_error(f"Maximum retry attempts ({max_retries}) reached.")
                        return None
                    ui.print_warning("Please try again.")
                    ui.newline()
                    continue
            else:
                # User chose "No" - return to model selection screen
                ui.newline()
                retry_count += 1
                if retry_count >= max_retries:
                    ui.print_warning(f"Maximum retry attempts ({max_retries}) reached.")
                    return None
                continue

        # If we get here, provider has credentials configured
        # Now ask for model selection
        ui.newline()
        ui.print_styled(
            styled_text(
                (None, "Selected: "),
                (StyleName.SUCCESS, PROVIDERS[selected_provider]["display"]),
            )
        )
        ui.newline()

        # Try to get models from provider
        models = []
        try:
            settings = load_settings()
            # Temporarily set provider in settings for model listing
            temp_settings = settings.model_copy(update={"llm_provider": selected_provider})
            provider_instance = get_provider(temp_settings, llm_provider=selected_provider)

            if hasattr(provider_instance, "list_chat_models"):
                models = provider_instance.list_chat_models()
            elif hasattr(provider_instance, "list_models"):
                models = provider_instance.list_models()
        except Exception:
            # If we can't get provider instance, use fallback
            ui.print(
                "Note: Could not fetch models from provider API (using fallback list)",
                StyleName.DIM,
            )
            ui.newline()

        # Apply supported models filter logic
        supported = SUPPORTED_MODELS.get(selected_provider, [])

        if models:
            if supported:
                # Intersection: Keep supported models that exist in fetched list
                models = [m for m in supported if m in models]
        else:
            # Fallback if fetch failed or returned empty
            models = supported

        # Get latest model selection for this provider
        latest_model = None
        if latest_selection and latest_selection[0] == selected_provider:
            latest_model = latest_selection[1]

        model = None
        if models:
            # Build choices list
            choices = []
            for model_name in models:
                choice = model_name
                if model_name == latest_model:
                    choice += " [bold cyan]← Last used[/bold cyan]"
                choices.append(choice)

            # Add "Skip" option
            choices.append("Skip (use default)")

            title = f"Select Model for {PROVIDERS[selected_provider]['display']}"
            # Set default to latest model if available
            default_index = None
            if latest_model and latest_model in models:
                default_index = models.index(latest_model)
            selected_model_choice = interactive_select(
                choices, title=title, default_index=default_index
            )

            if selected_model_choice and selected_model_choice != "Skip (use default)":
                # Extract model name (remove "← Last used" if present)
                model = selected_model_choice.split(" [")[0].strip()

                # Validate model by trying to use it
                try:
                    # Test if model is actually available
                    test_messages = [Message(role="user", content="test")]
                    request = GenerateRequest(messages=test_messages)
                    try:
                        asyncio.run(provider_instance.generate(request))
                        # If successful, model is available
                        ui.print_styled(
                            styled_text(
                                (None, "Model selected: "),
                                (StyleName.SUCCESS, model),
                            )
                        )
                        ui.newline()
                    except Exception as model_error:
                        # Model is not available
                        error_msg = str(model_error)
                        if "model" in error_msg.lower() and (
                            "not found" in error_msg.lower()
                            or "does not exist" in error_msg.lower()
                            or "not available" in error_msg.lower()
                        ):
                            friendly_msg = (
                                f"Model '{model}' is not available or not accessible "
                                "with your API key."
                            )
                        else:
                            friendly_msg = f"Model '{model}' is not available: {error_msg}"
                        ui.print_error(friendly_msg)

                        # Ask if user wants to force use the model anyway
                        from donkit_ragops.interactive_input import interactive_confirm

                        if interactive_confirm("Use this model anyway?", default=False):
                            ui.print_styled(
                                styled_text(
                                    (None, "Model selected (forced): "),
                                    (StyleName.WARNING, model),
                                )
                            )
                            ui.newline()
                        else:
                            ui.print_warning("Please select a different model.")
                            ui.newline()
                            # Ask user to select again
                            retry_model = interactive_select(
                                choices, title=title, default_index=default_index
                            )
                            if retry_model and retry_model != "Skip (use default)":
                                model = retry_model.split(" [")[0].strip()
                                # Try validation again (but don't loop forever)
                                try:
                                    provider_instance.generate(
                                        [Message(role="user", content="test")],
                                        model=model,
                                        max_tokens=1,
                                    )
                                    ui.print_styled(
                                        styled_text(
                                            (None, "Model selected: "),
                                            (StyleName.SUCCESS, model),
                                        )
                                    )
                                    ui.newline()
                                except Exception:
                                    # If still fails, set it anyway but warn
                                    ui.print_warning(
                                        "Model validation failed, but it will be set anyway."
                                    )
                                    ui.newline()
                                    # Keep the model even if validation fails on retry
                            else:
                                model = None  # User skipped or cancelled
                except Exception as e:
                    # If validation itself fails, still set the model but warn
                    ui.print_warning(f"Could not validate model '{model}': {e}")
                    ui.print(
                        f"Model '{model}' will be set, but may not be available.",
                        StyleName.DIM,
                    )
                    ui.newline()
                    model = None  # Set to None to be safe

            elif selected_model_choice == "Skip (use default)":
                ui.print("Using default model for this provider", StyleName.DIM)
                ui.newline()
                model = None
            else:
                # Cancelled - use None model
                model = None
        else:
            ui.print_warning(
                "No models available for selection. "
                "Model can be specified later via CLI flags or in .env file."
            )
            ui.newline()

        # Save selection
        save_model_selection(selected_provider, model)

        return (selected_provider, model)

    # Should not reach here, but added for safety
    return None
