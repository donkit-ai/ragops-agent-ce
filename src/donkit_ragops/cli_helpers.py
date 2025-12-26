"""Helper functions for CLI operations."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from donkit.llm import GenerateRequest, LLMModelAbstract, Message
from loguru import logger

from donkit_ragops import texts
from donkit_ragops.agent import AgentTool
from donkit_ragops.agent.agent import LLMAgent
from donkit_ragops.config import Settings, load_settings
from donkit_ragops.credential_checker import check_provider_credentials
from donkit_ragops.interactive_input import interactive_confirm, interactive_select
from donkit_ragops.llm.provider_factory import get_provider
from donkit_ragops.mcp.client import MCPClient
from donkit_ragops.model_selector import PROVIDERS, save_model_selection
from donkit_ragops.schemas.agent_schemas import AgentSettings
from donkit_ragops.setup_wizard import SetupWizard
from donkit_ragops.supported_models import SUPPORTED_MODELS
from donkit_ragops.ui import get_ui
from donkit_ragops.ui.styles import StyleName, styled_text


def configure_provider_credentials(
    provider: str, env_path: Path | None = None
) -> dict[str, str] | None:
    """
    Configure credentials for a provider using SetupWizard logic.

    Args:
        provider: Provider name (openai, vertex, azure_openai, etc.)
        env_path: Path to .env file (defaults to cwd/.env)

    Returns:
        Dictionary with configuration keys/values, or None if cancelled/failed
    """
    wizard = SetupWizard(env_path=env_path)

    # Call the appropriate configuration method
    success = wizard.configure_provider(provider)

    if not success:
        return None

    # Return the collected config without saving
    return wizard.config


def save_provider_config(config: dict[str, str], env_path: Path | None = None) -> bool:
    """
    Save provider configuration to .env file using SetupWizard logic.

    Args:
        config: Configuration dictionary to save
        env_path: Path to .env file (defaults to cwd/.env)

    Returns:
        True if saved successfully, False otherwise
    """
    wizard = SetupWizard(env_path=env_path)
    wizard.config = config

    # Use wizard's save logic (without showing summary)
    return wizard.save_config()


def get_available_models(prov: object, provider_key: str) -> list[str]:
    """
    Collect available chat models for provider.

    If SUPPORTED_MODELS has entries for this provider, return the intersection
    of fetched models and supported models (filtering out unsupported ones).

    If fetched list is empty or fails, fallback to supported models.
    If supported list is empty (e.g. Ollama), return all fetched models.
    """
    supported = SUPPORTED_MODELS.get(provider_key, [])
    fetched_models: list[str] = []

    if prov is not None:
        try:
            if hasattr(prov, "list_chat_models"):
                fetched_models = prov.list_chat_models()
            elif hasattr(prov, "list_models"):
                fetched_models = prov.list_models()
        except Exception as exc:
            logger.warning(f"Failed to list models for provider '{provider_key}': {exc}")

    if fetched_models:
        if supported:
            # Intersection: Keep supported models that exist in fetched list
            # Use order from supported list
            return [m for m in supported if m in fetched_models]
        else:
            # No whitelist -> return all
            return fetched_models

    # Fallback to supported if fetch failed
    return supported


def format_model_choices(models: Iterable[str], current_model: str | None) -> list[str]:
    """Attach current-model marker to a list of model names."""
    choices: list[str] = []
    for model_name in models:
        choice = model_name
        if current_model and model_name == current_model:
            choice += f" {texts.MODEL_CURRENT}"
        choices.append(choice)
    return choices


def validate_model_choice(
    prov: LLMModelAbstract,
    provider_key: str,
    model_name: str,
    agent_settings: AgentSettings,
) -> tuple[bool, list[str]]:
    """
    Validate selected model by issuing a lightweight generation request.

    Returns tuple (success, transcript_messages).
    """
    messages: list[str] = []

    try:
        test_messages = [Message(role="user", content="test")]
        try:
            request = GenerateRequest(messages=test_messages, max_tokens=1)
            prov.generate(request)
            agent_settings.model = model_name
            save_model_selection(provider_key, model_name)
            messages.append(texts.MODEL_SELECTED.format(model=model_name))
            return True, messages
        except Exception as model_error:  # noqa: BLE001
            error_msg = str(model_error)
            if "model" in error_msg.lower() and (
                "not found" in error_msg.lower()
                or "does not exist" in error_msg.lower()
                or "not available" in error_msg.lower()
            ):
                friendly_msg = texts.MODEL_NOT_AVAILABLE_FRIENDLY.format(model=model_name)
            else:
                friendly_msg = texts.MODEL_NOT_AVAILABLE_WITH_ERROR.format(
                    model=model_name, error=error_msg
                )
            messages.append(texts.MODEL_NOT_AVAILABLE.format(error=friendly_msg))
            return False, messages
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Model validation failed for '{model_name}': {exc}")
        agent_settings.model = model_name
        save_model_selection(provider_key, model_name)
        messages.append(
            f"{texts.MODEL_UPDATED.format(model=model_name)}\n{texts.MODEL_VALIDATION_ERROR}"
        )
        return True, messages


@dataclass
class ProviderSelectionResult:
    provider: str | None
    prov: LLMModelAbstract | None
    agent: LLMAgent | None
    settings: Settings
    agent_settings: AgentSettings
    model: str | None
    messages: list[str]
    prompt_model_selection: bool
    cancelled: bool = False


def _build_provider_choices(
    current_provider: str | None, env_path: Path
) -> tuple[list[str], dict[int, str]]:
    choices: list[str] = []
    provider_map: dict[int, str] = {}

    for idx, (prov_key, prov_info) in enumerate(PROVIDERS.items()):
        has_creds = check_provider_credentials(prov_key, env_path)
        choice_text = ""
        if has_creds:
            choice_text += "[bold green]✓[/bold green] "
        else:
            choice_text += "[bold yellow]⚠[/bold yellow] "
        choice_text += prov_info["display"]
        if has_creds:
            choice_text += f" {texts.PROVIDER_STATUS_READY}"
        else:
            choice_text += f" {texts.PROVIDER_STATUS_SETUP_REQUIRED}"
        if prov_key == current_provider:
            choice_text += f" {texts.PROVIDER_CURRENT}"

        choices.append(choice_text)
        provider_map[idx] = prov_key

    return choices, provider_map


async def select_provider_interactively(
    *,
    current_provider: str | None,
    current_model: str | None,
    settings: Settings,
    agent_settings: AgentSettings,
    prov: LLMModelAbstract | None,
    agent: LLMAgent | None,
    tools: list[AgentTool],
    mcp_clients: list[MCPClient],
    env_path: Path | None = None,
) -> ProviderSelectionResult | None:
    """Interactive provider selection with credential configuration support."""

    ui = get_ui()
    env_path = env_path or Path.cwd() / ".env"
    choices, provider_map = _build_provider_choices(current_provider, env_path)
    selected_choice = interactive_select(choices, title=texts.PROVIDER_SELECT_TITLE)

    if selected_choice is None:
        return ProviderSelectionResult(
            provider=current_provider,
            prov=prov,
            agent=agent,
            settings=settings,
            agent_settings=agent_settings,
            model=current_model,
            messages=[],
            prompt_model_selection=False,
            cancelled=True,
        )

    selected_idx = choices.index(selected_choice)
    new_provider = provider_map[selected_idx]

    messages: list[str] = []
    has_creds = check_provider_credentials(new_provider, env_path)

    updated_settings = settings
    if not has_creds:
        provider_display = PROVIDERS[new_provider]["display"]
        ui.newline()
        ui.print_styled(
            styled_text(
                (StyleName.WARNING, texts.PROVIDER_NOT_CONFIGURED.format(provider=provider_display))
            )
        )

        configure_now = interactive_confirm(texts.PROVIDER_CONFIGURE_PROMPT, default=True)
        if not configure_now:
            messages.append(texts.PROVIDER_SELECTION_CANCELLED)
            return ProviderSelectionResult(
                provider=current_provider,
                prov=prov,
                agent=agent,
                settings=settings,
                agent_settings=agent_settings,
                model=current_model,
                messages=messages,
                prompt_model_selection=False,
                cancelled=True,
            )

        config = configure_provider_credentials(new_provider, env_path=env_path)
        if config is None:
            messages.append(texts.PROVIDER_SELECTION_CANCELLED)
            return ProviderSelectionResult(
                provider=current_provider,
                prov=prov,
                agent=agent,
                settings=settings,
                agent_settings=agent_settings,
                model=current_model,
                messages=messages,
                prompt_model_selection=False,
                cancelled=True,
            )

        config["RAGOPS_LLM_PROVIDER"] = new_provider
        if save_provider_config(config, env_path=env_path):
            messages.append(texts.CREDENTIALS_SAVED)
        else:
            messages.append(
                texts.CREDENTIALS_CONFIG_ERROR.format(error="Failed to save configuration")
            )
            return ProviderSelectionResult(
                provider=current_provider,
                prov=prov,
                agent=agent,
                settings=settings,
                agent_settings=agent_settings,
                model=current_model,
                messages=messages,
                prompt_model_selection=False,
                cancelled=True,
            )

        updated_settings = load_settings()
    else:
        # Reload settings to ensure provider configuration is synced
        updated_settings = load_settings()

    try:
        new_settings = updated_settings.model_copy(update={"llm_provider": new_provider})
        new_prov = get_provider(new_settings, llm_provider=new_provider)
    except Exception as exc:  # noqa: BLE001
        messages.append(texts.ERROR_PROVIDER_INIT_FAILED.format(error=str(exc)))
        return ProviderSelectionResult(
            provider=current_provider,
            prov=prov,
            agent=agent,
            settings=settings,
            agent_settings=agent_settings,
            model=current_model,
            messages=messages,
            prompt_model_selection=False,
            cancelled=True,
        )

    new_agent = LLMAgent(new_prov, tools=tools, mcp_clients=mcp_clients)
    await new_agent.ainit_mcp_tools()

    agent_settings.llm_provider = new_prov
    agent_settings.model = None

    os.environ["RAGOPS_LLM_PROVIDER"] = new_provider

    messages.append(texts.PROVIDER_UPDATED.format(provider=PROVIDERS[new_provider]["display"]))

    return ProviderSelectionResult(
        provider=new_provider,
        prov=new_prov,
        agent=new_agent,
        settings=new_settings,
        agent_settings=agent_settings,
        model=None,
        messages=messages,
        prompt_model_selection=True,
    )
