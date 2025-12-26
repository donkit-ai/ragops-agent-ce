"""
Interactive setup wizard for first-time configuration.
Works without LLM - pure hardcoded logic for collecting user settings.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
from loguru import logger
from rich.prompt import Confirm, Prompt

from donkit_ragops.credential_checker import check_provider_credentials
from donkit_ragops.interactive_input import interactive_confirm, interactive_select
from donkit_ragops.ui import get_ui
from donkit_ragops.ui.styles import StyleName, styled_text


class SetupWizard:
    """Interactive setup wizard for configuring RAGOps Agent CE."""

    def __init__(self, env_path: Path | None = None):
        self.env_path = env_path or Path.cwd() / ".env"
        self.config: dict[str, str] = {}

    def run(self) -> bool:
        """Run the setup wizard. Returns True if setup completed successfully."""
        ui = get_ui()
        ui.clear()
        self._show_welcome()

        # Step 1: Choose LLM provider
        provider = self._choose_provider()
        if not provider:
            return False

        self.config["RAGOPS_LLM_PROVIDER"] = provider

        # Step 2: Configure provider credentials
        if not self.configure_provider(provider):
            return False

        # Step 3: Optional settings
        self._configure_optional_settings()

        # Step 4: Save configuration
        return self.save_config()

    def _show_welcome(self) -> None:
        """Show welcome message."""
        ui = get_ui()

        # Build welcome content
        welcome_lines = [
            styled_text(
                (None, "Welcome to "),
                (StyleName.INFO, "RAGOps Agent CE"),
                (None, " Setup Wizard!"),
            ),
            styled_text((None, "")),
            styled_text(
                (StyleName.INFO, "IMPORTANT: "),
                (StyleName.INFO, "Run the agent from a new, empty directory!"),
            ),
            styled_text(
                (StyleName.DIM, "The agent will create project files, .env, and other artifacts.")
            ),
            styled_text((StyleName.DIM, "Recommended:")),
            styled_text((StyleName.SUCCESS, "  mkdir ~/ragops-workspace && cd ~/ragops-workspace")),
            styled_text((None, "")),
            styled_text(
                (StyleName.DIM, "This wizard will help you configure the agent for first use.")
            ),
            styled_text((StyleName.DIM, "You'll need an API key for your chosen LLM provider.")),
            styled_text((None, "")),
            styled_text(
                (StyleName.SUCCESS, "Supported: "),
                (StyleName.SUCCESS, "Vertex AI, OpenAI, Azure OpenAI, Ollama, OpenRouter"),
            ),
            styled_text((StyleName.DIM, "More providers are coming soon!")),
        ]

        ui.print_panel(welcome_lines, title="Setup", border_style=StyleName.INFO)
        ui.newline()

    def _choose_provider(self, use_case: str = "chat") -> str | None:
        """Let user choose LLM provider."""
        ui = get_ui()
        if use_case == "embeddings":
            ui.print_styled(
                styled_text(
                    (StyleName.BOLD, "Step 1: "),
                    (None, "Choose your embeddings provider"),
                )
            )
            ui.newline()
        else:
            ui.print_styled(
                styled_text(
                    (StyleName.BOLD, "Step 1: "),
                    (None, "Choose your LLM provider"),
                )
            )
            ui.newline()
        providers = {
            "1": {
                "name": "vertex",
                "display": "Vertex AI (Google Cloud)",
                "description": "Google's Gemini models via Vertex AI",
                "available": True,
                "has_embeddings": "default",
            },
            "2": {
                "name": "openai",
                "display": "OpenAI",
                "description": "ChatGPT API and compatible providers",
                "available": True,
                "has_embeddings": "default",
            },
            "3": {
                "name": "azure_openai",
                "display": "Azure OpenAI",
                "description": "OpenAI models via Azure",
                "available": True,
                "has_embeddings": "custom",
            },
            "4": {
                "name": "ollama",
                "display": "Ollama (Local)",
                "description": "Local LLM server (OpenAI-compatible)",
                "available": True,
                "has_embeddings": False,
            },
            "5": {
                "name": "donkit",
                "display": "Donkit",
                "description": "Donkit default models via Donkit API",
                "available": True,
            },
        }
        if use_case == "chat":
            providers.update(
                {
                    "6": {
                        "name": "openrouter",
                        "display": "OpenRouter",
                        "description": "Access 100+ models via OpenRouter API (chat models only, "
                        "needs separate embeddings provider)",
                        "available": True,
                        "has_embeddings": False,
                    },
                }
            )
        # Build list of available choices for interactive selection
        available_providers = [(key, info) for key, info in providers.items() if info["available"]]
        choices = [
            f"{info['display']} - {info['description']}" for key, info in available_providers
        ]

        # Use interactive selection
        if use_case == "embeddings":
            selected = interactive_select(choices=choices, title="Choose your embeddings provider")
        else:
            selected = interactive_select(choices=choices, title="Choose your LLM provider")
        if selected is None:
            ui.print_error("Setup cancelled")
            return None

        # Find the selected provider by matching the choice
        selected_idx = choices.index(selected)
        provider_key = available_providers[selected_idx][0]
        provider = providers[provider_key]["name"]

        ui.newline()
        ui.print_styled(
            styled_text(
                (None, "Selected: "),
                (StyleName.SUCCESS, providers[provider_key]["display"]),
            )
        )
        ui.newline()
        return provider

    def configure_provider(self, provider: str, use_case: str = "chat") -> bool:
        """Configure credentials for chosen provider."""
        ui = get_ui()
        ui.print_styled(
            styled_text(
                (StyleName.BOLD, "Step 2: "),
                (None, f"Configure {provider} credentials"),
            )
        )
        ui.newline()
        match provider:
            case "vertex":
                return self._configure_vertex()
            case "openai":
                return self._configure_openai(use_case)
            case "azure_openai":
                return self._configure_azure_openai(use_case)
            case "anthropic":
                return self._configure_anthropic()
            case "ollama":
                return self._configure_ollama(use_case)
            case "openrouter":
                return self._configure_openrouter()
            case "donkit":
                return self._configure_donkit()
            case _:
                ui.print_error(f"Unknown provider: {provider}")
                return False

    def _configure_donkit(self):
        """Configure Donkit credentials."""
        ui = get_ui()
        ui.print("Get your API key at: https://donkit.ai/api", StyleName.DIM)
        ui.newline()
        api_key = Prompt.ask("Enter Donkit API key", password=True)
        if not api_key:
            ui.print_error("API key is required")
            retry = Confirm.ask("Try again?", default=True)
            if retry:
                return self._configure_donkit()
            return False
        self.config["RAGOPS_DONKIT_API_KEY"] = api_key
        self.config["RAGOPS_DONKIT_BASE_URL"] = "https://api.donkit.ai"
        ui.print_success("Donkit configured")
        ui.newline()
        return True

    def _configure_vertex(self) -> bool:
        """Configure Vertex AI credentials."""
        ui = get_ui()
        ui.print("You need a service account key file from Google Cloud.", StyleName.DIM)
        ui.print(
            "Get it at: https://console.cloud.google.com/iam-admin/serviceaccounts", StyleName.DIM
        )
        ui.newline()

        path = Prompt.ask("Enter path to service account JSON file")
        path = os.path.expanduser(path)

        if not Path(path).exists():
            ui.print_error(f"File not found: {path}")
            retry = Confirm.ask("Try again?", default=True)
            if retry:
                return self._configure_vertex()
            return False

        self.config["RAGOPS_VERTEX_CREDENTIALS"] = path
        ui.print_styled(
            styled_text(
                (None, "Credentials file: "),
                (StyleName.SUCCESS, path),
            )
        )
        ui.newline()
        return True

    def _configure_openai(self, use_case: str = "chat") -> bool:
        """Configure OpenAI credentials."""
        ui = get_ui()
        ui.print("Get your API key at: https://platform.openai.com/api-keys", StyleName.DIM)
        ui.print("Or use any OpenAI-compatible API provider", StyleName.DIM)
        ui.newline()

        api_key = Prompt.ask("Enter OpenAI API key", password=True)

        if not api_key:
            ui.print_error("API key is required")
            retry = Confirm.ask("Try again?", default=True)
            if retry:
                return self._configure_openai()
            return False

        # Validate OpenAI key format (but allow custom providers)
        if not api_key.startswith("sk-"):
            ui.print_warning("OpenAI API keys usually start with 'sk-'")
            ui.print_warning("  (Ignore this if using a custom provider)")
            retry = Confirm.ask("Continue anyway?", default=True)
            if not retry:
                return self._configure_openai()

        self.config["RAGOPS_OPENAI_API_KEY"] = api_key

        # Optional model name
        if use_case == "chat":
            ui.newline()
            use_custom_model = interactive_confirm("Specify model name?", default=False)

            if use_custom_model:
                model = Prompt.ask("Enter model name", default="gpt-5")
                self.config["RAGOPS_LLM_MODEL"] = model
                ui.print_styled(
                    styled_text(
                        (None, "Model: "),
                        (StyleName.SUCCESS, model),
                    )
                )

        # Optional embedding model
        ui.newline()
        use_embedding_model = interactive_confirm("Specify embedding model?", default=False)

        if use_embedding_model:
            embedding_model = Prompt.ask(
                "Enter embedding model name", default="text-embedding-3-small"
            )
            self.config["RAGOPS_OPENAI_EMBEDDINGS_MODEL"] = embedding_model
            ui.print_styled(
                styled_text(
                    (None, "Embedding model: "),
                    (StyleName.SUCCESS, embedding_model),
                )
            )

        # Optional custom base URL
        ui.newline()
        use_custom_url = interactive_confirm(
            "Use custom base URL? (for OpenAI-compatible providers)", default=False
        )

        if use_custom_url:
            base_url = Prompt.ask("Enter base URL", default="https://api.openai.com/v1")

            if not base_url.startswith("http"):
                ui.print_warning("Base URL should start with 'http://' or 'https://'")
                retry = interactive_confirm("Continue anyway?", default=False)
                if not retry:
                    return self._configure_openai()

            self.config["RAGOPS_OPENAI_BASE_URL"] = base_url
            ui.print_styled(
                styled_text(
                    (None, "Custom base URL: "),
                    (StyleName.SUCCESS, base_url),
                )
            )

        ui.print_success("OpenAI configured")
        ui.newline()
        return True

    def _configure_anthropic(self) -> bool:
        """Configure Anthropic credentials."""
        ui = get_ui()
        ui.print("Get your API key at: https://console.anthropic.com/", StyleName.DIM)
        ui.newline()

        api_key = Prompt.ask("Enter Anthropic API key", password=True)

        if not api_key or not api_key.startswith("sk-ant-"):
            ui.print_warning("API key should start with 'sk-ant-'")
            retry = Confirm.ask("Continue anyway?", default=False)
            if not retry:
                return self._configure_anthropic()
        ui.print(
            "Anthropic claude does not support embeddings, "
            "please configure a separate provider for embeddings"
        )
        embeddings_provider = self._choose_provider(use_case="embeddings")
        configured = self.configure_provider(embeddings_provider)
        if not configured:
            retry = Confirm.ask("Try again?", default=True)
            if retry:
                return self.configure_provider(embeddings_provider)
            else:
                return False
        self.config["RAGOPS_ANTHROPIC_API_KEY"] = api_key
        ui.print_success("API key configured")
        ui.newline()
        return True

    def _configure_azure_openai(self, use_case: str = "chat") -> bool:
        """Configure Azure OpenAI credentials."""
        ui = get_ui()
        ui.print("You need credentials from Azure OpenAI service.", StyleName.DIM)
        ui.print("Get them at: https://portal.azure.com", StyleName.DIM)
        ui.newline()

        api_key = Prompt.ask("Enter Azure OpenAI API key", password=True)

        if not api_key:
            ui.print_error("API key is required")
            retry = Confirm.ask("Try again?", default=True)
            if retry:
                return self._configure_azure_openai()
            return False

        endpoint = Prompt.ask(
            "Enter Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com)"
        )

        if not endpoint.startswith("https://"):
            ui.print_warning("Endpoint should start with 'https://'")
            retry = Confirm.ask("Continue anyway?", default=False)
            if not retry:
                return self._configure_azure_openai()

        api_version = Prompt.ask("Enter API version", default="2024-02-15-preview")
        if use_case == "chat":
            deployment = Prompt.ask("Enter chat completion deployment name (e.g., gpt-5)")
        else:
            deployment = None
        embeddings_deployment = Prompt.ask(
            "Enter embeddings deployment name (e.g., text-embedding-ada-002)",
            default="text-embedding-ada-002",
        )
        self.config["RAGOPS_AZURE_OPENAI_API_KEY"] = api_key
        self.config["RAGOPS_AZURE_OPENAI_ENDPOINT"] = endpoint
        self.config["RAGOPS_AZURE_OPENAI_API_VERSION"] = api_version
        if deployment:
            self.config["RAGOPS_AZURE_OPENAI_DEPLOYMENT"] = deployment
        self.config["RAGOPS_AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"] = embeddings_deployment

        ui.print_success("Azure OpenAI configured")
        ui.newline()
        return True

    def _configure_ollama(self, use_case: str = "chat") -> bool:
        """Configure Ollama local instance."""
        ui = get_ui()
        ui.print("Make sure Ollama is installed and running.", StyleName.DIM)
        ui.print("Install at: https://ollama.ai", StyleName.DIM)
        ui.newline()

        # Ollama uses its own host configuration
        default_url = "http://localhost:11434/v1"
        base_url = Prompt.ask("Ollama base URL", default=default_url)

        self.config["RAGOPS_OLLAMA_BASE_URL"] = base_url
        ui.print_styled(
            styled_text(
                (None, "Ollama URL: "),
                (StyleName.SUCCESS, base_url),
            )
        )

        # Chat model name
        ui.newline()
        if use_case == "chat":
            chat_model = Prompt.ask("Enter chat model name", default="gpt-oss:20b")
            self.config["RAGOPS_LLM_MODEL"] = chat_model
            self.config["RAGOPS_OLLAMA_CHAT_MODEL"] = chat_model
            ui.print_styled(
                styled_text(
                    (None, "Chat model: "),
                    (StyleName.SUCCESS, chat_model),
                )
            )
            ui.newline()
            vision_model = Prompt.ask(
                "Ensure that the chat model is supports VISION, "
                "otherwise specify a vision model (for image analysis)",
                default=chat_model,
            )
            self.config["RAGOPS_OLLAMA_VISION_MODEL"] = vision_model
            ui.print_styled(
                styled_text(
                    (None, "Vision model: "),
                    (StyleName.SUCCESS, vision_model),
                )
            )

        # Embedding model name
        ui.newline()
        embedding_model = Prompt.ask("Enter embedding model name", default="embeddinggemma")
        self.config["RAGOPS_OLLAMA_EMBEDDINGS_MODEL"] = embedding_model
        ui.print_styled(
            styled_text(
                (None, "Embedding model: "),
                (StyleName.SUCCESS, embedding_model),
            )
        )

        ui.print_success("Ollama configured")
        ui.newline()
        return True

    def _configure_openrouter(self) -> bool:
        """Configure OpenRouter credentials."""
        ui = get_ui()
        ui.print("Get your API key at: https://openrouter.ai/keys", StyleName.DIM)
        ui.newline()

        api_key = Prompt.ask("Enter OpenRouter API key", password=True)

        if not api_key:
            ui.print_error("API key is required")
            retry = Confirm.ask("Try again?", default=True)
            if retry:
                return self._configure_openrouter()
            return False

        # OpenRouter uses OpenAI-compatible API
        self.config["RAGOPS_OPENAI_API_KEY"] = api_key
        self.config["RAGOPS_OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        ui.print_styled(
            styled_text(
                (None, "OpenRouter URL: "),
                (StyleName.SUCCESS, "https://openrouter.ai/api/v1"),
            )
        )

        # Chat model name
        ui.newline()
        chat_model = Prompt.ask("Enter chat model name", default="openai/gpt-4o-mini")
        self.config["RAGOPS_LLM_MODEL"] = chat_model
        ui.print_styled(
            styled_text(
                (None, "Chat model: "),
                (StyleName.SUCCESS, chat_model),
            )
        )
        ui.print(
            "Openrouter does not support embeddings, "
            "please configure a separate provider for embeddings"
        )
        embeddings_provider = self._choose_provider(use_case="embeddings")
        configured = self.configure_provider(embeddings_provider, use_case="embeddings")
        if not configured:
            retry = Confirm.ask("Try again?", default=True)
            if retry:
                return self.configure_provider(embeddings_provider)
            else:
                return False
        # Embedding model name
        ui.print_success("OpenRouter configured")
        ui.newline()
        return True

    def _configure_optional_settings(self) -> None:
        """Configure optional settings."""
        ui = get_ui()
        ui.print_styled(
            styled_text(
                (StyleName.BOLD, "Step 3: "),
                (None, "Optional settings"),
            )
        )
        ui.newline()

        # Always use ERROR log level by default
        self.config["RAGOPS_LOG_LEVEL"] = "ERROR"
        ui.print("Using default log level: ERROR", StyleName.DIM)
        ui.newline()

    def save_config(self) -> bool:
        """Save configuration to .env file."""
        ui = get_ui()
        ui.print_styled(
            styled_text(
                (StyleName.BOLD, "Step 4: "),
                (None, "Save configuration"),
            )
        )
        ui.newline()

        # Show summary
        summary_lines = [styled_text((StyleName.BOLD, "Configuration summary:"))]
        summary_lines.append(styled_text((None, "")))
        for key, value in self.config.items():
            display_value = value
            # Mask sensitive values
            if "KEY" in key and len(value) > 10:
                display_value = value[:8] + "..." + value[-4:]
            summary_lines.append(
                styled_text(
                    (StyleName.DIM, f"  {key} = "),
                    (StyleName.SUCCESS, display_value),
                )
            )

        ui.print_panel(summary_lines, border_style=StyleName.INFO)
        ui.newline()

        # Check if we have write permissions
        target_dir = self.env_path.parent
        if not os.access(target_dir, os.W_OK):
            ui.print_error(f"No write permission in: {target_dir}")
            ui.newline()
            # Suggest alternative location
            home_dir = Path.home() / "ragops-workspace"
            ui.print_styled(
                styled_text(
                    (StyleName.WARNING, "Suggestion: "),
                    (None, "Create workspace directory first:"),
                )
            )
            ui.print(f"  mkdir -p {home_dir}")
            ui.print(f"  cd {home_dir}")
            ui.print("  donkit-ragops-ce --setup")
            ui.newline()
            return False

        # Read existing .env if it exists and merge with new config
        existing_config = {}
        if self.env_path.exists():
            try:
                existing_config = dict(dotenv_values(self.env_path))
            except Exception:
                # If we can't read existing file, ask to overwrite
                ui.print_warning(f"Could not read existing file: {self.env_path}")
                overwrite = interactive_confirm("Overwrite?", default=False)
                if not overwrite:
                    ui.print_error("Setup cancelled.")
                    return False

        # Save to .env
        try:
            self.env_path.parent.mkdir(parents=True, exist_ok=True)

            # For new files we keep structured output
            if not existing_config:
                lines = [
                    "# RAGOps Agent CE Configuration",
                    "# Generated by setup wizard",
                    "",
                ]

                for key, value in self.config.items():
                    lines.append(f"{key}={value}")

                if lines and lines[-1] != "":
                    lines.append("")

                self.env_path.write_text("\n".join(lines))
                ui.print_styled(
                    styled_text(
                        (None, "Configuration saved to: "),
                        (StyleName.SUCCESS, str(self.env_path)),
                    )
                )
                ui.newline()
                return True

            # Existing file: update only changed values
            existing_text = self.env_path.read_text(encoding="utf-8")
            existing_lines = existing_text.splitlines()

            updates = {k: v for k, v in self.config.items() if v is not None}
            remaining_updates = dict(updates)
            updated_lines: list[str] = []

            for line in existing_lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in line:
                    updated_lines.append(line)
                    continue

                key = stripped.split("=", 1)[0].strip()
                if key in remaining_updates:
                    updated_lines.append(f"{key}={remaining_updates.pop(key)}")
                else:
                    updated_lines.append(line)

            if remaining_updates:
                if updated_lines and updated_lines[-1].strip():
                    updated_lines.append("")
                for key, value in remaining_updates.items():
                    updated_lines.append(f"{key}={value}")

            updated_content = "\n".join(updated_lines)
            if not updated_content.endswith("\n"):
                updated_content += "\n"

            self.env_path.write_text(updated_content, encoding="utf-8")
            ui.print_styled(
                styled_text(
                    (None, "Configuration updated in: "),
                    (StyleName.SUCCESS, str(self.env_path)),
                )
            )
            ui.newline()
            return True
        except PermissionError:
            ui.print_error(f"Permission denied: Cannot write to {self.env_path}")
            ui.newline()
            ui.print_warning("Try running from a directory where you have write permissions.")
            return False
        except Exception as e:
            ui.print_error(f"Failed to save configuration: {e}")
            return False

    def show_success(self) -> None:
        """Show success message after setup."""
        ui = get_ui()
        success_lines = [
            styled_text((StyleName.SUCCESS, "Setup completed successfully!")),
            styled_text((None, "")),
            styled_text((None, "You can now start the agent with:")),
            styled_text((StyleName.INFO, "  donkit-ragops")),
            styled_text((None, "")),
            styled_text(
                (StyleName.DIM, "Or edit "),
                (StyleName.WARNING, str(self.env_path)),
                (StyleName.DIM, " manually to change settings."),
            ),
        ]

        ui.print_panel(success_lines, title="Ready", border_style=StyleName.SUCCESS)


def check_needs_setup(env_path: Path | None = None) -> bool:
    """Check if setup is needed (no .env file or missing required settings)."""
    ui = get_ui()
    env_path = env_path or Path.cwd() / ".env"

    if not env_path.exists():
        ui.print(f"No .env file found at {env_path}")
        return True

    # Check if .env has required settings
    try:
        config = dotenv_values(env_path)
        provider = config.get("RAGOPS_LLM_PROVIDER")
        ui.print(f"Found .env with provider {provider}")
        if not provider:
            return True

        # Use shared credential checking logic
        return not check_provider_credentials(provider, env_path)
    except Exception as ex:
        logger.exception("Failed to check provider credentials - {}", ex)
        return True


def run_setup_if_needed(force: bool = False) -> bool:
    """Run setup wizard if needed. Returns True if agent can proceed."""
    ui = get_ui()
    env_path = Path.cwd() / ".env"

    if force or check_needs_setup(env_path):
        if not force:
            ui.print_warning("No configuration found. Running setup wizard...")
            ui.newline()

        wizard = SetupWizard(env_path)
        success = wizard.run()

        if success:
            wizard.show_success()
            ui.newline()
            return True
        else:
            ui.print_error("Setup failed or cancelled. Cannot start agent.")
            return False

    return True
