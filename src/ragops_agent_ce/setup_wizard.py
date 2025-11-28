"""
Interactive setup wizard for first-time configuration.
Works without LLM - pure hardcoded logic for collecting user settings.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text

from ragops_agent_ce.credential_checker import check_provider_credentials
from ragops_agent_ce.interactive_input import interactive_confirm, interactive_select

console = Console()


class SetupWizard:
    """Interactive setup wizard for configuring RAGOps Agent CE."""

    def __init__(self, env_path: Path | None = None):
        self.env_path = env_path or Path.cwd() / ".env"
        self.config: dict[str, str] = {}

    def run(self) -> bool:
        """Run the setup wizard. Returns True if setup completed successfully."""
        console.clear()
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
        welcome_text = Text()
        welcome_text.append("Welcome to ", style="white")
        welcome_text.append("RAGOps Agent CE", style="bold cyan")
        welcome_text.append(" Setup Wizard!\n\n", style="white")

        # Strong recommendation about workspace
        welcome_text.append("💡 ", style="bold cyan")
        welcome_text.append("IMPORTANT: ", style="bold cyan")
        welcome_text.append("Run the agent from a new, empty directory!\n", style="cyan")
        welcome_text.append(
            "The agent will create project files, .env, and other artifacts.\n", style="dim"
        )
        welcome_text.append("Recommended:\n", style="dim")
        welcome_text.append(
            "  mkdir ~/ragops-workspace && cd ~/ragops-workspace\n\n", style="green"
        )

        welcome_text.append(
            "This wizard will help you configure the agent for first use.\n", style="dim"
        )
        welcome_text.append("You'll need an API key for your chosen LLM provider.\n\n", style="dim")
        welcome_text.append("✅ ", style="green")
        welcome_text.append("Supported: ", style="green bold")
        welcome_text.append(
            "Vertex AI (Recommended), OpenAI, Azure OpenAI, Ollama, OpenRouter\n",
            style="green",
        )
        welcome_text.append("More providers are coming soon!", style="dim italic")

        console.print(Panel(welcome_text, title="🚀 Setup", border_style="cyan"))
        console.print()

    def _choose_provider(self, use_case: str = "chat") -> str | None:
        """Let user choose LLM provider."""
        if use_case == "embeddings":
            console.print("[bold]Step 1:[/bold] Choose your embeddings provider\n")
        else:
            console.print("[bold]Step 1:[/bold] Choose your LLM provider\n")
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
            console.print("[red]Setup cancelled[/red]")
            return None

        # Find the selected provider by matching the choice
        selected_idx = choices.index(selected)
        provider_key = available_providers[selected_idx][0]
        provider = providers[provider_key]["name"]

        console.print(f"\n✓ Selected: [green]{providers[provider_key]['display']}[/green]\n")
        return provider

    def configure_provider(self, provider: str, use_case: str = "chat") -> bool:
        """Configure credentials for chosen provider."""
        console.print(f"[bold]Step 2:[/bold] Configure {provider} credentials\n")
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
                console.print(f"[red]✗ Unknown provider: {provider}[/red]")
                return False

    def _configure_donkit(self):
        """Configure Donkit credentials."""

        """Configure Donkit credentials."""
        console.print("[dim]Get your API key at: https://donkit.ai/api[/dim]\n")
        api_key = Prompt.ask("Enter Donkit API key", password=True)
        if not api_key:
            console.print("[red]✗ API key is required[/red]")
            retry = Confirm.ask("Try again?", default=True)
            if retry:
                return self._configure_donkit()
            return False
        self.config["RAGOPS_DONKIT_API_KEY"] = api_key
        self.config["RAGOPS_DONKIT_BASE_URL"] = "https://api.donkit.ai"
        console.print("✓ Donkit configured\n")
        return True

    def _configure_vertex(self) -> bool:
        """Configure Vertex AI credentials."""
        console.print("[dim]You need a service account key file from Google Cloud.[/dim]")
        console.print(
            "[dim]Get it at: https://console.cloud.google.com/iam-admin/serviceaccounts[/dim]\n"
        )

        path = Prompt.ask("Enter path to service account JSON file")
        path = os.path.expanduser(path)

        if not Path(path).exists():
            console.print(f"[red]✗ File not found:[/red] {path}")
            retry = Confirm.ask("Try again?", default=True)
            if retry:
                return self._configure_vertex()
            return False

        self.config["RAGOPS_VERTEX_CREDENTIALS"] = path
        console.print(f"✓ Credentials file: [green]{path}[/green]\n")
        return True

    def _configure_openai(self, use_case: str = "chat") -> bool:
        """Configure OpenAI credentials."""
        console.print("[dim]Get your API key at: https://platform.openai.com/api-keys[/dim]")
        console.print("[dim]Or use any OpenAI-compatible API provider[/dim]\n")

        api_key = Prompt.ask("Enter OpenAI API key", password=True)

        if not api_key:
            console.print("[red]✗ API key is required[/red]")
            retry = Confirm.ask("Try again?", default=True)
            if retry:
                return self._configure_openai()
            return False

        # Validate OpenAI key format (but allow custom providers)
        if not api_key.startswith("sk-"):
            console.print("[yellow]⚠ OpenAI API keys usually start with 'sk-'[/yellow]")
            console.print("[yellow]  (Ignore this if using a custom provider)[/yellow]")
            retry = Confirm.ask("Continue anyway?", default=True)
            if not retry:
                return self._configure_openai()

        self.config["RAGOPS_OPENAI_API_KEY"] = api_key

        # Optional model name
        if use_case == "chat":
            console.print()
            use_custom_model = interactive_confirm("Specify model name?", default=False)

            if use_custom_model:
                model = Prompt.ask("Enter model name", default="gpt-5")
                self.config["RAGOPS_LLM_MODEL"] = model
                console.print(f"✓ Model: [green]{model}[/green]")

        # Optional embedding model
        console.print()
        use_embedding_model = interactive_confirm("Specify embedding model?", default=False)

        if use_embedding_model:
            embedding_model = Prompt.ask(
                "Enter embedding model name", default="text-embedding-3-small"
            )
            self.config["RAGOPS_OPENAI_EMBEDDINGS_MODEL"] = embedding_model
            console.print(f"✓ Embedding model: [green]{embedding_model}[/green]")

        # Optional custom base URL
        console.print()
        use_custom_url = interactive_confirm(
            "Use custom base URL? (for OpenAI-compatible providers)", default=False
        )

        if use_custom_url:
            base_url = Prompt.ask("Enter base URL", default="https://api.openai.com/v1")

            if not base_url.startswith("http"):
                console.print(
                    "[yellow]⚠ Base URL should start with 'http://' or 'https://'[/yellow]"
                )
                retry = interactive_confirm("Continue anyway?", default=False)
                if not retry:
                    return self._configure_openai()

            self.config["RAGOPS_OPENAI_BASE_URL"] = base_url
            console.print(f"✓ Custom base URL: [green]{base_url}[/green]")

        console.print("✓ OpenAI configured\n")
        return True

    def _configure_anthropic(self) -> bool:
        """Configure Anthropic credentials."""
        console.print("[dim]Get your API key at: https://console.anthropic.com/[/dim]\n")

        api_key = Prompt.ask("Enter Anthropic API key", password=True)

        if not api_key or not api_key.startswith("sk-ant-"):
            console.print("[yellow]⚠ API key should start with 'sk-ant-'[/yellow]")
            retry = Confirm.ask("Continue anyway?", default=False)
            if not retry:
                return self._configure_anthropic()
        console.print(
            "Anthropic claude does not support embeddings,"
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
        console.print("✓ API key configured\n")
        return True

    def _configure_azure_openai(self, use_case: str = "chat") -> bool:
        """Configure Azure OpenAI credentials."""
        console.print("[dim]You need credentials from Azure OpenAI service.[/dim]")
        console.print("[dim]Get them at: https://portal.azure.com[/dim]\n")

        api_key = Prompt.ask("Enter Azure OpenAI API key", password=True)

        if not api_key:
            console.print("[red]✗ API key is required[/red]")
            retry = Confirm.ask("Try again?", default=True)
            if retry:
                return self._configure_azure_openai()
            return False

        endpoint = Prompt.ask(
            "Enter Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com)"
        )

        if not endpoint.startswith("https://"):
            console.print("[yellow]⚠ Endpoint should start with 'https://'[/yellow]")
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

        console.print("✓ Azure OpenAI configured\n")
        return True

    def _configure_ollama(self, use_case: str = "chat") -> bool:
        """Configure Ollama local instance."""
        console.print("[dim]Make sure Ollama is installed and running.[/dim]")
        console.print("[dim]Install at: https://ollama.ai[/dim]\n")

        # Ollama uses its own host configuration
        default_url = "http://localhost:11434/v1"
        base_url = Prompt.ask("Ollama base URL", default=default_url)

        self.config["RAGOPS_OLLAMA_BASE_URL"] = base_url
        console.print(f"✓ Ollama URL: [green]{base_url}[/green]")

        # Chat model name
        console.print()
        if use_case == "chat":
            chat_model = Prompt.ask("Enter chat model name", default="gpt-oss:20b")
            self.config["RAGOPS_LLM_MODEL"] = chat_model
            self.config["RAGOPS_OLLAMA_CHAT_MODEL"] = chat_model
            console.print(f"✓ Chat model: [green]{chat_model}[/green]")
            console.print()
            vision_model = Prompt.ask(
                "Ensure that the chat model is supports VISION, "
                "otherwise specify a vision model (for image analysis)",
                default=chat_model,
            )
            self.config["RAGOPS_OLLAMA_VISION_MODEL"] = vision_model
            console.print(f"✓ Vision model: [green]{vision_model}[/green]")

        # Embedding model name
        console.print()
        embedding_model = Prompt.ask("Enter embedding model name", default="embeddinggemma")
        self.config["RAGOPS_OLLAMA_EMBEDDINGS_MODEL"] = embedding_model
        console.print(f"✓ Embedding model: [green]{embedding_model}[/green]")

        console.print("✓ Ollama configured\n")
        return True

    def _configure_openrouter(self) -> bool:
        """Configure OpenRouter credentials."""
        console.print("[dim]Get your API key at: https://openrouter.ai/keys[/dim]\n")

        api_key = Prompt.ask("Enter OpenRouter API key", password=True)

        if not api_key:
            console.print("[red]✗ API key is required[/red]")
            retry = Confirm.ask("Try again?", default=True)
            if retry:
                return self._configure_openrouter()
            return False

        # OpenRouter uses OpenAI-compatible API
        self.config["RAGOPS_OPENAI_API_KEY"] = api_key
        self.config["RAGOPS_OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        console.print("✓ OpenRouter URL: [green]https://openrouter.ai/api/v1[/green]")

        # Chat model name
        console.print()
        chat_model = Prompt.ask("Enter chat model name", default="openai/gpt-4o-mini")
        self.config["RAGOPS_LLM_MODEL"] = chat_model
        console.print(f"✓ Chat model: [green]{chat_model}[/green]")
        console.print(
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
        console.print("✓ OpenRouter configured\n")
        return True

    def _configure_optional_settings(self) -> None:
        """Configure optional settings."""
        console.print("[bold]Step 3:[/bold] Optional settings\n")

        # Always use ERROR log level by default
        self.config["RAGOPS_LOG_LEVEL"] = "ERROR"
        console.print("[dim]Using default log level: ERROR[/dim]\n")

    def save_config(self) -> bool:
        """Save configuration to .env file."""
        console.print("[bold]Step 4:[/bold] Save configuration\n")

        # Show summary
        summary = Text()
        summary.append("Configuration summary:\n\n", style="bold")
        for key, value in self.config.items():
            display_value = value
            # Mask sensitive values
            if "KEY" in key and len(value) > 10:
                display_value = value[:8] + "..." + value[-4:]
            summary.append(f"  {key} = ", style="dim")
            summary.append(f"{display_value}\n", style="green")

        console.print(Panel(summary, border_style="cyan"))
        console.print()

        # Check if we have write permissions
        target_dir = self.env_path.parent
        if not os.access(target_dir, os.W_OK):
            console.print(f"[red]✗ No write permission in:[/red] {target_dir}\n")
            # Suggest alternative location
            home_dir = Path.home() / "ragops-workspace"
            console.print(
                f"[yellow]Suggestion:[/yellow] Create workspace directory first:\n"
                f"  mkdir -p {home_dir}\n"
                f"  cd {home_dir}\n"
                f"  donkit-ragops-ce --setup\n"
            )
            return False

        # Read existing .env if it exists and merge with new config
        existing_config = {}
        if self.env_path.exists():
            try:
                existing_config = dict(dotenv_values(self.env_path))
            except Exception:
                # If we can't read existing file, ask to overwrite
                console.print(f"[yellow]⚠ Could not read existing file:[/yellow] {self.env_path}")
                overwrite = interactive_confirm("Overwrite?", default=False)
                if not overwrite:
                    console.print("[red]Setup cancelled.[/red]")
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
                console.print(f"✓ Configuration saved to: [green]{self.env_path}[/green]\n")
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
            console.print(f"✓ Configuration updated in: [green]{self.env_path}[/green]\n")
            return True
        except PermissionError:
            console.print(f"[red]✗ Permission denied:[/red] Cannot write to {self.env_path}\n")
            console.print(
                "[yellow]Try running from a directory where you have write permissions.[/yellow]"
            )
            return False
        except Exception as e:
            console.print(f"[red]✗ Failed to save configuration:[/red] {e}")
            return False

    def show_success(self) -> None:
        """Show success message after setup."""
        success_text = Text()
        success_text.append("🎉 Setup completed successfully!\n\n", style="bold green")
        success_text.append("You can now start the agent with:\n", style="white")
        success_text.append("  donkit-ragops-ce\n\n", style="bold cyan")
        success_text.append("Or edit ", style="dim")
        success_text.append(f"{self.env_path}", style="yellow")
        success_text.append(" manually to change settings.", style="dim")

        console.print(Panel(success_text, title="✓ Ready", border_style="green"))


def check_needs_setup(env_path: Path | None = None) -> bool:
    """Check if setup is needed (no .env file or missing required settings)."""
    env_path = env_path or Path.cwd() / ".env"

    if not env_path.exists():
        console.print(f"No .env file found at {env_path}")
        return True

    # Check if .env has required settings
    try:
        config = dotenv_values(env_path)
        provider = config.get("RAGOPS_LLM_PROVIDER")
        console.print(f"Found .env with provider {provider}")
        if not provider:
            return True

        # Use shared credential checking logic
        return not check_provider_credentials(provider, env_path)
    except Exception as ex:
        logger.exception("Failed to check provider credentials - {}", ex)
        return True


def run_setup_if_needed(force: bool = False) -> bool:
    """Run setup wizard if needed. Returns True if agent can proceed."""
    env_path = Path.cwd() / ".env"

    if force or check_needs_setup(env_path):
        if not force:
            console.print("[yellow]⚠ No configuration found. Running setup wizard...[/yellow]\n")

        wizard = SetupWizard(env_path)
        success = wizard.run()

        if success:
            wizard.show_success()
            console.print()
            return True
        else:
            console.print("[red]Setup failed or cancelled. Cannot start agent.[/red]")
            return False

    return True
