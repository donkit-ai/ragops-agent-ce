"""Local REPL implementation.

Handles local mode REPL with LLM agent and MCP servers.
"""

from __future__ import annotations

import asyncio

from donkit.llm import Message, ModelCapability
from loguru import logger

from donkit_ragops import texts
from donkit_ragops.display import ScreenRenderer
from donkit_ragops.history_manager import compress_history_if_needed
from donkit_ragops.prints import RAGOPS_LOGO_ART, RAGOPS_LOGO_TEXT
from donkit_ragops.repl.base import BaseREPL, ReplContext
from donkit_ragops.repl.commands import CommandRegistry, create_default_registry
from donkit_ragops.repl_helpers import (
    MCPEventHandler,
    build_stream_render_helper,
    format_timestamp,
    render_markdown_to_rich,
)
from donkit_ragops.ui.styles import StyleName


class LocalREPL(BaseREPL):
    """REPL for local mode with LLM agent.

    Handles streaming responses, MCP tool calls, and command processing.
    """

    def __init__(self, context: ReplContext) -> None:
        """Initialize LocalREPL.

        Args:
            context: REPL context with dependencies
        """
        super().__init__(context)
        self._command_registry: CommandRegistry = create_default_registry()
        self._setup_command_handlers()

    def _setup_command_handlers(self) -> None:
        """Set up additional command handlers for local mode."""
        # Provider and model commands are handled separately
        # as they need access to local-specific state
        pass

    async def initialize(self) -> None:
        """Initialize REPL resources."""
        # Print logo
        self.context.ui.print(RAGOPS_LOGO_TEXT)
        self.context.ui.print(RAGOPS_LOGO_ART)

        # Set up render helpers
        if self.context.renderer is None:
            self.context.renderer = ScreenRenderer()

        if self.context.render_helper is None and self.context.agent_settings is not None:
            self.context.render_helper = build_stream_render_helper(
                transcript=self.context.transcript,
                renderer=self.context.renderer,
                agent_settings=self.context.agent_settings,
                session_started_at=self.context.session_started_at,
                show_checklist=self.context.show_checklist,
            )

        if self.context.mcp_handler is None and self.context.agent_settings is not None:
            self.context.mcp_handler = MCPEventHandler(
                render_helper=self.context.render_helper,
                agent_settings=self.context.agent_settings,
                session_started_at=self.context.session_started_at,
                show_checklist=self.context.show_checklist,
            )

        # Initialize agent MCP tools
        if self.context.agent is not None:
            await self.context.agent.ainit_mcp_tools()

        # Add system prompt to history
        if self.context.system_prompt:
            self.context.history.append(Message(role="system", content=self.context.system_prompt))

    async def run(self) -> None:
        """Run the REPL main loop."""
        await self.initialize()
        self.print_welcome()
        self._print_welcome_message()

        while self._running:
            try:
                user_input = self.context.ui.text_input()
            except KeyboardInterrupt:
                self.context.ui.print_warning(
                    "Input cancelled. Press Ctrl+C again or type :quit to exit"
                )
                continue

            if not user_input:
                continue

            try:
                should_continue = await self.handle_input(user_input)
                if not should_continue:
                    break
            except (KeyboardInterrupt, asyncio.CancelledError):
                self.context.ui.print_warning("Operation interrupted. Press Ctrl+C again to exit.")
                if self.context.mcp_handler:
                    self.context.mcp_handler.clear_progress()
            except Exception as e:
                if self.context.render_helper:
                    self.context.render_helper.append_error(str(e))
                logger.error(f"Error in main loop: {e}", exc_info=True)

        await self.cleanup()

    async def handle_input(self, user_input: str) -> bool:
        """Handle user input.

        Args:
            user_input: Raw user input

        Returns:
            False to exit, True to continue
        """
        # Check for commands
        if self._command_registry.is_command(user_input) or user_input in {"quit", "exit"}:
            return await self._handle_command(user_input)

        # Handle special commands not in registry
        if user_input == ":provider":
            await self._handle_provider_command()
            return True

        if user_input == ":model":
            await self._handle_model_command()
            return True

        # Regular message
        await self.handle_message(user_input)
        return True

    async def _handle_command(self, user_input: str) -> bool:
        """Handle a registered command."""
        cmd = self._command_registry.get_command(user_input)
        if cmd is None:
            self.context.ui.print(f"Unknown command: {user_input}", StyleName.WARNING)
            return True

        result = await cmd.execute(self.context)

        # Print styled messages using UI abstraction
        for styled_msg in result.styled_messages:
            self.context.ui.print_styled(styled_msg)

        # Legacy: also handle plain messages for backward compatibility
        for msg in result.messages:
            self.context.transcript.append(msg)

        if result.should_exit:
            if self.context.render_helper:
                self.context.render_helper.render_current_screen()
            if self.context.renderer:
                self.context.renderer.render_goodbye_screen()
            return False

        return True

    async def _handle_provider_command(self) -> None:
        """Handle :provider command."""
        # Import here to avoid circular imports
        from donkit_ragops.cli_helpers import (
            format_model_choices,
            get_available_models,
            select_provider_interactively,
            validate_model_choice,
        )
        from donkit_ragops.config import load_settings
        from donkit_ragops.interactive_input import interactive_confirm, interactive_select
        from donkit_ragops.model_selector import PROVIDERS

        if self.context.render_helper:
            self.context.render_helper.render_current_screen()

        settings = load_settings()
        result = await select_provider_interactively(
            current_provider=self.context.provider_name,
            current_model=self.context.model,
            settings=settings,
            agent_settings=self.context.agent_settings,
            prov=self.context.provider,
            agent=self.context.agent,
            tools=self.context.agent.tools if self.context.agent else [],
            mcp_clients=self.context.mcp_clients,
        )

        if result is None:
            if self.context.render_helper:
                self.context.render_helper.render_current_screen()
            return

        self.context.transcript.extend(result.messages)

        if result.cancelled:
            if self.context.render_helper:
                self.context.render_helper.render_current_screen()
            return

        # Update context
        self.context.provider_name = result.provider
        self.context.provider = result.prov
        self.context.agent = result.agent
        self.context.agent_settings = result.agent_settings
        self.context.model = result.model

        if self.context.mcp_handler:
            self.context.mcp_handler.agent_settings = result.agent_settings
            self.context.mcp_handler.clear_progress()

        if self.context.render_helper:
            self.context.render_helper.render_current_screen()

        # Prompt for model selection
        if result.prompt_model_selection and result.provider:
            current_model_display = (
                (self.context.agent_settings.model if self.context.agent_settings else None)
                or self.context.model
                or "default"
            )
            want_to_change = interactive_confirm(
                question=f"Current model: {current_model_display}. Select a different model?",
                default=False,
            )
            if want_to_change:
                models = get_available_models(result.prov, result.provider)
                if models:
                    choices = format_model_choices(models, self.context.model)
                    choices.append(texts.MODEL_SELECT_SKIP)
                    title = texts.MODEL_SELECT_TITLE.format(
                        provider=PROVIDERS[result.provider]["display"]
                    )
                    selected = interactive_select(choices, title=title)
                    if selected and selected != texts.MODEL_SELECT_SKIP:
                        new_model = selected.split(" [")[0].strip()
                        success, messages = validate_model_choice(
                            result.prov, result.provider, new_model, self.context.agent_settings
                        )
                        self.context.transcript.extend(messages)
                        if success:
                            self.context.model = new_model
                    elif selected == texts.MODEL_SELECT_SKIP:
                        self.context.transcript.append(texts.MODEL_USING_DEFAULT)
                else:
                    self.context.transcript.append(texts.MODEL_NO_AVAILABLE)
            else:
                self.context.transcript.append(
                    f"[dim]Keeping current model: {current_model_display}[/dim]"
                )

        if self.context.render_helper:
            self.context.render_helper.render_current_screen()

    async def _handle_model_command(self) -> None:
        """Handle :model command."""
        from donkit_ragops.cli_helpers import (
            format_model_choices,
            get_available_models,
            validate_model_choice,
        )
        from donkit_ragops.config import load_settings
        from donkit_ragops.interactive_input import interactive_select

        if self.context.render_helper:
            self.context.render_helper.render_current_screen()

        settings = load_settings()
        current_provider = self.context.provider_name or settings.llm_provider or "openai"
        current_model = (
            self.context.agent_settings.model if self.context.agent_settings else None
        ) or self.context.model

        models = get_available_models(self.context.provider, current_provider)
        if not models:
            self.context.transcript.append(
                texts.MODEL_NO_PREDEFINED.format(current_model_name=current_model or "not set")
            )
            if self.context.render_helper:
                self.context.render_helper.render_current_screen()
            return

        choices = format_model_choices(models, current_model)
        choices.append("Custom (enter manually)")
        title = f"Select Model for {current_provider}"
        selected = interactive_select(choices, title=title)

        if selected is None:
            if self.context.render_helper:
                self.context.render_helper.render_current_screen()
            return

        if selected == "Custom (enter manually)":
            self.context.transcript.append(
                "[bold yellow]Please specify model name in .env file or use CLI flag.[/bold yellow]"
            )
            if self.context.render_helper:
                self.context.render_helper.render_current_screen()
            return

        new_model = selected.split(" [")[0].strip()
        success, messages = validate_model_choice(
            self.context.provider, current_provider, new_model, self.context.agent_settings
        )
        self.context.transcript.extend(messages)
        if success:
            self.context.model = new_model

        if self.context.render_helper:
            self.context.render_helper.render_current_screen()

    async def handle_message(self, message: str) -> None:
        """Handle a chat message."""
        if self.context.render_helper:
            self.context.render_helper.append_user_line(message)

        self.context.history.append(Message(role="user", content=message))

        if self.context.provider and self.context.provider.supports_capability(
            ModelCapability.STREAMING
        ):
            await self._handle_streaming_response()
        else:
            await self._handle_non_streaming_response()

        # Compress history if needed
        if self.context.provider:
            self.context.history[:] = await compress_history_if_needed(
                self.context.history, self.context.provider
            )

    async def _handle_streaming_response(self) -> None:
        """Handle streaming response from agent."""
        reply = ""
        interrupted = False
        first_content = True
        response_index = None

        if self.context.render_helper:
            response_index = self.context.render_helper.start_agent_placeholder()

        spinner = self.context.ui.create_spinner(texts.THINKING_MESSAGE_PLAIN)
        spinner.start()

        try:
            display_content = ""
            temp_executing = ""

            async for event in self.context.agent.arespond_stream(self.context.history):
                # Stop spinner on first event (content OR tool call)
                # This prevents Rich "only one live display" error when tools
                # use interactive UI (confirm/select dialogs with Live)
                if first_content:
                    spinner.stop()
                    first_content = False
                    if event.type.name == "CONTENT" and event.content:
                        if self.context.render_helper:
                            self.context.render_helper.print_agent_prefix()

                # Print content directly
                if event.type.name == "CONTENT" and event.content:
                    self.context.ui.print(event.content, end="")

                # Print tool messages
                if event.type.name == "TOOL_CALL_START" and self.context.mcp_handler:
                    self.context.ui.print(
                        self.context.mcp_handler.tool_executing_message(
                            event.tool_name, event.tool_args
                        )
                    )
                elif event.type.name == "TOOL_CALL_END" and self.context.mcp_handler:
                    self.context.ui.print(
                        self.context.mcp_handler.tool_done_message(event.tool_name)
                    )
                elif event.type.name == "TOOL_CALL_ERROR" and self.context.mcp_handler:
                    self.context.ui.print(
                        self.context.mcp_handler.tool_error_message(
                            event.tool_name, event.error or ""
                        )
                    )

                if self.context.mcp_handler:
                    reply, display_content, temp_executing = (
                        self.context.mcp_handler.process_stream_event(
                            event,
                            self.context.history,
                            reply,
                            display_content,
                            temp_executing,
                        )
                    )

                if self.context.render_helper and response_index is not None:
                    self.context.render_helper.set_agent_line(
                        response_index, display_content, temp_executing
                    )

        except (KeyboardInterrupt, asyncio.CancelledError):
            if first_content:
                spinner.stop()
            interrupted = True
            if reply:
                self.context.history.append(Message(role="assistant", content=reply))
                rendered = render_markdown_to_rich(display_content)
                if self.context.render_helper and response_index is not None:
                    self.context.render_helper.set_agent_line(
                        response_index, f"{rendered}\nGeneration interrupted by user", ""
                    )
            elif response_index is not None:
                self.context.transcript[response_index] = (
                    f"{format_timestamp()}[yellow]Generation interrupted by user[/yellow]"
                )
            self.context.ui.print_warning("\nGeneration interrupted")

        except Exception as e:
            if first_content:
                spinner.stop()
            error_msg = f"{format_timestamp()}[bold red]Error:[/bold red] {e}"
            if response_index is not None:
                self.context.transcript[response_index] = error_msg
            else:
                self.context.transcript.append(error_msg)
            logger.error(f"Error during streaming: {e}", exc_info=True)
            self.context.ui.print_error(str(e))
            if self.context.mcp_handler:
                self.context.mcp_handler.clear_progress()

        # Ensure spinner stopped
        if first_content:
            spinner.stop()

        # Add to history if we got a response
        if reply and not interrupted:
            self.context.history.append(Message(role="assistant", content=reply))
            rendered = render_markdown_to_rich(display_content)
            if self.context.render_helper and response_index is not None:
                self.context.render_helper.set_agent_line(response_index, rendered, "")
            self.context.ui.newline()
        elif not interrupted and not reply:
            error_msg = f"{format_timestamp()}[bold red]Error:[/bold red] No response from agent"
            if response_index is not None:
                self.context.transcript[response_index] = error_msg
            else:
                self.context.transcript.append(error_msg)
            self.context.ui.print_error("No response from agent")

        if self.context.mcp_handler:
            self.context.mcp_handler.clear_progress()

    async def _handle_non_streaming_response(self) -> None:
        """Handle non-streaming response from agent."""
        response_index = None
        if self.context.render_helper:
            response_index = self.context.render_helper.start_agent_placeholder()

        spinner = self.context.ui.create_spinner(texts.THINKING_MESSAGE_PLAIN)
        spinner.start()

        try:
            reply = await self.context.agent.arespond(self.context.history)
            spinner.stop()

            if reply:
                self.context.history.append(Message(role="assistant", content=reply))
                rendered = render_markdown_to_rich(reply)
                if self.context.render_helper and response_index is not None:
                    self.context.render_helper.set_agent_line(response_index, rendered, "")
                    self.context.render_helper.print_agent_prefix()
                self.context.ui.print_markdown(reply)
            else:
                if response_index is not None:
                    self.context.transcript[response_index] = (
                        f"{format_timestamp()}[bold red]Error:[/bold red] "
                        "No response from agent. Check logs for details."
                    )
                self.context.ui.print_error("No response from agent.")

        except (KeyboardInterrupt, asyncio.CancelledError):
            spinner.stop()
            self.context.ui.print_warning("Generation interrupted by user")
            if response_index is not None:
                self.context.transcript[response_index] = (
                    f"{format_timestamp()}[yellow]Generation interrupted[/yellow]"
                )

        except Exception as e:
            spinner.stop()
            error_msg = f"{format_timestamp()}[bold red]Error:[/bold red] {e}"
            if response_index is not None:
                self.context.transcript[response_index] = error_msg
            logger.error(f"Error during agent response: {e}", exc_info=True)
            self.context.ui.print_error(str(e))

        if self.context.mcp_handler:
            self.context.mcp_handler.clear_progress()

    async def cleanup(self) -> None:
        """Clean up REPL resources."""
        self._running = False

    def _print_welcome_message(self) -> None:
        """Print welcome message."""
        rendered = render_markdown_to_rich(texts.WELCOME_MESSAGE)
        if self.context.render_helper:
            self.context.render_helper.append_agent_message(rendered)
        self.context.ui.print(f"{texts.AGENT_PREFIX} {rendered}")
        self.context.ui.newline()
