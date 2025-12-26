"""Tests for UI abstraction layer."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from donkit_ragops.ui import UI, StyleName, get_ui, reset_ui
from donkit_ragops.ui.adapters.plain_adapter import PlainUI
from donkit_ragops.ui.adapters.rich_adapter import RichUI
from donkit_ragops.ui.styles import RICH_STYLES, get_rich_style, styled_text


class TestStyleName:
    """Tests for StyleName enum."""

    def test_style_name_values(self):
        """Test that StyleName enum has expected values."""
        assert StyleName.ERROR.value == "error"
        assert StyleName.SUCCESS.value == "success"
        assert StyleName.WARNING.value == "warning"
        assert StyleName.INFO.value == "info"
        assert StyleName.DIM.value == "dim"
        assert StyleName.BOLD.value == "bold"

    def test_all_styles_have_rich_mapping(self):
        """Test that all StyleName values have Rich mappings."""
        for style in StyleName:
            assert style in RICH_STYLES, f"StyleName.{style.name} has no Rich mapping"

    def test_get_rich_style(self):
        """Test get_rich_style function."""
        assert get_rich_style(StyleName.ERROR) == "bold red"
        assert get_rich_style(StyleName.SUCCESS) == "bold green"
        assert get_rich_style(StyleName.DIM) == "dim"


class TestStyledText:
    """Tests for StyledText type and helpers."""

    def test_styled_text_creation(self):
        """Test creating StyledText."""
        st = styled_text(
            (StyleName.DIM, "Loading: "),
            (StyleName.BOLD, "file.txt"),
            (None, " done"),
        )
        assert len(st) == 3
        assert st[0] == (StyleName.DIM, "Loading: ")
        assert st[1] == (StyleName.BOLD, "file.txt")
        assert st[2] == (None, " done")


class TestGetUI:
    """Tests for get_ui factory function."""

    def setup_method(self):
        """Reset UI before each test."""
        reset_ui()

    def teardown_method(self):
        """Reset UI after each test."""
        reset_ui()

    def test_get_ui_returns_ui_instance(self):
        """Test that get_ui returns a UI instance."""
        ui = get_ui(force_plain=True)
        assert ui is not None

    def test_get_ui_force_plain(self):
        """Test that force_plain returns PlainUI."""
        ui = get_ui(force_plain=True)
        assert isinstance(ui, PlainUI)

    def test_get_ui_caches_instance(self):
        """Test that get_ui returns the same instance."""
        reset_ui()
        ui1 = get_ui(force_plain=True)
        ui2 = get_ui()  # Should return cached instance
        assert ui1 is ui2

    def test_get_ui_in_tty_returns_rich(self):
        """Test that get_ui returns RichUI in TTY mode."""
        reset_ui()
        with patch.object(sys.stdout, "isatty", return_value=True):
            ui = get_ui()
            assert isinstance(ui, RichUI)

    def test_get_ui_in_pipe_returns_plain(self):
        """Test that get_ui returns PlainUI when piped."""
        reset_ui()
        with patch.object(sys.stdout, "isatty", return_value=False):
            ui = get_ui()
            assert isinstance(ui, PlainUI)


class TestPlainUI:
    """Tests for PlainUI adapter."""

    def setup_method(self):
        """Create PlainUI instance."""
        self.ui = PlainUI()

    def test_print(self, capsys):
        """Test print method."""
        self.ui.print("Hello, World!")
        captured = capsys.readouterr()
        assert "Hello, World!" in captured.out

    def test_print_with_style(self, capsys):
        """Test print with style (style is ignored in PlainUI)."""
        self.ui.print("Error message", StyleName.ERROR)
        captured = capsys.readouterr()
        assert "Error message" in captured.out

    def test_print_error(self, capsys):
        """Test print_error method."""
        self.ui.print_error("Something went wrong")
        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "Something went wrong" in captured.out

    def test_print_success(self, capsys):
        """Test print_success method."""
        self.ui.print_success("Done!")
        captured = capsys.readouterr()
        assert "Done!" in captured.out

    def test_print_warning(self, capsys):
        """Test print_warning method."""
        self.ui.print_warning("Be careful")
        captured = capsys.readouterr()
        assert "Warning:" in captured.out
        assert "Be careful" in captured.out

    def test_print_info(self, capsys):
        """Test print_info method."""
        self.ui.print_info("FYI")
        captured = capsys.readouterr()
        assert "Info:" in captured.out
        assert "FYI" in captured.out

    def test_print_markdown(self, capsys):
        """Test print_markdown method."""
        self.ui.print_markdown("# Title\nParagraph")
        captured = capsys.readouterr()
        assert "Title" in captured.out

    def test_print_panel(self, capsys):
        """Test print_panel method."""
        self.ui.print_panel("Content", title="Title")
        captured = capsys.readouterr()
        assert "Title" in captured.out
        assert "Content" in captured.out

    def test_print_styled(self, capsys):
        """Test print_styled method."""
        st = styled_text(
            (StyleName.DIM, "Dim: "),
            (StyleName.BOLD, "Bold"),
        )
        self.ui.print_styled(st)
        captured = capsys.readouterr()
        assert "Dim: Bold" in captured.out

    def test_newline(self, capsys):
        """Test newline method."""
        self.ui.newline()
        captured = capsys.readouterr()
        assert captured.out == "\n"

    def test_create_spinner_context_manager(self, capsys):
        """Test spinner as context manager."""
        with self.ui.create_spinner("Loading...") as spinner:
            spinner.update("Still loading...")
        captured = capsys.readouterr()
        assert "Loading..." in captured.out
        assert "Still loading..." in captured.out

    def test_create_progress_context_manager(self, capsys):
        """Test progress bar as context manager."""
        with self.ui.create_progress(total=10, description="Working") as progress:
            progress.update(5)
        captured = capsys.readouterr()
        assert "[0/10]" in captured.out
        assert "[5/10]" in captured.out

    def test_select_with_input(self):
        """Test select method with mocked input."""
        with patch("builtins.input", return_value="1"):
            result = self.ui.select(["Option A", "Option B"], "Choose:")
            assert result == "Option A"

    def test_select_cancel(self):
        """Test select method cancellation."""
        with patch("builtins.input", return_value="q"):
            result = self.ui.select(["Option A", "Option B"], "Choose:")
            assert result is None

    def test_select_default(self):
        """Test select method with default (empty input)."""
        with patch("builtins.input", return_value=""):
            result = self.ui.select(["Option A", "Option B"], "Choose:", default_index=1)
            assert result == "Option B"

    def test_confirm_yes(self):
        """Test confirm method with yes."""
        with patch("builtins.input", return_value="y"):
            result = self.ui.confirm("Are you sure?")
            assert result is True

    def test_confirm_no(self):
        """Test confirm method with no."""
        with patch("builtins.input", return_value="n"):
            result = self.ui.confirm("Are you sure?")
            assert result is False

    def test_confirm_default(self):
        """Test confirm method with default (empty input)."""
        with patch("builtins.input", return_value=""):
            result = self.ui.confirm("Are you sure?", default=False)
            assert result is False

    def test_text_input(self):
        """Test text_input method."""
        with patch("builtins.input", return_value="  hello  "):
            result = self.ui.text_input()
            assert result == "hello"

    def test_text_input_interrupt(self):
        """Test text_input raises KeyboardInterrupt."""
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                self.ui.text_input()

    def test_create_live_context(self, capsys):
        """Test live context."""
        with self.ui.create_live_context() as live:
            live.update("First")
            live.update("First Second")
        captured = capsys.readouterr()
        assert "First" in captured.out


class TestRichUI:
    """Tests for RichUI adapter."""

    def setup_method(self):
        """Create RichUI instance."""
        self.ui = RichUI()

    def test_has_console(self):
        """Test that RichUI has a console."""
        assert self.ui.console is not None

    def test_print_styled(self):
        """Test print_styled method."""
        # Just ensure it doesn't raise
        st = styled_text(
            (StyleName.DIM, "Dim: "),
            (StyleName.BOLD, "Bold"),
        )
        self.ui.print_styled(st)

    def test_create_spinner(self):
        """Test spinner creation."""
        spinner = self.ui.create_spinner("Loading...")
        assert spinner is not None
        # Test context manager
        with spinner:
            spinner.update("Still loading...")

    def test_create_progress(self):
        """Test progress bar creation."""
        progress = self.ui.create_progress(total=100, description="Working")
        assert progress is not None
        # Test context manager
        with progress:
            progress.update(50)
            progress.advance(10)

    def test_create_live_context(self):
        """Test live context creation."""
        live = self.ui.create_live_context()
        assert live is not None
        # Test context manager
        with live:
            live.update("Content")
            live.refresh()


class TestUIProtocol:
    """Tests for UI protocol compliance."""

    def test_plain_ui_implements_protocol(self):
        """Test that PlainUI implements UI protocol."""
        ui = PlainUI()
        assert isinstance(ui, UI)

    def test_rich_ui_implements_protocol(self):
        """Test that RichUI implements UI protocol."""
        ui = RichUI()
        assert isinstance(ui, UI)
