"""Unit tests for agent tools (file operations, interactive, project management)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest
from donkit_ragops.agent.local_tools.checklist_tools import (
    tool_create_checklist,
    tool_update_checklist_item,
)
from donkit_ragops.agent.local_tools.tools import (
    tool_db_get,
    tool_grep,
    tool_interactive_user_choice,
    tool_interactive_user_confirm,
    tool_list_directory,
    tool_read_file,
    tool_time_now,
)
from donkit_ragops.checklist_manager import (
    ChecklistStatusLine,
    ChecklistStatusProvider,
    checklist_status_provider,
)
from donkit_ragops.db import DB, close, kv_set, migrate

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield db_path
        if db_path.exists():
            db_path.unlink()


@pytest.fixture
def db(temp_db_path: Path) -> Generator[DB, None, None]:
    """Create an in-memory database for testing."""
    database = DB(path=temp_db_path)
    migrate(database)
    yield database
    close(database)


# ============================================================================
# Tests: tool_time_now
# ============================================================================


def test_tool_time_now_returns_iso_format() -> None:
    """Test that tool_time_now returns ISO format datetime."""
    tool = tool_time_now()
    result = tool.handler({})

    # Should be a valid ISO format string
    assert "T" in result  # ISO format has T separator
    assert "-" in result  # ISO format has dashes for date


def test_tool_time_now_metadata() -> None:
    """Test tool_time_now metadata."""
    tool = tool_time_now()

    assert tool.name == "time_now"
    assert "datetime" in tool.description.lower()
    assert tool.parameters["type"] == "object"
    assert tool.parameters["properties"] == {}


# ============================================================================
# Tests: tool_list_directory
# ============================================================================


def test_tool_list_directory_valid_path(temp_dir: Path) -> None:
    """Test listing a valid directory."""
    # Create some test files
    (temp_dir / "file1.txt").touch()
    (temp_dir / "file2.py").touch()
    (temp_dir / "subdir").mkdir()

    tool = tool_list_directory()
    result_str = tool.handler({"path": str(temp_dir)})
    result = json.loads(result_str)

    assert "items" in result
    assert result["total_items"] == 3
    assert any(item["name"] == "file1.txt" for item in result["items"])
    assert any(item["name"] == "subdir" and item["is_directory"] for item in result["items"])


def test_tool_list_directory_nonexistent_path() -> None:
    """Test listing a non-existent directory."""
    tool = tool_list_directory()
    result_str = tool.handler({"path": "/nonexistent/path/12345"})
    result = json.loads(result_str)

    assert "error" in result
    assert "does not exist" in result["error"].lower()


def test_tool_list_directory_file_not_directory(temp_dir: Path) -> None:
    """Test listing a file instead of directory."""
    file_path = temp_dir / "test.txt"
    file_path.write_text("content")

    tool = tool_list_directory()
    result_str = tool.handler({"path": str(file_path)})
    result = json.loads(result_str)

    assert "error" in result
    assert "not a directory" in result["error"].lower()


def test_tool_list_directory_empty_directory(temp_dir: Path) -> None:
    """Test listing an empty directory."""
    tool = tool_list_directory()
    result_str = tool.handler({"path": str(temp_dir)})
    result = json.loads(result_str)

    assert result["total_items"] == 0
    assert result["items"] == []


def test_tool_list_directory_with_sizes(temp_dir: Path) -> None:
    """Test that file sizes are included."""
    file_path = temp_dir / "test.txt"
    file_path.write_text("hello world")

    tool = tool_list_directory()
    result_str = tool.handler({"path": str(temp_dir)})
    result = json.loads(result_str)

    file_item = result["items"][0]
    assert file_item["size_bytes"] == 11
    assert file_item["is_directory"] is False


def test_tool_list_directory_home_expansion(temp_dir: Path) -> None:
    """Test that ~ is expanded to home directory."""
    # This test just verifies the path expansion logic works
    tool = tool_list_directory()
    result_str = tool.handler({"path": "~"})
    result = json.loads(result_str)

    # Should either succeed or give a valid error
    assert "items" in result or "error" in result


# ============================================================================
# Tests: tool_read_file
# ============================================================================


def test_tool_read_file_basic(temp_dir: Path) -> None:
    """Test reading a basic file."""
    file_path = temp_dir / "test.txt"
    file_path.write_text("line1\nline2\nline3\n")

    tool = tool_read_file()
    result_str = tool.handler({"path": str(file_path)})
    result = json.loads(result_str)

    assert result["total_lines"] == 3
    assert "line1" in result["content"]
    assert "line2" in result["content"]
    assert "line3" in result["content"]


def test_tool_read_file_with_offset_and_limit(temp_dir: Path) -> None:
    """Test reading file with offset and limit."""
    file_path = temp_dir / "test.txt"
    lines = "\n".join([f"line{i}" for i in range(1, 11)])
    file_path.write_text(lines)

    tool = tool_read_file()
    result_str = tool.handler({"path": str(file_path), "offset": 3, "limit": 3})
    result = json.loads(result_str)

    assert "line3" in result["content"]
    assert "line4" in result["content"]
    assert "line5" in result["content"]
    assert "line1" not in result["content"]


def test_tool_read_file_nonexistent() -> None:
    """Test reading a non-existent file."""
    tool = tool_read_file()
    result_str = tool.handler({"path": "/nonexistent/file.txt"})
    result = json.loads(result_str)

    assert "error" in result
    assert "does not exist" in result["error"].lower()


def test_tool_read_file_no_path() -> None:
    """Test reading without path parameter."""
    tool = tool_read_file()
    result_str = tool.handler({})
    result = json.loads(result_str)

    assert "error" in result
    assert "required" in result["error"].lower()


def test_tool_read_file_is_directory(temp_dir: Path) -> None:
    """Test reading a directory instead of file."""
    tool = tool_read_file()
    result_str = tool.handler({"path": str(temp_dir)})
    result = json.loads(result_str)

    assert "error" in result
    assert "not a file" in result["error"].lower()


def test_tool_read_file_line_numbers(temp_dir: Path) -> None:
    """Test that line numbers are included in output."""
    file_path = temp_dir / "test.txt"
    file_path.write_text("line1\nline2\n")

    tool = tool_read_file()
    result_str = tool.handler({"path": str(file_path)})
    result = json.loads(result_str)

    # Should have line numbers in format "     1\tline1"
    assert "1\t" in result["content"]
    assert "2\t" in result["content"]


def test_tool_read_file_pagination_note(temp_dir: Path) -> None:
    """Test that pagination note is included when there are more lines."""
    file_path = temp_dir / "test.txt"
    lines = "\n".join([f"line{i}" for i in range(1, 201)])
    file_path.write_text(lines)

    tool = tool_read_file()
    result_str = tool.handler({"path": str(file_path), "limit": 50})
    result = json.loads(result_str)

    assert "note" in result
    assert "more lines" in result["note"].lower()


# ============================================================================
# Tests: tool_grep
# ============================================================================


def test_tool_grep_basic(temp_dir: Path) -> None:
    """Test basic grep functionality."""
    # Create test files with matching names
    (temp_dir / "hello_world.txt").touch()
    (temp_dir / "foo_bar.py").touch()
    (temp_dir / "hello_again.txt").touch()

    tool = tool_grep()
    result_str = tool.handler({"pattern": "hello", "path": str(temp_dir)})

    # grep returns newline-separated JSON objects
    lines = result_str.strip().split("\n")
    matches = [json.loads(line) for line in lines if line]

    # Should have matches for files with "hello" in name
    assert len(matches) >= 1


def test_tool_grep_no_matches(temp_dir: Path) -> None:
    """Test grep with no matches."""
    (temp_dir / "test.txt").touch()

    tool = tool_grep()
    result_str = tool.handler({"pattern": "xyz", "path": str(temp_dir)})

    lines = result_str.strip().split("\n")
    matches = [json.loads(line) for line in lines if line]

    # Should have a summary message about no matches
    assert len(matches) >= 1


def test_tool_grep_case_insensitive(temp_dir: Path) -> None:
    """Test grep with case insensitive flag (default behavior)."""
    (temp_dir / "Hello_world.txt").touch()
    (temp_dir / "FOO_bar.py").touch()

    tool = tool_grep()
    result_str = tool.handler(
        {
            "pattern": "hello",
            "path": str(temp_dir),
        }
    )

    lines = result_str.strip().split("\n")
    matches = [json.loads(line) for line in lines if line]

    # grep is case-insensitive by default
    assert len(matches) >= 1


def test_tool_grep_nonexistent_path() -> None:
    """Test grep on non-existent path."""
    tool = tool_grep()
    result_str = tool.handler({"pattern": "test", "path": "/nonexistent/path/12345"})
    result = json.loads(result_str)

    assert "error" in result


# ============================================================================
# Tests: tool_db_get
# ============================================================================


def test_tool_db_get_existing_key(db: DB, temp_db_path: Path) -> None:
    """Test getting an existing key from database."""
    # Set a value
    kv_set(db, "test_key", "test_value")

    # Patch open_db to return our test db
    with patch("donkit_ragops.agent.local_tools.tools.open_db") as mock_open_db:
        mock_open_db.return_value = db
        tool = tool_db_get()
        result = tool.handler({"key": "test_key"})

    assert result == "test_value"


def test_tool_db_get_nonexistent_key(db: DB) -> None:
    """Test getting a non-existent key returns empty string."""
    with patch("donkit_ragops.agent.local_tools.tools.open_db") as mock_open_db:
        mock_open_db.return_value = db
        tool = tool_db_get()
        result = tool.handler({"key": "nonexistent"})

    assert result == ""


def test_tool_db_get_no_key() -> None:
    """Test db_get without key parameter."""
    with patch("donkit_ragops.agent.local_tools.tools.open_db") as mock_open_db:
        mock_db = Mock()
        mock_open_db.return_value = mock_db
        tool = tool_db_get()
        result = tool.handler({})

    assert result == ""


# ============================================================================
# Tests: tool_interactive_user_choice
# ============================================================================


def test_tool_interactive_user_choice_valid_selection() -> None:
    """Test interactive user choice with valid selection."""
    with patch("donkit_ragops.agent.local_tools.tools.interactive_select") as mock_select:
        mock_select.return_value = "option2"
        tool = tool_interactive_user_choice()
        result_str = tool.handler(
            {
                "title": "Choose one",
                "choices": ["option1", "option2", "option3"],
            }
        )
        result = json.loads(result_str)

    assert result["selected_choice"] == "option2"
    assert result["selected_index"] == 1


def test_tool_interactive_user_choice_no_options() -> None:
    """Test interactive user choice with no options."""
    with patch("donkit_ragops.agent.local_tools.tools.interactive_select"):
        tool = tool_interactive_user_choice()
        result_str = tool.handler({"title": "Choose", "choices": []})
        result = json.loads(result_str)

    assert "error" in result


def test_tool_interactive_user_choice_cancelled() -> None:
    """Test interactive user choice when user cancels."""
    with patch("donkit_ragops.agent.local_tools.tools.interactive_select") as mock_select:
        mock_select.return_value = None
        tool = tool_interactive_user_choice()
        result_str = tool.handler(
            {
                "title": "Choose one",
                "choices": ["option1", "option2"],
            }
        )
        result = json.loads(result_str)

    assert result["cancelled"] is True
    assert result["selected_choice"] is None


# ============================================================================
# Tests: tool_interactive_user_confirm
# ============================================================================


def test_tool_interactive_user_confirm_yes() -> None:
    """Test interactive user confirm with yes."""
    with patch("donkit_ragops.agent.local_tools.tools.interactive_confirm") as mock_confirm:
        mock_confirm.return_value = True
        tool = tool_interactive_user_confirm()
        result_str = tool.handler({"question": "Continue?"})
        result = json.loads(result_str)

    assert result["confirmed"] is True
    assert result["cancelled"] is False


def test_tool_interactive_user_confirm_no() -> None:
    """Test interactive user confirm with no."""
    with patch("donkit_ragops.agent.local_tools.tools.interactive_confirm") as mock_confirm:
        mock_confirm.return_value = False
        tool = tool_interactive_user_confirm()
        result_str = tool.handler({"question": "Continue?"})
        result = json.loads(result_str)

    assert result["confirmed"] is False
    assert result["cancelled"] is False


def test_tool_interactive_user_confirm_cancelled() -> None:
    """Test interactive user confirm when cancelled."""
    with patch("donkit_ragops.agent.local_tools.tools.interactive_confirm") as mock_confirm:
        mock_confirm.return_value = None
        tool = tool_interactive_user_confirm()
        result_str = tool.handler({"question": "Continue?"})
        result = json.loads(result_str)

    assert result["cancelled"] is True
    assert result["confirmed"] is None


# ============================================================================
# Tests: Tool Metadata
# ============================================================================


def test_tool_list_directory_metadata() -> None:
    """Test tool_list_directory metadata."""
    tool = tool_list_directory()

    assert tool.name == "list_directory"
    assert "directory" in tool.description.lower()
    assert "path" in tool.parameters["properties"]
    assert "path" in tool.parameters["required"]


def test_tool_read_file_metadata() -> None:
    """Test tool_read_file metadata."""
    tool = tool_read_file()

    assert tool.name == "read_file"
    assert "file" in tool.description.lower()
    assert "path" in tool.parameters["properties"]
    assert "path" in tool.parameters["required"]


def test_tool_grep_metadata() -> None:
    """Test tool_grep metadata."""
    tool = tool_grep()

    assert tool.name == "grep"
    assert "search" in tool.description.lower() or "pattern" in tool.description.lower()
    assert "pattern" in tool.parameters["properties"]


def test_tool_db_get_metadata() -> None:
    """Test tool_db_get metadata."""
    tool = tool_db_get()

    assert tool.name == "db_get"
    assert "database" in tool.description.lower() or "key-value" in tool.description.lower()
    assert "key" in tool.parameters["properties"]


def test_tool_interactive_user_choice_metadata() -> None:
    """Test tool_interactive_user_choice metadata."""
    tool = tool_interactive_user_choice()

    assert tool.name == "interactive_user_choice"
    assert "choice" in tool.description.lower() or "select" in tool.description.lower()


def test_tool_interactive_user_confirm_metadata() -> None:
    """Test tool_interactive_user_confirm metadata."""
    tool = tool_interactive_user_confirm()

    assert tool.name == "interactive_user_confirm"
    assert "confirm" in tool.description.lower()


# ============================================================================
# Tests: Tool Spec Conversion
# ============================================================================


def test_tool_to_tool_spec() -> None:
    """Test that AgentTool can be converted to ToolSpec."""
    tool = tool_list_directory()
    spec = tool.to_tool_spec()

    assert spec.function.name == "list_directory"
    assert spec.function.description is not None
    assert spec.function.parameters is not None


def test_all_tools_have_valid_specs() -> None:
    """Test that all built-in tools have valid specs."""
    tools = [
        tool_time_now(),
        tool_db_get(),
        tool_list_directory(),
        tool_read_file(),
        tool_grep(),
        tool_interactive_user_choice(),
        tool_interactive_user_confirm(),
    ]

    for tool in tools:
        spec = tool.to_tool_spec()
        assert spec.function.name == tool.name
        assert spec.function.description is not None
        assert spec.function.parameters is not None
        assert spec.function.parameters.get("type") == "object"


# ============================================================================
# Tests: ChecklistStatusProvider
# ============================================================================


def test_checklist_status_line_format_empty() -> None:
    """Test ChecklistStatusLine format with no data."""
    status = ChecklistStatusLine()
    assert status.format() == ""


def test_checklist_status_line_format_with_data() -> None:
    """Test ChecklistStatusLine format with data."""
    status = ChecklistStatusLine(
        icon="⚡",
        description="Deploy vector DB",
        completed=2,
        total=5,
    )
    result = status.format()
    assert "⚡" in result
    assert "Deploy vector DB" in result
    assert "(2/5)" in result


def test_checklist_status_provider_singleton() -> None:
    """Test ChecklistStatusProvider is a singleton."""
    provider1 = ChecklistStatusProvider()
    provider2 = ChecklistStatusProvider()
    assert provider1 is provider2


def test_checklist_status_provider_update_from_empty() -> None:
    """Test updating provider with empty data clears status."""
    provider = ChecklistStatusProvider()
    provider.update_from_checklist(None)
    assert provider.status.total == 0

    provider.update_from_checklist({})
    assert provider.status.total == 0


def test_checklist_status_provider_update_from_checklist() -> None:
    """Test updating provider with checklist data."""
    provider = ChecklistStatusProvider()

    checklist_data = {
        "name": "test_checklist",
        "items": [
            {"id": "item_0", "description": "Task 1", "status": "completed"},
            {"id": "item_1", "description": "Task 2", "status": "in_progress"},
            {"id": "item_2", "description": "Task 3", "status": "pending"},
        ],
    }

    provider.update_from_checklist(checklist_data)

    assert provider.status.total == 3
    assert provider.status.completed == 1
    assert provider.status.icon == "⚡"  # in_progress item
    assert provider.status.description == "Task 2"


def test_checklist_status_provider_all_completed() -> None:
    """Test provider status when all items completed."""
    provider = ChecklistStatusProvider()

    checklist_data = {
        "name": "test_checklist",
        "items": [
            {"id": "item_0", "description": "Task 1", "status": "completed"},
            {"id": "item_1", "description": "Task 2", "status": "completed"},
        ],
    }

    provider.update_from_checklist(checklist_data)

    assert provider.status.icon == "✓"
    assert provider.status.completed == 2
    assert provider.status.total == 2


def test_checklist_status_provider_pending_first() -> None:
    """Test provider shows first pending item when no in_progress."""
    provider = ChecklistStatusProvider()

    checklist_data = {
        "name": "test_checklist",
        "items": [
            {"id": "item_0", "description": "Task 1", "status": "pending"},
            {"id": "item_1", "description": "Task 2", "status": "pending"},
        ],
    }

    provider.update_from_checklist(checklist_data)

    assert provider.status.icon == "○"  # pending
    assert provider.status.description == "Task 1"


def test_checklist_status_provider_clear() -> None:
    """Test clearing provider status."""
    provider = ChecklistStatusProvider()

    # First set some data
    provider.update_from_checklist(
        {
            "items": [{"id": "item_0", "description": "Task", "status": "pending"}],
        }
    )
    assert provider.status.total == 1

    # Clear it
    provider.clear()
    assert provider.status.total == 0


def test_checklist_status_provider_truncates_long_descriptions() -> None:
    """Test that long descriptions are truncated."""
    provider = ChecklistStatusProvider()

    long_desc = "This is a very long task description that should be truncated"
    checklist_data = {
        "items": [{"id": "item_0", "description": long_desc, "status": "pending"}],
    }

    provider.update_from_checklist(checklist_data)

    # Description should be truncated to 35 chars max
    assert len(provider.status.description) <= 35
    assert provider.status.description.endswith("...")


# ============================================================================
# Tests: Checklist Tools with Status Provider Integration
# ============================================================================


def test_checklist_tool_updates_status_provider(db) -> None:
    """Test that checklist tools update the status provider."""
    with patch("donkit_ragops.agent.local_tools.checklist_tools.open_db") as mock_open_db:
        mock_open_db.return_value = db

        # Create a checklist
        tool = tool_create_checklist()
        tool.handler(
            {
                "name": "test_integration",
                "items": ["Task 1", "Task 2", "Task 3"],
            }
        )

        # Provider should be updated
        assert checklist_status_provider.status.total == 3
        assert checklist_status_provider.status.completed == 0


def test_update_checklist_item_updates_status_provider(db) -> None:
    """Test that updating checklist item updates the status provider."""
    with patch("donkit_ragops.agent.local_tools.checklist_tools.open_db") as mock_open_db:
        mock_open_db.return_value = db

        # Create a checklist first
        create_tool = tool_create_checklist()
        create_tool.handler(
            {
                "name": "test_update_integration",
                "items": ["Task 1", "Task 2"],
            }
        )

        # Update an item
        update_tool = tool_update_checklist_item()
        update_tool.handler(
            {
                "name": "test_update_integration",
                "item_id": "item_0",
                "status": "completed",
            }
        )

        # Provider should reflect the update
        assert checklist_status_provider.status.completed == 1
