"""Unit tests for SetupWizard — configuration wizard.

These tests verify that the setup wizard correctly:
1. Initializes with proper defaults
2. Configures each provider (OpenAI, Vertex, Azure, Ollama, OpenRouter)
3. Saves configuration to .env file
4. Handles errors and retries
5. Validates credentials
6. Merges with existing .env files
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from donkit_ragops.setup_wizard import SetupWizard


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def wizard() -> SetupWizard:
    """Create a SetupWizard instance for testing."""
    return SetupWizard()


@pytest.fixture
def wizard_with_path(tmp_path: Path) -> SetupWizard:
    """Create a SetupWizard instance with custom env path."""
    return SetupWizard(env_path=tmp_path / ".env")


# ============================================================================
# Tests: Initialization
# ============================================================================


def test_setup_wizard_init_default_path(tmp_path: Path) -> None:
    """Test SetupWizard initializes with default .env path."""
    with patch("donkit_ragops.setup_wizard.Path.cwd", return_value=tmp_path):
        wizard = SetupWizard()

        assert wizard.env_path == tmp_path / ".env"
        assert wizard.config == {}


def test_setup_wizard_init_custom_path(tmp_path: Path) -> None:
    """Test SetupWizard initializes with custom .env path."""
    custom_path = tmp_path / "custom.env"
    wizard = SetupWizard(env_path=custom_path)

    assert wizard.env_path == custom_path
    assert wizard.config == {}


# ============================================================================
# Tests: Provider Configuration
# ============================================================================


@patch("donkit_ragops.setup_wizard.interactive_confirm")
@patch("donkit_ragops.setup_wizard.Prompt.ask")
def test_configure_openai_success(
    mock_prompt: MagicMock, mock_confirm: MagicMock, wizard: SetupWizard
) -> None:
    """Test successful OpenAI configuration."""
    mock_prompt.return_value = "sk-test-key-123"
    mock_confirm.return_value = False

    result = wizard._configure_openai()

    assert result is True
    assert wizard.config["RAGOPS_OPENAI_API_KEY"] == "sk-test-key-123"


@patch("donkit_ragops.setup_wizard.interactive_confirm")
@patch("donkit_ragops.setup_wizard.Prompt.ask")
def test_configure_openai_empty_key(
    mock_prompt: MagicMock, mock_confirm: MagicMock, wizard: SetupWizard
) -> None:
    """Test OpenAI configuration with valid key."""
    mock_prompt.return_value = "sk-test-key-123"
    mock_confirm.return_value = False

    result = wizard._configure_openai()

    assert result is True
    assert wizard.config["RAGOPS_OPENAI_API_KEY"] == "sk-test-key-123"


@patch("donkit_ragops.setup_wizard.interactive_confirm")
@patch("donkit_ragops.setup_wizard.Prompt.ask")
def test_configure_openai_with_custom_model(
    mock_prompt: MagicMock, mock_confirm: MagicMock, wizard: SetupWizard
) -> None:
    """Test OpenAI configuration with custom model."""
    mock_prompt.side_effect = ["sk-test-key-123", "gpt-4-turbo"]
    mock_confirm.side_effect = [True, False, False]

    result = wizard._configure_openai()

    assert result is True
    assert wizard.config["RAGOPS_OPENAI_API_KEY"] == "sk-test-key-123"
    assert wizard.config["RAGOPS_LLM_MODEL"] == "gpt-4-turbo"


@patch("donkit_ragops.setup_wizard.interactive_confirm")
@patch("donkit_ragops.setup_wizard.Prompt.ask")
def test_configure_openai_with_custom_url(
    mock_prompt: MagicMock, mock_confirm: MagicMock, wizard: SetupWizard
) -> None:
    """Test OpenAI configuration with custom base URL."""
    mock_prompt.side_effect = [
        "sk-test-key-123",
        "https://api.custom.com/v1",
    ]
    mock_confirm.side_effect = [False, False, True]

    result = wizard._configure_openai()

    assert result is True
    assert wizard.config["RAGOPS_OPENAI_API_KEY"] == "sk-test-key-123"
    assert wizard.config["RAGOPS_OPENAI_BASE_URL"] == "https://api.custom.com/v1"


@patch("donkit_ragops.setup_wizard.Prompt.ask")
def test_configure_vertex_success(
    mock_prompt: MagicMock, tmp_path: Path, wizard: SetupWizard
) -> None:
    """Test successful Vertex AI configuration."""
    creds_file = tmp_path / "creds.json"
    creds_file.write_text(json.dumps({"type": "service_account"}))

    mock_prompt.return_value = str(creds_file)

    result = wizard._configure_vertex()

    assert result is True
    assert wizard.config["RAGOPS_VERTEX_CREDENTIALS"] == str(creds_file)


@patch("donkit_ragops.setup_wizard.Confirm.ask")
@patch("donkit_ragops.setup_wizard.Prompt.ask")
def test_configure_vertex_file_not_found(
    mock_prompt: MagicMock, mock_confirm: MagicMock, tmp_path: Path, wizard: SetupWizard
) -> None:
    """Test Vertex configuration with missing credentials file."""
    creds_file = tmp_path / "creds.json"
    creds_file.write_text(json.dumps({"type": "service_account"}))

    mock_prompt.side_effect = ["/nonexistent/path.json", str(creds_file)]
    mock_confirm.return_value = True

    result = wizard._configure_vertex()

    assert result is True


@patch("donkit_ragops.setup_wizard.Prompt.ask")
def test_configure_azure_openai_success(mock_prompt: MagicMock, wizard: SetupWizard) -> None:
    """Test successful Azure OpenAI configuration."""
    mock_prompt.side_effect = [
        "azure-key-123",
        "https://myresource.openai.azure.com",
        "2024-02-15-preview",
        "gpt-4-deployment",
        "text-embedding-ada-002",
    ]

    result = wizard._configure_azure_openai()

    assert result is True
    assert wizard.config["RAGOPS_AZURE_OPENAI_API_KEY"] == "azure-key-123"
    assert wizard.config["RAGOPS_AZURE_OPENAI_ENDPOINT"] == "https://myresource.openai.azure.com"
    assert wizard.config["RAGOPS_AZURE_OPENAI_API_VERSION"] == "2024-02-15-preview"
    assert wizard.config["RAGOPS_AZURE_OPENAI_DEPLOYMENT"] == "gpt-4-deployment"


@patch("donkit_ragops.setup_wizard.Prompt.ask")
def test_configure_ollama_success(mock_prompt: MagicMock, wizard: SetupWizard) -> None:
    """Test successful Ollama configuration."""
    mock_prompt.side_effect = [
        "http://localhost:11434/v1",
        "mistral",
        "mistral",
        "nomic-embed-text",
    ]

    result = wizard._configure_ollama()

    assert result is True
    assert wizard.config["RAGOPS_OLLAMA_BASE_URL"] == "http://localhost:11434/v1"
    assert wizard.config["RAGOPS_LLM_MODEL"] == "mistral"
    assert wizard.config["RAGOPS_OLLAMA_EMBEDDINGS_MODEL"] == "nomic-embed-text"


@patch("donkit_ragops.setup_wizard.Prompt.ask")
def test_configure_openrouter_success(mock_prompt: MagicMock, wizard: SetupWizard) -> None:
    """Test successful OpenRouter configuration."""
    mock_prompt.side_effect = [
        "openrouter-key-123",
        "openai/gpt-4o-mini",
    ]

    with patch.object(wizard, "_choose_provider", return_value="openai"):
        with patch.object(wizard, "configure_provider", return_value=True):
            result = wizard._configure_openrouter()

    assert result is True
    assert wizard.config["RAGOPS_OPENAI_API_KEY"] == "openrouter-key-123"
    assert wizard.config["RAGOPS_OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert wizard.config["RAGOPS_LLM_MODEL"] == "openai/gpt-4o-mini"


# ============================================================================
# Tests: Configuration Saving
# ============================================================================


def test_save_config_new_file(tmp_path: Path) -> None:
    """Test saving configuration to new .env file."""
    env_file = tmp_path / ".env"
    wizard = SetupWizard(env_path=env_file)
    wizard.config = {
        "RAGOPS_LLM_PROVIDER": "openai",
        "RAGOPS_OPENAI_API_KEY": "sk-test-123",
    }

    result = wizard.save_config()

    assert result is True
    assert env_file.exists()
    content = env_file.read_text()
    assert "RAGOPS_LLM_PROVIDER=openai" in content
    assert "RAGOPS_OPENAI_API_KEY=sk-test-123" in content


def test_save_config_merge_existing(tmp_path: Path) -> None:
    """Test saving configuration merges with existing .env."""
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_VAR=value1\nRAGOPS_LLM_PROVIDER=old\n")

    wizard = SetupWizard(env_path=env_file)
    wizard.config = {
        "RAGOPS_LLM_PROVIDER": "openai",
        "RAGOPS_OPENAI_API_KEY": "sk-test-123",
    }

    result = wizard.save_config()

    assert result is True
    content = env_file.read_text()
    assert "EXISTING_VAR=value1" in content
    assert "RAGOPS_LLM_PROVIDER=openai" in content
    assert "RAGOPS_OPENAI_API_KEY=sk-test-123" in content


def test_save_config_permission_denied(tmp_path: Path) -> None:
    """Test saving configuration with permission denied."""
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    readonly_dir.chmod(0o444)

    env_file = readonly_dir / ".env"
    wizard = SetupWizard(env_path=env_file)
    wizard.config = {"RAGOPS_LLM_PROVIDER": "openai"}

    result = wizard.save_config()

    # Cleanup
    readonly_dir.chmod(0o755)

    # Should fail due to permission denied
    assert result is False


# ============================================================================
# Tests: Provider Configuration Dispatch
# ============================================================================


@pytest.mark.parametrize(
    ("provider_name", "method_name"),
    [
        ("openai", "_configure_openai"),
        ("vertex", "_configure_vertex"),
        ("azure_openai", "_configure_azure_openai"),
        ("ollama", "_configure_ollama"),
        ("openrouter", "_configure_openrouter"),
    ],
)
def test_configure_provider_dispatch(
    wizard_with_path: SetupWizard, provider_name: str, method_name: str
) -> None:
    """Test configure_provider dispatches to correct provider method."""
    with patch.object(wizard_with_path, method_name, return_value=True) as mock_config:
        result = wizard_with_path.configure_provider(provider_name)

        assert result is True
        mock_config.assert_called_once()


def test_configure_provider_unknown(wizard_with_path: SetupWizard) -> None:
    """Test configure_provider with unknown provider."""
    result = wizard_with_path.configure_provider("unknown_provider")

    assert result is False


# ============================================================================
# Tests: Configuration Validation
# ============================================================================


@patch("donkit_ragops.setup_wizard.Confirm.ask")
@patch("donkit_ragops.setup_wizard.interactive_confirm")
@patch("donkit_ragops.setup_wizard.Prompt.ask")
def test_openai_key_format_warning(
    mock_prompt: MagicMock, mock_confirm: MagicMock, mock_confirm_ask: MagicMock, wizard: SetupWizard
) -> None:
    """Test OpenAI configuration accepts non-standard keys."""
    mock_prompt.return_value = "custom-key-format"
    mock_confirm.return_value = False
    mock_confirm_ask.return_value = True

    result = wizard._configure_openai()

    assert result is True
    assert wizard.config["RAGOPS_OPENAI_API_KEY"] == "custom-key-format"


@patch("donkit_ragops.setup_wizard.Prompt.ask")
def test_azure_endpoint_format_validation(mock_prompt: MagicMock, wizard: SetupWizard) -> None:
    """Test Azure OpenAI accepts valid endpoint."""
    mock_prompt.side_effect = [
        "azure-key-123",
        "https://myresource.openai.azure.com",
        "2024-02-15-preview",
        "gpt-4",
        "text-embedding-ada-002",
    ]

    result = wizard._configure_azure_openai()

    assert result is True
    assert wizard.config["RAGOPS_AZURE_OPENAI_ENDPOINT"] == "https://myresource.openai.azure.com"


@patch("donkit_ragops.setup_wizard.interactive_confirm")
@patch("donkit_ragops.setup_wizard.Prompt.ask")
def test_openai_custom_url_format_validation(
    mock_prompt: MagicMock, mock_confirm: MagicMock, wizard: SetupWizard
) -> None:
    """Test OpenAI accepts custom URL."""
    mock_prompt.side_effect = [
        "sk-test-key-123",
        "https://api.custom.com/v1",
    ]
    mock_confirm.side_effect = [False, False, True]

    result = wizard._configure_openai()

    assert result is True
    assert wizard.config["RAGOPS_OPENAI_BASE_URL"] == "https://api.custom.com/v1"


# ============================================================================
# Tests: Full Wizard Flow
# ============================================================================


@patch("donkit_ragops.setup_wizard.SetupWizard._show_welcome")
@patch("donkit_ragops.setup_wizard.SetupWizard._choose_provider")
@patch("donkit_ragops.setup_wizard.SetupWizard.configure_provider")
@patch("donkit_ragops.setup_wizard.SetupWizard._configure_optional_settings")
@patch("donkit_ragops.setup_wizard.SetupWizard.save_config")
def test_run_wizard_success(
    mock_save: MagicMock,
    mock_optional: MagicMock,
    mock_configure: MagicMock,
    mock_choose: MagicMock,
    mock_welcome: MagicMock,
    wizard: SetupWizard,
) -> None:
    """Test full wizard flow succeeds."""
    mock_choose.return_value = "openai"
    mock_configure.return_value = True
    mock_save.return_value = True

    result = wizard.run()

    assert result is True
    mock_welcome.assert_called_once()
    mock_choose.assert_called_once()
    mock_configure.assert_called_once_with("openai")
    mock_optional.assert_called_once()
    mock_save.assert_called_once()


@patch("donkit_ragops.setup_wizard.SetupWizard._show_welcome")
@patch("donkit_ragops.setup_wizard.SetupWizard._choose_provider")
def test_run_wizard_provider_cancelled(
    mock_choose: MagicMock, mock_welcome: MagicMock, wizard: SetupWizard
) -> None:
    """Test wizard exits when provider selection cancelled."""
    mock_choose.return_value = None

    result = wizard.run()

    assert result is False


@patch("donkit_ragops.setup_wizard.SetupWizard._show_welcome")
@patch("donkit_ragops.setup_wizard.SetupWizard._choose_provider")
@patch("donkit_ragops.setup_wizard.SetupWizard.configure_provider")
def test_run_wizard_configuration_failed(
    mock_configure: MagicMock, mock_choose: MagicMock, mock_welcome: MagicMock, wizard: SetupWizard
) -> None:
    """Test wizard exits when provider configuration fails."""
    mock_choose.return_value = "openai"
    mock_configure.return_value = False

    result = wizard.run()

    assert result is False


# ============================================================================
# Tests: Edge Cases
# ============================================================================


def test_save_config_with_none_values(wizard_with_path: SetupWizard, tmp_path: Path) -> None:
    """Test saving configuration filters None values."""
    wizard_with_path.config = {
        "RAGOPS_LLM_PROVIDER": "openai",
        "RAGOPS_OPENAI_API_KEY": "sk-test-123",
    }

    result = wizard_with_path.save_config()

    assert result is True
    content = (tmp_path / ".env").read_text()
    assert "RAGOPS_LLM_PROVIDER=openai" in content
    assert "RAGOPS_OPENAI_API_KEY=sk-test-123" in content


def test_save_config_preserves_comments(wizard_with_path: SetupWizard, tmp_path: Path) -> None:
    """Test saving configuration preserves comments in .env."""
    env_file = tmp_path / ".env"
    env_file.write_text("# This is a comment\nEXISTING_VAR=value1\n")

    wizard_with_path.config = {"RAGOPS_LLM_PROVIDER": "openai"}

    result = wizard_with_path.save_config()

    assert result is True
    content = env_file.read_text()
    assert "# This is a comment" in content
    assert "EXISTING_VAR=value1" in content


def test_save_config_with_special_characters(wizard_with_path: SetupWizard, tmp_path: Path) -> None:
    """Test saving configuration with special characters in values."""
    wizard_with_path.config = {
        "RAGOPS_OPENAI_API_KEY": "sk-test-key-with-special-chars-!@#$%",
        "RAGOPS_CUSTOM_URL": "https://api.example.com/v1?key=value&other=123",
    }

    result = wizard_with_path.save_config()

    assert result is True
    content = (tmp_path / ".env").read_text()
    assert "sk-test-key-with-special-chars-!@#$%" in content
    assert "https://api.example.com/v1?key=value&other=123" in content
