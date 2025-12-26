# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**RAGOps Agent CE (Community Edition)** - An LLM-powered CLI agent for building and managing RAG (Retrieval-Augmented Generation) pipelines. Published to PyPI as `donkit-ragops`.

- **Language**: Python 3.12+
- **Package Manager**: Poetry
- **Key Technologies**: FastMCP (MCP servers), Typer (CLI), Rich (terminal UI), Pydantic (validation)

## Development Commands

```bash
# Install dependencies
poetry install

# Run CLI in local mode (default)
poetry run donkit-ragops

# Run CLI in enterprise mode
poetry run donkit-ragops --enterprise

# Run interactive setup wizard
poetry run donkit-ragops --setup

# Check status
poetry run donkit-ragops status

# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_agent.py

# Run specific test function
poetry run pytest tests/test_agent.py::test_function_name

# Lint code
poetry run ruff check .

# Format code
poetry run ruff format .

# Check formatting without changes
poetry run ruff format --check .
```

## Architecture

### Two Operating Modes

**Local Mode (default)**
- LLM agent orchestrates built-in MCP servers
- Builds RAG pipelines locally using Docker Compose
- All processing happens on user's machine
- Activated by default or with `--local` flag

**Enterprise Mode (`--enterprise`)**
- Connects to cloud-hosted ragops-api-gateway via WebSocket
- Enables document upload, experiment management, and collaboration
- Requires authentication via `donkit-ragops login`
- Auto-activated when auth token exists in keyring

### Core Components

```
src/donkit_ragops/
├── cli.py                 # CLI commands and routing only (SRP)
├── repl/                  # REPL implementations (SOLID architecture)
│   ├── base.py           # BaseREPL abstract class + ReplContext (DIP)
│   ├── commands.py       # CommandRegistry + command classes (OCP)
│   ├── local_repl.py     # LocalREPL for local mode (LSP)
│   └── enterprise_repl.py # EnterpriseREPL for cloud mode (LSP)
├── history_manager.py     # Conversation history compression (SRP)
├── texts.py              # UI text constants for localization
├── prints.py             # Logo and print utilities
├── display.py            # Screen rendering (ScreenRenderer)
├── interactive_input.py  # User input handling
├── repl_helpers.py       # REPL helper utilities
├── ui/                    # UI abstraction layer (framework-agnostic)
│   ├── protocol.py       # UI protocol (abstract interface)
│   ├── styles.py         # StyleName enum + theme mapping
│   ├── components.py     # Spinner, ProgressBar, LiveContext protocols
│   └── adapters/         # Framework implementations
│       ├── rich_adapter.py   # RichUI (default, full-featured)
│       └── plain_adapter.py  # PlainUI (for testing/pipes)
├── agent/                 # LLM agent core
│   ├── agent.py          # LLMAgent class, tool execution loop
│   ├── prompts.py        # System prompts for different modes
│   └── local_tools/      # Built-in agent tools (non-MCP)
│       ├── tools.py              # File ops, interactive prompts
│       ├── project_tools.py      # Project management
│       └── checklist_tools.py    # Workflow tracking
├── mcp/
│   ├── client.py         # MCP client for connecting to servers
│   └── servers/          # Built-in MCP servers
│       ├── planner_server.py           # RAG config planning
│       ├── chunker_server.py           # Document chunking
│       ├── vectorstore_loader_server.py # Vector DB operations
│       ├── compose_manager_server.py    # Docker orchestration
│       ├── rag_query_server.py         # RAG querying
│       ├── rag_evaluation_server.py    # RAG evaluation
│       ├── read_engine_server.py       # Document parsing
│       └── donkit_ragops_mcp.py        # Unified server (all above)
├── llm/                  # LLM provider integrations
│   ├── provider_factory.py  # Factory for creating providers
│   └── providers/           # OpenAI, Anthropic, Vertex, etc.
├── enterprise/           # Enterprise mode components
│   ├── auth.py          # Token-based authentication (keyring)
│   ├── config.py        # Enterprise settings
│   ├── upload.py        # Document upload to cloud
│   ├── analyzer.py      # File analysis before upload
│   └── ws_client.py     # WebSocket client for API gateway
└── mode.py              # Mode detection logic
```

### SOLID Architecture (REPL System)

The REPL system follows SOLID principles:

1. **SRP (Single Responsibility)**:
   - `cli.py` - CLI commands and routing only
   - `history_manager.py` - conversation history compression only
   - `LocalREPL` / `EnterpriseREPL` - mode-specific REPL logic

2. **OCP (Open/Closed)**:
   - `CommandRegistry` - new commands can be added without modifying existing code
   - New REPLs can be added by inheriting from `BaseREPL`

3. **LSP (Liskov Substitution)**:
   - `LocalREPL` and `EnterpriseREPL` are fully interchangeable via `BaseREPL`

4. **DIP (Dependency Inversion)**:
   - `ReplContext` - dependency container for REPL state
   - Injection via constructor (manual DI)

### Agent Workflow

1. **Startup**: CLI starts → Setup wizard runs if `.env` missing
2. **Initialization**: LLM agent initialized with configured provider (OpenAI, Vertex, Anthropic, Ollama)
3. **Tool Access**: Agent has access to:
   - Local tools (file ops, project management, checklist tracking)
   - MCP servers (chunking, vector loading, Docker orchestration)
4. **Execution**: Agent creates checklists, asks for user confirmation before steps, executes tasks
5. **Project Creation**: Projects stored in `projects/<project-id>/` with compose files, chunks, configs

### MCP Servers

All MCP servers are implemented using FastMCP and can run standalone or unified:

- **Standalone**: `poetry run ragops-chunker`, `poetry run ragops-rag-planner`, etc.
- **Unified**: `poetry run donkit-ragops-mcp` (combines all servers with prefixed tools)

Each server exposes tools via the Model Context Protocol for the agent to use.

### Local Tools vs MCP Servers

**Local Tools** (`src/donkit_ragops/agent/local_tools/`):
- Built directly into agent (no separate process)
- Fast, synchronous operations
- Examples: `read_file`, `list_directory`, `grep`, `create_project`, `interactive_user_confirm`

**MCP Servers** (`src/donkit_ragops/mcp/servers/`):
- Run as separate processes (stdio transport)
- Heavier operations (chunking, vector loading, Docker management)
- Agent connects via MCP client

### Mode Detection

Mode is determined by `src/donkit_ragops/mode.py`:

1. **Explicit flags**: `--local` or `--enterprise` override auto-detection
2. **Auto-detection**: If auth token exists in keyring → Enterprise mode
3. **Default**: Local mode if no token found

Enterprise components (`src/donkit_ragops/enterprise/`) are only loaded in enterprise mode.

## Configuration

### Local Mode Configuration

Uses `.env` file in working directory. Interactive setup wizard creates this automatically:

```bash
# Run setup wizard
poetry run donkit-ragops --setup
```

**Key Environment Variables**:
```bash
# LLM Provider (required)
RAGOPS_LLM_PROVIDER=openai  # or vertex, anthropic, ollama, azure_openai

# OpenAI
RAGOPS_OPENAI_API_KEY=sk-...
RAGOPS_LLM_MODEL=gpt-4o-mini

# Vertex AI (Google Cloud)
RAGOPS_VERTEX_CREDENTIALS=/path/to/service-account-key.json
RAGOPS_VERTEX_PROJECT=your-project-id
RAGOPS_VERTEX_LOCATION=us-central1

# Anthropic
RAGOPS_ANTHROPIC_API_KEY=sk-ant-...

# Logging
RAGOPS_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### Enterprise Mode Configuration

Enterprise mode uses:
- `DONKIT_API_URL` - API Gateway endpoint (default: `https://api.donkit.ai`)
- Auth token stored securely in system keyring (set via `donkit-ragops login`)

No manual token management needed - keyring handles storage automatically.

## Project Structure

When the agent creates a RAG project, it generates:

```
projects/<project-id>/
├── compose/
│   ├── docker-compose.yml    # Qdrant + RAG service
│   └── .env                  # Service configuration
├── chunks/                   # Processed document chunks
│   └── *.json               # Chunked documents with metadata
└── rag_config.json          # RAG pipeline configuration
```

## Testing

All tests located in `tests/` directory:

- `test_agent.py` - Agent core functionality
- `test_cli.py` - CLI commands and REPL routing
- `test_mcp_client.py` - MCP client
- `test_all_mcp_servers.py` - MCP server functionality
- `test_integration_agent_mcp.py` - Agent + MCP integration
- `test_integration_agent_tools.py` - Agent + local tools integration
- `test_db.py` - Database operations
- `test_setup_wizard.py` - Setup wizard
- `test_tools.py` - Local tools

**174 tests total** (all passing)

Test configuration in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
addopts = "-v --tb=short -p no:warnings --disable-warnings"
testpaths = ["tests"]
```

## Key Patterns

- **Poetry** for dependency management
- **Pydantic** for settings and schemas (`config.py`, `schemas/`)
- **Loguru** for logging
- **Typer** for CLI framework
- **Rich** for terminal UI (progress bars, panels, markdown rendering)
- **FastMCP** for MCP server implementation
- **Path dependencies** for local development - shared libraries from `../ragops-agent/shared/`:
  - `donkit-llm` - LLM provider abstractions
  - `donkit-chunker` - Document chunking
  - `donkit-vectorstore-loader` - Vector store operations
  - `donkit-read-engine` - Document parsing
  - `donkit-embeddings` - Embedding providers
  - `donkit-ragops-api-gateway-client` - API Gateway client

## Development Guidelines

### Critical Guidelines

See parent `CLAUDE.md` at repository root (`../CLAUDE.md`) for critical organizational guidelines:
- **CRITICAL: No Mock Code** - Never write placeholders or stubs
- **REQUIRED: Write Tests** - All new features must include tests

### Testing Requirements

**MANDATORY: Always Run Tests**
- Run `poetry run pytest` before considering any work complete
- Tests MUST pass before submitting changes
- If tests fail, fix them - never ignore failing tests
- Run tests after every significant change

**MANDATORY: Write Tests for New Functionality**
- Every new feature MUST have corresponding tests
- Every new function/method MUST be tested
- Test coverage expectations:
  - New REPL commands → add tests in `tests/test_cli.py`
  - New MCP servers → add tests in `tests/test_all_mcp_servers.py`
  - New local tools → add tests in `tests/test_tools.py`
  - New agent features → add tests in `tests/test_agent.py`
  - Integration scenarios → add tests in `tests/test_integration_*.py`
- Write tests BEFORE or DURING implementation, not after
- Tests are not optional - they are part of the feature

**MANDATORY: Update Tests When Changing Code**
- When modifying existing functionality, update corresponding tests
- If function signatures change, update all affected tests
- If behavior changes, update test assertions
- When fixing bugs, add regression tests to prevent recurrence
- Never leave broken tests - fix them immediately

**Test Quality Standards**
- Tests must be clear and readable
- Test both success and error cases
- Use descriptive test names: `test_<what>_<condition>_<expected_result>`
- Mock external dependencies (API calls, file system when appropriate)
- Keep tests fast - use fixtures and conftest.py for shared setup

### Linting and Formatting Requirements

**MANDATORY: Run Linter Before Committing**

Code must pass linting and formatting checks before committing:

```bash
# 1. Format code (auto-fix)
poetry run ruff format .

# 2. Check linting (auto-fix where possible)
poetry run ruff check . --fix
```

**Linter Configuration** (in `pyproject.toml`):
```toml
[tool.ruff]
line-length = 100
exclude = [".venv", "venv", "build", "dist", ".git"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]  # pycodestyle, pyflakes, isort, pyupgrade
ignore = ["E402"]
```

**Before Every Commit**:
1. Run `poetry run ruff format .` - auto-formats code
2. Run `poetry run ruff check . --fix` - auto-fixes linting issues
3. Fix any remaining linting errors manually

**Common Linting Rules**:
- Line length: 100 characters maximum
- Import sorting: stdlib → third-party → local
- No unused imports or variables
- Follow pycodestyle conventions (E rules)
- Use modern Python syntax (UP rules)

**CI Enforcement**:
- CI runs `ruff format --check` and `ruff check`
- PR will fail if code is not properly formatted
- Always run linter locally before pushing

### CE-Specific Guidelines

**Code Organization**:
- UI abstraction → `src/donkit_ragops/ui/` (protocol, adapters, styles)
- REPL implementations → `src/donkit_ragops/repl/`
- REPL commands → `src/donkit_ragops/repl/commands.py`
- UI text constants → `src/donkit_ragops/texts.py`
- Enterprise features → `src/donkit_ragops/enterprise/`
- MCP servers → `src/donkit_ragops/mcp/servers/`
- Local tools → `src/donkit_ragops/agent/local_tools/`
- LLM providers → `src/donkit_ragops/llm/providers/`

**Python Version**:
- Use Python 3.12+ features
- Maintain compatibility with 3.12, 3.13

**Code Style**:
- Line length: 100 characters (configured in `pyproject.toml`)
- Ruff for linting and formatting
- Follow existing patterns in codebase

**Imports**:
- Use `from __future__ import annotations` for forward references
- Organize imports: stdlib → third-party → local

## Version Management

**CRITICAL: Manual Version Updates Required**

The application version in `pyproject.toml` is managed manually and must be updated for every change:

```toml
[tool.poetry]
name = "donkit-ragops"
version = "0.3.18"  # Update this manually
```

**MANDATORY: Update Version Before Committing**

1. **Check current version in main branch**:
   ```bash
   git fetch origin main
   git show origin/main:pyproject.toml | grep "^version"
   ```

2. **Update version in your branch**:
   - Open `pyproject.toml`
   - Increment version according to change type:
     - **Patch** (0.3.18 → 0.3.19): Bug fixes, minor changes
     - **Minor** (0.3.18 → 0.4.0): New features, backwards compatible
     - **Major** (0.3.18 → 1.0.0): Breaking changes
   - Version MUST be higher than main branch version

3. **CI will fail if version is not updated**:
   - CI checks enforce version increment
   - Your PR will be blocked if version matches main branch
   - Always update version as part of your changes

**Version Update Workflow**:
```bash
# 1. Check version in main
git show origin/main:pyproject.toml | grep "^version"
# Output: version = "0.3.18"

# 2. Edit pyproject.toml and increment version
# Change: version = "0.3.18" → version = "0.3.19"

# 3. Commit with version update
git add pyproject.toml
git commit -m "feat: your feature description"

# 4. CI will pass because version was incremented
```

**Important Notes**:
- Never commit without updating version
- Version update is part of every PR
- Check main branch version before updating to avoid conflicts
- Version must follow semantic versioning (semver.org)

## Common Development Tasks

### Adding a New REPL Command

1. Create command class in `src/donkit_ragops/repl/commands.py`:
   ```python
   class MyCommand(ReplCommand):
       @property
       def name(self) -> str:
           return "mycommand"

       @property
       def aliases(self) -> list[str]:
           return ["mc"]

       @property
       def description(self) -> str:
           return "Description for help"

       async def execute(self, context: ReplContext) -> CommandResult:
           # Implementation here
           return CommandResult(messages=["Done!"])
   ```
2. Register in `create_default_registry()` or in REPL's `_setup_command_handlers()`
3. Add to help text in `src/donkit_ragops/texts.py`
4. Write tests

### Adding a New MCP Server

1. Create server file in `src/donkit_ragops/mcp/servers/your_server.py`
2. Implement using FastMCP framework
3. Add entry point in `pyproject.toml` [tool.poetry.scripts]
4. Add to unified server in `donkit_ragops_mcp.py`
5. Write tests in `tests/test_all_mcp_servers.py`

### Adding a New Local Tool

1. Add tool function in `src/donkit_ragops/agent/local_tools/tools.py`
2. Return an `AgentTool` dataclass with schema and handler
3. Register in `default_tools()` in `src/donkit_ragops/agent/agent.py`
4. Write tests in `tests/test_tools.py`

### Adding a New LLM Provider

1. Create provider in `src/donkit_ragops/llm/providers/your_provider.py`
2. Implement `LLMProviderAbstract` interface from `donkit-llm`
3. Register in `provider_factory.py`
4. Add configuration support in `config.py`
5. Update setup wizard in `setup_wizard.py`

### Running MCP Servers in Development

```bash
# Run individual server
poetry run ragops-chunker

# Run unified server (all tools)
poetry run donkit-ragops-mcp

# Test server with MCP inspector
npx @modelcontextprotocol/inspector poetry run donkit-ragops-mcp
```
