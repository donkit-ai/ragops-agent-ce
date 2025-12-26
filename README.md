# RAGOps Agent CE (Community Edition)

[![PyPI version](https://badge.fury.io/py/donkit-ragops.svg)](https://badge.fury.io/py/donkit-ragops)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An LLM-powered CLI agent that automates the creation and maintenance of Retrieval-Augmented Generation (RAG) pipelines. The agent orchestrates built-in tools and Model Context Protocol (MCP) servers to plan, chunk, and load documents into vector stores.

Built by [Donkit AI](https://donkit.ai) - Open Source RAG Infrastructure.

## Key Features

* **Interactive REPL** — Start an interactive session with readline history and autocompletion
* **Checklist-driven workflow** — The agent creates project checklists, asks for approval before each step, and tracks progress
* **Session-scoped checklists** — Only current session checklists appear in the UI
* **Integrated MCP servers** — Built-in support for full RAG build pipeline (planning, chunking, reading, vector loading)
* **Docker Compose orchestration** — Automated deployment of RAG infrastructure (vector DB, RAG service)
* **Multiple LLM providers** — Supports Vertex AI (Recommended), OpenAI, Azure OpenAI, Ollama, OpenRouter. Coming soon: Anthropic Claude

## Installation

### Option A: Using pip

```bash
pip install donkit-ragops
```

### Option B: Using Poetry (Recommended for Python 3.12+)

```bash
# Create a new project directory
mkdir ~/ragops-workspace
cd ~/ragops-workspace

# Initialize Poetry project
poetry init --no-interaction --python="^3.12"

# Add donkit-ragops
poetry add donkit-ragops

# Activate the virtual environment
poetry shell
```

After activation, you can run the agent with:
```bash
donkit-ragops
```

Or run directly without activating the shell:
```bash
poetry run donkit-ragops
```

## Quick Start

### Prerequisites

- **Python 3.12+** installed
- **Docker Desktop** installed and running (required for vector database)
  - **Windows users**: Docker Desktop with WSL2 backend is fully supported
- API key for your chosen LLM provider (Vertex AI, OpenAI, or Anthropic)

### Step 1: Install the package

```bash
pip install donkit-ragops
```

### Step 2: Run the agent (first time)

```bash
donkit-ragops
```

On first run, an **interactive setup wizard** will guide you through configuration:

1. Choose your LLM provider (Vertex AI, OpenAI, Anthropic, or Ollama)
2. Enter API key or credentials path
3. Optional: Configure log level
4. Configuration is saved to `.env` file automatically

**That's it!** No manual `.env` creation needed - the wizard handles everything.

### Alternative: Manual configuration

If you prefer to configure manually or reconfigure later:

```bash
# Run setup wizard again
donkit-ragops --setup
```

Or create a `.env` file manually in your working directory:

```bash
# Vertex AI (Google Cloud)
RAGOPS_LLM_PROVIDER=vertex
RAGOPS_VERTEX_CREDENTIALS=/path/to/service-account-key.json

# OpenAI
RAGOPS_LLM_PROVIDER=openai
RAGOPS_OPENAI_API_KEY=sk-...
RAGOPS_LLM_MODEL=gpt-4o-mini # Specify the OpenAI model to use
# RAGOPS_OPENAI_BASE_URL=https://api.openai.com/v1

# Anthropic Claude
RAGOPS_LLM_PROVIDER=anthropic
RAGOPS_ANTHROPIC_API_KEY=sk-ant-...

# Ollama (local)
RAGOPS_LLM_PROVIDER=ollama
RAGOPS_OLLAMA_BASE_URL=http://localhost:11434
```

### Step 3: Start using the agent

Tell the agent what you want to build:

```
you> Create a RAG pipeline for my documents in /Users/myname/Documents/work_docs
```

The agent will automatically:
- ✅ Create a `projects/<project_id>/` directory
- ✅ Plan RAG configuration
- ✅ Process and chunk your documents
- ✅ Start Qdrant vector database (via Docker)
- ✅ Load data into the vector store
- ✅ Deploy RAG query service

### What gets created

```
./
├── .env                          # Your configuration (auto-created by wizard)
└── projects/
    └── my-project-abc123/        # Auto-created by agent
        ├── compose/              # Docker Compose files
        │   ├── docker-compose.yml
        │   └── .env
        ├── chunks/               # Processed document chunks
        └── rag_config.json       # RAG configuration
```

## Usage

> **Note:** The command `ragops-agent` is also available as an alias for backward compatibility.
> 
> The agent starts in interactive REPL mode by default. Use subcommands like `ping` for specific actions.

### Interactive Mode (REPL)

```bash
# Start interactive session
donkit-ragops

# With specific provider
donkit-ragops -p vertexai

# With custom model
donkit-ragops -p openai -m gpt-4
```

### Command-line Options

- `-p, --provider` — Override LLM provider from settings
- `-m, --model` — Specify model name
- `-s, --system` — Custom system prompt
- `--show-checklist/--no-checklist` — Toggle checklist panel (default: shown)
- `--mcp-command` — Add custom MCP server (can be used multiple times)

### Subcommands

```bash
# Health check
donkit-ragops ping
```

### Environment Variables

#### LLM Provider Configuration
- `RAGOPS_LLM_PROVIDER` — LLM provider name (e.g., `openai`, `vertex`, `azure_openai`, `ollama`, `openrouter`)
- `RAGOPS_LLM_MODEL` — Specify model name (e.g., `gpt-4o-mini` for OpenAI, `gemini-2.5-flash` for Vertex)

#### OpenAI / OpenRouter / Ollama
- `RAGOPS_OPENAI_API_KEY` — OpenAI API key (also used for OpenRouter and Ollama)
- `RAGOPS_OPENAI_BASE_URL` — OpenAI base URL (default: https://api.openai.com/v1)
  - OpenRouter: `https://openrouter.ai/api/v1`
  - Ollama: `http://localhost:11434/v1`
- `RAGOPS_OPENAI_EMBEDDINGS_MODEL` — Embedding model name (default: text-embedding-3-small)

#### Azure OpenAI
- `RAGOPS_AZURE_OPENAI_API_KEY` — Azure OpenAI API key
- `RAGOPS_AZURE_OPENAI_ENDPOINT` — Azure OpenAI endpoint URL
- `RAGOPS_AZURE_OPENAI_API_VERSION` — Azure API version (default: 2025-03-01-preview)
- `RAGOPS_AZURE_OPENAI_DEPLOYMENT` — Azure deployment name for chat model
- `RAGOPS_AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` — Azure deployment name for embeddings model

#### Vertex AI (Google Cloud)
- `RAGOPS_VERTEX_CREDENTIALS` — Path to Vertex AI service account JSON
- `RAGOPS_VERTEX_PROJECT` — Google Cloud project ID (optional, extracted from credentials if not set)
- `RAGOPS_VERTEX_LOCATION` — Vertex AI location (default: us-central1)

#### Anthropic
- `RAGOPS_ANTHROPIC_API_KEY` — Anthropic API key

#### Logging
- `RAGOPS_LOG_LEVEL` — Logging level (default: INFO)
- `RAGOPS_MCP_COMMANDS` — Comma-separated list of MCP commands

## Agent Workflow

The agent follows a structured workflow:

1. **Language Detection** — Detects user's language from first message
2. **Project Creation** — Creates project directory structure
3. **Checklist Creation** — Generates task checklist in user's language
4. **Step-by-Step Execution**:
   - Asks for permission before each step
   - Marks item as `in_progress`
   - Executes the task using appropriate MCP tool
   - Reports results
   - Marks item as `completed`
5. **Deployment** — Sets up Docker Compose infrastructure
6. **Data Loading** — Loads documents into vector store

## MCP Servers

RAGOps Agent CE includes built-in MCP servers:

### `ragops-rag-planner`

Plans RAG pipeline configuration based on requirements.

```bash
# Example usage
donkit-ragops --mcp-command "ragops-rag-planner"
```

**Tools:**
- `plan_rag_config` — Generate RAG configuration from requirements

### `ragops-chunker`

Chunks documents for vector storage.

```bash
# Example usage
donkit-ragops --mcp-command "ragops-chunker"
```

**Tools:**
- `chunk_documents` — Split documents into chunks with configurable strategies
- `list_chunked_files` — List processed chunk files

### `ragops-vectorstore-loader`

Loads chunks into vector databases.

```bash
# Example usage
donkit-ragops --mcp-command "ragops-vectorstore-loader"
```

**Tools:**
- `vectorstore_load` — Load documents into Qdrant, Chroma, or Milvus
- `delete_from_vectorstore` — Remove documents from vector store

### `ragops-compose-manager`

Manages Docker Compose infrastructure.

```bash
# Example usage
donkit-ragops --mcp-command "ragops-compose-manager"
```

**Tools:**
- `init_project_compose` — Initialize Docker Compose for project
- `compose_up` — Start services
- `compose_down` — Stop services
- `compose_status` — Check service status
- `compose_logs` — View service logs

### `ragops-checklist`

Manages project checklists and progress tracking.

**Tools:**
- `create_checklist` — Create new checklist
- `get_checklist` — Get current checklist
- `update_checklist_item` — Update item status

### `donkit-ragops-mcp`

**Unified MCP server** that combines all servers above into a single endpoint

```bash
# Run unified server
donkit-ragops-mcp
```

**Claude Desktop configuration:**

```json
{
  "mcpServers": {
    "donkit-ragops-mcp": {
      "command": "donkit-ragops-mcp"
    }
  }
}
```

All tools are available with prefixes:
- `checklist_*` — Checklist management
- `chunker_*` — Document chunking  
- `compose_*` — Docker Compose orchestration
- `planner_*` — RAG configuration planning
- `query_*` — RAG query execution
- `reader_*` — Document reading/parsing
- `vectorstore_*` — Vector store operations

📖 **[Full documentation](docs/UNIFIED_SERVER.md)**

## Examples

### Basic RAG Pipeline

```bash
donkit-ragops
```

```
you> Create a RAG pipeline for customer support docs in ./docs folder
```

The agent will:
1. Create project structure
2. Plan RAG configuration
3. Chunk documents from `./docs`
4. Set up Qdrant + RAG service
5. Load data into vector store

### Custom Configuration

```bash
donkit-ragops -p vertexai -m gemini-1.5-pro
```

```
you> Build RAG for legal documents with 1000 token chunks and reranking
```

### Multiple Projects

Each project gets its own:
- Project directory (`projects/<project_id>`)
- Docker Compose setup
- Vector store collection
- Configuration

## Development

### Project Structure

```
donkit-ragops/
├── src/donkit_ragops/
│   ├── agent/          # LLM agent core
│   ├── llm/            # LLM provider integrations
│   ├── mcp/            # MCP servers and client
│   │   └── servers/    # Built-in MCP servers
│   ├── cli.py          # CLI commands
│   └── config.py       # Configuration
├── tests/
└── pyproject.toml
```

### Running Tests

```bash
poetry run pytest
```

### Code Quality

```bash
# Format code
poetry run ruff format .

# Lint code
poetry run ruff check .
```

## Docker Compose Services

The agent can deploy these services:

### Qdrant (Vector Database)

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
```

### RAG Service

```yaml
services:
  rag-service:
    image: donkit/rag-service:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URI=http://qdrant:6333
      - CONFIG=<base64-encoded-config>
```

## Architecture

```
┌─────────────────┐
│  RAGOps Agent   │
│     (CLI)       │
└────────┬────────┘
         │
         ├── MCP Servers ───────────────┐
         │   ├── ragops-rag-planner     │
         │   ├── ragops-chunker         │
         │   ├── ragops-vectorstore     │
         │   └── ragops-compose         │
         │                              │
         └── LLM Providers ─────────────┤
             ├── Vertex AI              │
             ├── OpenAI                 │
             ├── Anthropic              │
             └── Ollama                 │
                                        │
                                        ▼
                            ┌──────────────────┐
                            │ Docker Compose   │
                            ├──────────────────┤
                            │ • Qdrant         │
                            │ • RAG Service    │
                            └──────────────────┘
```

## Troubleshooting

### Windows + Docker Desktop with WSL2

The agent **fully supports Windows with Docker Desktop running in WSL2 mode**. Path conversion and Docker communication are handled automatically.

**Requirements:**
- Docker Desktop for Windows with WSL2 backend enabled
- Python 3.12+ installed on Windows (not inside WSL2)
- Run the agent from Windows PowerShell or Command Prompt

**How it works:**
- The agent detects WSL2 Docker automatically
- Windows paths like `C:\Users\...` are converted to `/mnt/c/Users/...` for Docker
- No manual configuration needed

**Troubleshooting:**

```bash
# 1. Verify Docker is accessible from Windows
docker info

# 2. Check Docker reports Linux (indicates WSL2)
docker info --format "{{.OperatingSystem}}"
# Should output: Docker Desktop (or similar with "linux")

# 3. If Docker commands fail, ensure Docker Desktop is running
```

### MCP Server Connection Issues

If MCP servers fail to start:

```bash
# Check MCP server logs
RAGOPS_LOG_LEVEL=DEBUG donkit-ragops
```

### Vector Store Connection

Ensure Docker services are running:

```bash
cd projects/<project_id>
docker-compose ps
docker-compose logs qdrant
```

### Credentials Issues

Verify your credentials:

```bash
# Vertex AI
gcloud auth application-default print-access-token

# OpenAI
echo $RAGOPS_OPENAI_API_KEY
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Related Projects

- [donkit-chunker](https://pypi.org/project/donkit-chunker/) — Document chunking library
- [donkit-vectorstore-loader](https://pypi.org/project/donkit-vectorstore-loader/) — Vector store loading utilities
- [donkit-read-engine](https://pypi.org/project/donkit-read-engine/) — Document parsing engine

---

Built with ❤️ by [Donkit AI](https://donkit.ai)
