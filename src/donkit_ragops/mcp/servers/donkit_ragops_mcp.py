"""
Unified MCP Server - combines all RagOps MCP servers into one.

This server uses FastMCP's mounting feature to combine multiple servers:
- chunker: Document chunking utilities
- compose: Docker compose service management
- planner: RAG config planning
- query: RAG query execution
- evaluation: RAG evaluation (batch CSV processing)
- reader: Document reading/parsing
- vectorstore: Vector store loading

Note: Checklist management is now handled by built-in agent tools, not MCP.

Usage:
    python -m donkit_ragops.mcp.servers.donkit_ragops_mcp

Or in Claude Desktop config:
    {
      "mcpServers": {
        "donkit-ragops-mcp": {
          "command": "python",
          "args": ["-m", "donkit_ragops.mcp.servers.donkit_ragops_mcp"]
        }
      }
    }
"""

import os
import warnings

# Suppress all warnings immediately, before any other imports
warnings.filterwarnings("ignore")
# Suppress warnings from importlib bootstrap (SWIG-related)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib._bootstrap")
# Suppress all DeprecationWarnings globally
warnings.simplefilter("ignore", DeprecationWarning)

from fastmcp import FastMCP

# Import all subservers (checklist removed - now handled by built-in agent tools)
from .chunker_server import server as chunker_server
from .compose_manager_server import server as compose_server
from .planner_server import server as planner_server
from .rag_evaluation_server import server as evaluation_server
from .rag_query_server import server as query_server
from .read_engine_server import server as reader_server
from .vectorstore_loader_server import server as vectorstore_server

# Create unified server
unified_server = FastMCP(name="donkit-ragops-mcp")

# Mount all servers with appropriate prefixes
# Using mount() for live linking - servers remain independent
unified_server.mount(chunker_server, prefix="chunker")
unified_server.mount(compose_server, prefix="compose")
unified_server.mount(planner_server, prefix="planner")
unified_server.mount(query_server, prefix="query")
unified_server.mount(evaluation_server, prefix="evaluation")
unified_server.mount(reader_server, prefix="reader")
unified_server.mount(vectorstore_server, prefix="vectorstore")


def main() -> None:
    """Run the unified server."""
    unified_server.run(
        transport="stdio",
        log_level=os.getenv("RAGOPS_LOG_LEVEL", "CRITICAL"),
        show_banner=False,
    )


if __name__ == "__main__":
    main()
