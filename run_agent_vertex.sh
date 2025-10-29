#!/bin/bash

# Script to run the RagOps Agent in REPL mode with Vertex AI and all MCP tools.

# Check for .env file and load it if it exists
if [ -f .env ]; then
  export $(cat .env | sed 's/#.*//g' | xargs)
fi

# Check if Vertex AI credentials are set
if [ -z "$RAGOPS_VERTEX_CREDENTIALS" ] || [ ! -f "$RAGOPS_VERTEX_CREDENTIALS" ]; then
    echo "Error: RAGOPS_VERTEX_CREDENTIALS is not set or the file does not exist."
    echo "Please set it in your .env file."
    exit 1
fi

# Run the agent
poetry run ragops-agent run -p vertexai
