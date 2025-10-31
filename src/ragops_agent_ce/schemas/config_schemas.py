"""
Shared configuration schemas for RagOps MCP servers.

This module contains common Pydantic models used across multiple servers
to ensure consistency and avoid duplication.

Shared Models:
    - EmbedderType: Enum of supported embedding providers
    - GenerationModelType: Enum of supported generation model providers
    - SplitType: Enum of text splitting methods for chunking
    - RetrieverType: Enum of supported vector database types
    - Embedder: Embedder configuration
    - ChunkingConfig: Document chunking configuration
    - RetrieverOptions: Retriever configuration options
    - RagConfig: Unified RAG configuration (base schema)
"""

from enum import StrEnum
from enum import auto
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

# ============================================================================
# Enums
# ============================================================================


DEFAULT_PROMPT = """
You are an intelligent assistant designed to help users by
providing accurate and concise answers based on the given context.

**Instructions:**
- Always respond in the same language as the user question
- Base your answers on the provided context
- If the context does not contain relevant information, clearly state that
- Be clear, concise, and helpful

**Context**: {context}

**Question**: {question}""".strip()


class EmbedderType(StrEnum):
    """Supported embedding providers."""

    OPENAI = auto()
    VERTEX = auto()
    AZURE_OPENAI = auto()


class GenerationModelType(StrEnum):
    """Supported generation model providers."""

    # GEMINI = auto()
    OPENAI = auto()
    AZURE_OPENAI = auto()
    # CLAUDE = auto()
    VERTEX = auto()


class SplitType(StrEnum):
    """Text splitting methods for chunking."""

    CHARACTER = auto()
    SENTENCE = auto()
    PARAGRAPH = auto()
    SEMANTIC = auto()
    JSON = auto()


class RetrieverType(StrEnum):
    """Supported vector database types."""

    # MILVUS = auto()
    QDRANT = auto()
    # CHROMA = auto()


# ============================================================================
# Environment variable mappings and descriptions
# ============================================================================


MODEL_ENV_MAPPING = """
        for openai
            - RAGOPS_OPENAI_API_KEY
        for azure openai
            - RAGOPS_AZURE_OPENAI_API_KEY
            - RAGOPS_AZURE_OPENAI_ENDPOINT
            - RAGOPS_AZURE_OPENAI_API_VERSION
            - RAGOPS_AZURE_OPENAI_DEPLOYMENT (for chat completions)
            - RAGOPS_AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT (for embeddings)
        for claude
            - RAGOPS_CLAUDE_API_KEY
        for vertex
            - RAGOPS_VERTEX_CREDENTIALS_JSON
            or
            - RAGOPS_VERTEX_PROJECT_ID
            - RAGOPS_VERTEX_LOCATION
            - RAGOPS_VERTEX_API_KEY   
"""

GENERATION_MODEL_TYPE_DESCRIPTION = f"""
    IMPORTANT!
    for any generation model must be set environment variables:
        {MODEL_ENV_MAPPING}
    """

EMBEDDER_TYPE_DESCRIPTION = f"""
    IMPORTANT!
    for any embedder must be set environment variables:
        {MODEL_ENV_MAPPING}
"""


# ============================================================================
# Configuration Models
# ============================================================================


class Embedder(BaseModel):
    """Embedder configuration."""

    embedder_type: str = Field(default=EmbedderType.VERTEX, description=EMBEDDER_TYPE_DESCRIPTION)


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""

    split_type: SplitType = Field(
        default=SplitType.JSON, description="Use only JSON for .json files chunking"
    )
    chunk_size: int = Field(default=250)
    chunk_overlap: int = Field(default=0)


class RetrieverOptions(BaseModel):
    """Retriever configuration options."""

    collection_name: str | None = Field(
        default=None,
        description=(
            "Collection name to use in vector DB. "
            "If not provided, defaults to project_id."
            "For 'milvus' the first character of a "
            "collection name must be an underscore or letter ."
        ),
    )
    composite_query_detection: bool = Field(
        default=False, description="Split composite query into several simple questions"
    )
    partial_search: bool = Field(
        default=True, description="Search by small chunks and take it`s neighbors."
    )
    query_rewrite: bool = Field(default=True, description="Enable rewriting user query")
    max_retrieved_docs: int = Field(
        default=5, description="Maximum number of documents to retrieve."
    )
    ranked_documents: int = Field(default=1, description="Number of ranked documents to return.")
    minimum_relevance: float = Field(
        default=0.5, description="Minimum relevance score for ranked documents."
    )


class RagConfig(BaseModel):
    """
    Unified RAG configuration schema.
    """

    files_path: str = Field(
        description=(
            "Path to the folder with processed documents. "
            "This directory is created after document processing (process_documents tool). "
            "Format: projects/<project_id>/processed/"
        )
    )
    generation_prompt: str = Field(
        default=DEFAULT_PROMPT,
        description="Prompt for generation model. Generate you own based on project goal.",
    )
    embedder: Embedder = Field(default_factory=Embedder)
    ranker: bool = Field(default=False)
    db_type: Literal["qdrant", "chroma", "milvus"] = Field(default="qdrant")
    retriever_options: RetrieverOptions = Field(default_factory=RetrieverOptions)
    chunking_options: ChunkingConfig = Field(default_factory=ChunkingConfig)
    generation_model_type: GenerationModelType = Field(
        description=GENERATION_MODEL_TYPE_DESCRIPTION
    )
    generation_model_name: str = Field(
        description="Generation model name. must be model of selected generation model type"
    )
    database_uri: str = Field(description="Vector database URI inside DOCKER")
