from __future__ import annotations

import warnings

# Suppress all warnings immediately, before any other imports
warnings.filterwarnings("ignore")
# Suppress warnings from importlib bootstrap (SWIG-related)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib._bootstrap")
# Suppress all DeprecationWarnings globally
warnings.simplefilter("ignore", DeprecationWarning)
import os
import re
from typing import Self

from donkit_ragops.schemas.config_schemas import RagConfig
from fastmcp import FastMCP
from pydantic import BaseModel, Field, model_validator


class RagConfigPlanArgs(BaseModel):
    project_id: str
    rag_config: RagConfig = Field(default_factory=RagConfig)

    @model_validator(mode="after")
    def _set_default_collection_name(self) -> Self:
        """Ensure retriever_options.collection_name is set.
        If missing/empty, use project_id as a sensible default.
        For Milvus, ensure collection name starts with underscore or letter.
        """
        if not getattr(self.rag_config.retriever_options, "collection_name", None):
            self.rag_config.retriever_options.collection_name = self.project_id

        # Fix collection name for Milvus if needed
        if self.rag_config.db_type == "milvus":
            collection_name = self.rag_config.retriever_options.collection_name
            if not re.match(r"^[a-zA-Z_]", collection_name):
                self.rag_config.retriever_options.collection_name = f"_{collection_name}"

        # Ensure embedder.embedder_type is preserved if explicitly set
        # If embedder was passed but embedder_type is default (vertex),
        # check if we should preserve it. This prevents overwriting user's choice
        return self


server = FastMCP(
    "rag-config-planner",
)


@server.tool(
    name="rag_config_plan",
    description=(
        "Suggest a RAG configuration (vectorstore/chunking/retriever/ranker) "
        "for the given project and sources. "
        "IMPORTANT: When passing rag_config parameter, ensure embedder.embedder_type "
        "is explicitly set to match user's choice (openai, vertex, or azure_openai). "
        "Do not rely on defaults."
    ),
)
async def rag_config_plan(args: RagConfigPlanArgs) -> str:
    plan = args.rag_config.model_dump_json()
    return plan


def main() -> None:
    server.run(
        transport="stdio",
        log_level=os.getenv("RAGOPS_LOG_LEVEL", "CRITICAL"),
        show_banner=False,
    )


if __name__ == "__main__":
    main()
