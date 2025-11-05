from __future__ import annotations

import os
import re

from fastmcp import FastMCP
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from ragops_agent_ce.schemas.config_schemas import RagConfig


class RagConfigPlanArgs(BaseModel):
    project_id: str
    goal: str
    rag_config: RagConfig = Field(default_factory=RagConfig)

    @model_validator(mode="after")
    def _set_default_collection_name(self) -> RagConfigPlanArgs:
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
        return self


server = FastMCP(
    "rag-config-planner",
    log_level=os.getenv("RAGOPS_LOG_LEVEL", "CRITICAL"),  # noqa
)


@server.tool(
    name="rag_config_plan",
    description=(
        "Suggest a RAG configuration (vectorstore/chunking/retriever/ranker) "
        "for the given project and sources."
    ),
)
async def rag_config_plan(args: RagConfigPlanArgs) -> str:
    plan = args.rag_config.model_dump_json()
    return plan


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
