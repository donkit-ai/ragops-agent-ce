from __future__ import annotations

import os

import mcp
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from ragops_agent_ce.schemas.config_schemas import RagConfig

load_dotenv()


class RagConfigPlanArgs(BaseModel):
    project_id: str
    goal: str
    rag_config: RagConfig = Field(default_factory=RagConfig)

    @model_validator(mode="after")
    def _set_default_collection_name(self) -> RagConfigPlanArgs:
        """Ensure retriever_options.collection_name is set.
        If missing/empty, use project_id as a sensible default.
        """
        if not getattr(self.rag_config.retriever_options, "collection_name", None):
            self.rag_config.retriever_options.collection_name = self.project_id
        return self


server = mcp.server.FastMCP(
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
async def rag_config_plan(args: RagConfigPlanArgs) -> mcp.types.TextContent:
    plan = args.rag_config.model_dump_json()
    return mcp.types.TextContent(type="text", text=plan)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
