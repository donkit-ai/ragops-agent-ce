from __future__ import annotations

import warnings

# Suppress all warnings immediately, before any other imports
warnings.filterwarnings("ignore")
# Suppress warnings from importlib bootstrap (SWIG-related)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib._bootstrap")
# Suppress all DeprecationWarnings globally
warnings.simplefilter("ignore", DeprecationWarning)
import json
import os

import httpx
from fastmcp import FastMCP
from pydantic import BaseModel, Field


class SearchQueryArgs(BaseModel):
    query: str = Field(description="Search query text")
    k: int = Field(default=10, description="Number of top results to return")
    rag_service_url: str = Field(
        default="http://localhost:8000",
        description="RAG service base URL (e.g., http://localhost:8000)",
    )


server = FastMCP(
    "rag-query",
)


@server.tool(
    name="search_documents",
    description=(
        "Search for relevant documents in the RAG vector database. "
        "Returns the most relevant document chunks based on the query."
        "This tool just use retriever without any options. Result may be inaccurate."
        "Use this tool only for testing purposes. Not for answering questions."
    ).strip(),
)
async def search_documents(args: SearchQueryArgs) -> str:
    """Search for documents using RAG service HTTP API."""
    url = f"{args.rag_service_url.rstrip('/')}/api/query/search"

    # Prepare request payload (only query in body)
    payload = {"query": args.query}

    # k parameter goes as query parameter
    params = {"k": args.k}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, params=params)
            response.raise_for_status()

            result = response.json()

            # Format results for better readability
            formatted_results = {
                "query": args.query,
                "total_results": len(result) if isinstance(result, list) else 0,
                "documents": [],
            }

            # Result is a list of documents
            documents = result if isinstance(result, list) else []
            for doc in documents:
                metadata = doc.get("metadata", {})
                formatted_doc = {
                    "content": doc.get("page_content", "").strip(),
                    "metadata": metadata,
                }
                formatted_results["documents"].append(formatted_doc)

            return json.dumps(formatted_results, ensure_ascii=False, indent=2)

    except httpx.HTTPStatusError as e:
        error_detail = f"HTTP {e.response.status_code}: {e.response.text}"
        return json.dumps(
            {"error": "HTTP request failed", "detail": error_detail, "url": url},
            ensure_ascii=False,
            indent=2,
        )
    except httpx.RequestError as e:
        return json.dumps(
            {
                "error": "Request error",
                "detail": str(e),
                "url": url,
                "hint": "Make sure RAG service is running and accessible",
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {"error": "Unexpected error", "detail": str(e)}, ensure_ascii=False, indent=2
        )


@server.tool(
    name="get_rag_prompt",
    description=(
        "Get a formatted RAG prompt with retrieved context for a query. "
        "Returns ready-to-use prompt string with relevant document chunks embedded."
        "Use full rag-config for prompt generation."
        "Use this tool for answering."
    ).strip(),
)
async def get_rag_prompt(args: SearchQueryArgs) -> str:
    """Get formatted RAG prompt using RAG service HTTP API."""
    url = f"{args.rag_service_url.rstrip('/')}/api/query/prompt"

    # Prepare request payload
    payload = {"query": args.query}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

            # Response is plain text prompt
            prompt = response.text

            return prompt

    except httpx.HTTPStatusError as e:
        error_detail = f"HTTP {e.response.status_code}: {e.response.text}"
        return json.dumps(
            {"error": "HTTP request failed", "detail": error_detail, "url": url},
            ensure_ascii=False,
            indent=2,
        )
    except httpx.RequestError as e:
        return json.dumps(
            {
                "error": "Request error",
                "detail": str(e),
                "url": url,
                "hint": "Make sure RAG service is running and accessible",
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {"error": "Unexpected error", "detail": str(e)}, ensure_ascii=False, indent=2
        )


def main() -> None:
    server.run(
        transport="stdio",
        log_level=os.getenv("RAGOPS_LOG_LEVEL", "CRITICAL"),
        show_banner=False,
    )


if __name__ == "__main__":
    main()
