import json
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

import mcp
from donkit.embeddings import get_vertexai_embeddings
from donkit.vectorstore_loader import create_vectorstore_loader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from pydantic import BaseModel, Field, validate_call


def create_embedder(embedder_type: str) -> Embeddings:
    def __get_vertex_credentials():
        creds_path = os.getenv("RAGOPS_VERTEX_CREDENTIALS")
        if not creds_path:
            raise ValueError("env variable 'RAGOPS_VERTEX_CREDENTIALS' is not set")
        with open(creds_path) as f:
            credentials_data = json.load(f)
        return credentials_data

    def __check_openai():
        api_key = os.getenv("OPENAI_API_KEY", os.getenv("RAGOPS_OPENAI_API_KEY"))
        if not api_key:
            raise ValueError("env variable 'OPENAI_API_KEY' or 'RAGOPS_OPENAI_API_KEY' is not set")
        base_url = os.getenv("OPENAI_BASE_URL", os.getenv("RAGOPS_OPENAI_BASE_URL"))
        model = os.getenv("OPENAI_EMBEDDINGS_MODEL", os.getenv("RAGOPS_OPENAI_EMBEDDINGS_MODEL"))
        return api_key, base_url, model

    def __check_azure():
        api_key = os.getenv("AZURE_OPENAI_API_KEY", os.getenv("RAGOPS_AZURE_OPENAI_API_KEY"))
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", os.getenv("RAGOPS_AZURE_OPENAI_ENDPOINT"))
        api_version = os.getenv(
            "AZURE_OPENAI_API_VERSION", os.getenv("RAGOPS_AZURE_OPENAI_API_VERSION")
        )
        deployment = os.getenv(
            "RAGOPS_AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
            os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        )
        if not api_key or not endpoint or not api_version:
            raise ValueError(
                "env variables 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT' "
                "and 'AZURE_OPENAI_API_VERSION' must be set"
            )
        return api_key, endpoint, api_version, deployment

    if embedder_type == "openai":
        api_key, base_url, model = __check_openai()
        return OpenAIEmbeddings(
            api_key=api_key,  # noqa
            openai_api_base=base_url,
            model=model or "text-embedding-3-small",
        )
    elif embedder_type == "vertex":
        credentials = __get_vertex_credentials()
        return get_vertexai_embeddings(
            credentials_data=credentials,
        )
    elif embedder_type == "azure_openai":
        api_key, endpoint, api_version, deployment = __check_azure()
        return AzureOpenAIEmbeddings(
            openai_api_key=api_key,
            azure_endpoint=endpoint,
            openai_api_version=api_version,
            deployment=deployment
            if deployment and "embed" in deployment
            else "text-embedding-ada-002",
        )
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


class VectorstoreParams(BaseModel):
    backend: Literal["qdrant", "chroma", "milvus"] = Field(default="qdrant")
    embedder_type: str = Field(default="vertex")
    collection_name: str = Field(description="Use collection name from rag config")
    database_uri: str = Field(
        default="http://localhost:6333", description="local vectorstore database URI outside docker"
    )


class VectorstoreLoadArgs(BaseModel):
    chunks_path: str = Field(
        description=(
            "Path to chunked files: directory, single JSON file, or comma-separated list. "
            "Examples: '/path/to/chunked/', '/path/file.json', "
            "'/path/file1.json,/path/file2.json'"
        )
    )
    params: VectorstoreParams


server = mcp.server.FastMCP(
    "rag-vectorstore-loader",
    log_level=os.getenv("RAGOPS_LOG_LEVEL", "CRITICAL"),
)


@server.tool(
    name="vectorstore_load",
    description=(
        "Loads document chunks from JSON files into a specified vectorstore collection. "
        "Supports: directory (all JSON files), single file, or comma-separated file list. "
        "For INCREMENTAL loading (adding new files to existing RAG): pass specific file path(s) "
        "like '/path/new_file.json' or '/path/file1.json,/path/file2.json', NOT directory path. "
        "Use list_directory on chunked folder to find which files to load."
    ),
)
@validate_call
async def vectorstore_load(
    chunks_path: str,
    params: VectorstoreParams,
) -> mcp.types.TextContent:
    if "localhost" not in params.database_uri:
        return mcp.types.TextContent(
            type="text",
            text="Error: database URI must be outside "
            "docker like 'localhost' or '127.0.0.1' or '0.0.0.0'",
        )

    # Determine files to load based on chunks_path
    json_files: list[Path] = []

    # Check if it's a comma-separated list
    if "," in chunks_path:
        file_paths = [p.strip() for p in chunks_path.split(",")]
        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            if file_path.exists() and file_path.is_file() and file_path.suffix == ".json":
                json_files.append(file_path)
    # Check if it's a single file
    elif Path(chunks_path).is_file():
        file_path = Path(chunks_path)
        if file_path.suffix == ".json":
            json_files.append(file_path)
        else:
            return mcp.types.TextContent(
                type="text", text=f"Error: file must be JSON, got {file_path.suffix}"
            )
    # Check if it's a directory
    elif Path(chunks_path).is_dir():
        dir_path = Path(chunks_path)
        json_files = sorted([f for f in dir_path.iterdir() if f.is_file() and f.suffix == ".json"])
    else:
        return mcp.types.TextContent(type="text", text=f"Error: path not found: {chunks_path}")

    if not json_files:
        return mcp.types.TextContent(
            type="text", text=f"Error: no JSON files found in {chunks_path}"
        )

    try:
        embeddings = create_embedder(params.embedder_type)
        loader = create_vectorstore_loader(
            db_type=params.backend,
            embeddings=embeddings,
            collection_name=params.collection_name,
            database_uri=params.database_uri,
        )
    except ValueError as e:
        return mcp.types.TextContent(type="text", text=f"Error initializing vectorstore: {e}")
    except Exception as e:
        return mcp.types.TextContent(
            type="text", text=f"Unexpected error during initialization: {e}"
        )

    # Load files one by one for detailed tracking
    total_chunks_loaded = 0
    successful_files: list[tuple[str, int]] = []  # (filename, chunk_count)
    failed_files: list[tuple[str, str]] = []  # (filename, error_message)

    for file in json_files:
        try:
            # Read and parse JSON file
            with file.open("r", encoding="utf-8") as f:
                chunks = json.load(f)

            if not isinstance(chunks, list):
                failed_files.append((file.name, f"expected list, got {type(chunks).__name__}"))
                continue

            # Convert chunks to Document objects
            documents: list[Document] = []
            for chunk_data in chunks:
                if not isinstance(chunk_data, dict) or "page_content" not in chunk_data:
                    failed_files.append((file.name, "invalid chunk format"))
                    break

                doc = Document(
                    page_content=chunk_data["page_content"],
                    metadata=chunk_data.get("metadata", {}),
                )
                documents.append(doc)

            if not documents:
                failed_files.append((file.name, "no valid chunks found"))
                continue

            try:
                task_id = uuid4()
                loader.load_documents(task_id=task_id, documents=documents)

                chunk_count = len(documents)
                total_chunks_loaded += chunk_count
                successful_files.append((file.name, chunk_count))

            except Exception as e:
                failed_files.append((file.name, f"vectorstore error: {str(e)}"))

        except FileNotFoundError:
            failed_files.append((file.name, "file not found"))
            raise
        except json.JSONDecodeError as e:
            failed_files.append((file.name, f"invalid JSON: {str(e)}"))
            raise
        except Exception as e:
            failed_files.append((file.name, f"unexpected error: {str(e)}"))
            raise

    collection_name = params.collection_name
    backend = params.backend

    summary_lines = [
        f"Vectorstore loading completed for collection '{collection_name}' ({backend}):",
        "",
        f"✓ Successfully loaded: {len(successful_files)} file(s), {total_chunks_loaded} chunk(s)",
    ]

    if successful_files:
        summary_lines.append("")
        summary_lines.append("Successful files:")
        for filename, count in successful_files:
            summary_lines.append(f"  • {filename}: {count} chunks")

    if failed_files:
        summary_lines.append("")
        summary_lines.append(f"✗ Failed: {len(failed_files)} file(s)")
        summary_lines.append("Failed files:")
        for filename, error in failed_files:
            summary_lines.append(f"  • {filename}: {error}")

    return mcp.types.TextContent(type="text", text="\n".join(summary_lines))


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
