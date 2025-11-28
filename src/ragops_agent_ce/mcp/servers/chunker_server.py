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
from pathlib import Path

from donkit.chunker import ChunkerConfig, DonkitChunker
from fastmcp import FastMCP
from loguru import logger
from pydantic import BaseModel, Field


class ChunkDocumentsArgs(BaseModel):
    source_path: str = Field(description="Path to the source directory with processed documents")
    project_id: str = Field(
        description="Project ID to store chunked documents "
        "in projects/<project_id>/processed/chunked/"
    )
    params: ChunkerConfig
    incremental: bool = Field(
        default=True,
        description="If True, only process new/modified files. If False, reprocess all files.",
    )


server = FastMCP(
    "rag-chunker",
)


@server.tool(
    name="chunk_documents",
    description=(
        "Reads documents from given paths, "
        "splits them into smaller text chunks, "
        "and saves to projects/<project_id>/processed/chunked/. "
        "Supports incremental processing - only new/modified files. "
        "Support only .json"
    ).strip(),
)
def chunk_documents(args: ChunkDocumentsArgs) -> str:
    logger.debug(f"chunk_documents called with: {args.model_dump()}")
    chunker = DonkitChunker(args.params)
    source_dir = Path(args.source_path)

    if not source_dir.exists() or not source_dir.is_dir():
        logger.error(f"Source path not found: {source_dir}")
        return json.dumps({"status": "error", "message": f"Source path not found: {source_dir}"})

    # Create output directory in project
    output_path = Path(f"projects/{args.project_id}/processed/chunked").resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output path: {output_path}")

    results = {
        "status": "success",
        "output_path": str(output_path),
        "successful": [],
        "failed": [],
        "skipped": [],
        "incremental": args.incremental,
    }

    # Get list of files to process
    files_to_process = [f for f in source_dir.iterdir() if f.is_file()]
    logger.debug(f"Found {len(files_to_process)} files to process")

    for file in files_to_process:
        logger.debug(f"Processing file: {file.name}")
        output_file = output_path / f"{file.stem}.json"

        # Check if we should skip this file (incremental mode)
        if args.incremental and output_file.exists():
            # Compare modification times
            if file.stat().st_mtime <= output_file.stat().st_mtime:
                results["skipped"].append(
                    {
                        "file": str(file),
                        "reason": "File not modified since last chunking",
                    }
                )
                continue

        try:
            logger.debug(f"Starting chunking for {file.name}")
            chunked_documents = chunker.chunk_file(
                file_path=str(file),
            )
            logger.debug(f"Chunking complete for {file.name}, got {len(chunked_documents)} chunks")

            payload = [
                {"page_content": chunk.page_content, "metadata": chunk.metadata}
                for chunk in chunked_documents
            ]

            logger.debug(f"Writing chunks to {output_file.name}")
            output_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug(f"Write complete for {file.name}")

            results["successful"].append(
                {
                    "file": str(file),
                    "output": str(output_file),
                    "chunks_count": len(chunked_documents),
                }
            )
        except Exception as e:
            logger.error(f"Failed to process {file.name}: {e}")
            results["failed"].append({"file": str(file), "error": str(e)})

    # Add summary
    results["message"] = (
        f"Processed: {len(results['successful'])}, "
        f"Skipped: {len(results['skipped'])}, "
        f"Failed: {len(results['failed'])}"
    )

    # Return results as JSON string
    return json.dumps(results, ensure_ascii=False, indent=2)


def main() -> None:
    server.run(
        transport="stdio",
        log_level=os.getenv("RAGOPS_LOG_LEVEL", "CRITICAL"),
        show_banner=False,
    )


if __name__ == "__main__":
    main()
