from __future__ import annotations

import json
import os
from pathlib import Path

import mcp
from donkit.chunker import ChunkerConfig, DonkitChunker
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
        description=("If True, only process new/modified files. " "If False, reprocess all files."),
    )


server = mcp.server.FastMCP(
    "rag-chunker",
    log_level=os.getenv("RAGOPS_LOG_LEVEL", "CRITICAL"),  # noqa
)


@server.tool(
    name="chunk_documents",
    description=(
        "Reads documents from given paths, "
        "splits them into smaller text chunks, "
        "and saves to projects/<project_id>/processed/chunked/. "
        "Supports incremental processing - only new/modified files. "
        "Support only text files eg. .txt, .json"
    ).strip(),
)
async def chunk_documents(args: ChunkDocumentsArgs) -> mcp.types.TextContent:
    chunker = DonkitChunker(args.params)
    source_dir = Path(args.source_path)

    if not source_dir.exists() or not source_dir.is_dir():
        return mcp.types.TextContent(
            type="text",
            text=json.dumps({"status": "error", "message": f"Source path not found: {source_dir}"}),
        )

    # Create output directory in project
    output_path = Path(f"projects/{args.project_id}/processed/chunked").resolve()
    output_path.mkdir(parents=True, exist_ok=True)

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

    for file in files_to_process:
        output_file = output_path / f"{file.name}.json"

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
            chunked_documents = chunker.chunk_file(
                file_path=str(file),
            )
            payload = [
                {"page_content": chunk.page_content, "metadata": chunk.metadata}
                for chunk in chunked_documents
            ]
            output_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            results["successful"].append(
                {
                    "file": str(file),
                    "output": str(output_file),
                    "chunks_count": len(chunked_documents),
                }
            )
        except Exception as e:
            results["failed"].append({"file": str(file), "error": str(e)})

    # Add summary
    results["message"] = (
        f"Processed: {len(results['successful'])}, "
        f"Skipped: {len(results['skipped'])}, "
        f"Failed: {len(results['failed'])}"
    )

    # Return results as JSON string
    return mcp.types.TextContent(
        type="text", text=json.dumps(results, ensure_ascii=False, indent=2)
    )


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
