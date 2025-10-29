from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import mcp
from donkit.read_engine.read_engine import DonkitReader
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

load_dotenv()


class ProcessDocumentsArgs(BaseModel):
    source_path: str = Field(description="Path to the source directory with documents")
    output_type: Literal["text", "json", "markdown"] = Field(
        default="json",
        description="Output format for processed documents",
    )


server = mcp.server.FastMCP(
    "rag-read-engine",
    log_level=os.getenv("RAGOPS_LOG_LEVEL", "CRITICAL"),  # noqa
)


@server.tool(
    name="process_documents",
    description=(
        "Process documents from various formats (PDF, DOCX, PPTX, XLSX, images, etc.) "
        "and convert them to text/json/markdown. "
        "Supports: PDF, DOCX/DOC, PPTX, XLSX/XLS, TXT, CSV, JSON, Images (PNG/JPG). "
        "Output is saved to 'processed/' subdirectory inside source_path. "
        "Returns the path to the processed directory which can be used by chunk_documents tool."
        "Don't use this tool to get documents content! It returns only path to processed directory."
    ).strip(),
)
async def process_documents(args: ProcessDocumentsArgs) -> mcp.types.TextContent:
    """Process documents from source directory using DonkitReader.

    This tool converts various document formats to text-based formats that can be
    processed by the chunker. It creates a 'processed/' subdirectory with converted files.
    """
    reader = DonkitReader()
    logger.info(reader.readers)
    source_dir = Path(args.source_path)
    if not source_dir.exists() or not source_dir.is_dir():
        return mcp.types.TextContent(
            type="text",
            text=json.dumps(
                {"status": "error", "message": f"Source directory not found: {source_dir}"}
            ),
        )

    # Get list of supported files
    supported_extensions = set(reader.readers.keys())
    files_to_process = [
        f for f in source_dir.rglob("*") if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if not files_to_process:
        return mcp.types.TextContent(
            type="text",
            text=json.dumps(
                {
                    "status": "error",
                    "message": f"No supported files found in {source_dir}. "
                    f"Supported: {list(supported_extensions)}",
                }
            ),
        )

    processed_files: list[str] = []
    failed_files: list[dict[str, str]] = []

    # Process each file
    for file_path in files_to_process:
        try:
            output_path = reader.read_document(
                str(file_path),
                output_type=args.output_type,  # type: ignore
            )
            processed_files.append(output_path)
            logger.info(f"✓ Processed: {file_path.name} -> {output_path}")
        except Exception as e:
            error_msg = str(e)
            failed_files.append({"file": str(file_path), "error": error_msg})
            logger.error(f"✗ Failed to process {file_path.name}: {error_msg}")

    # Determine output directory (all files should be in same processed/ dir)
    if processed_files:
        # Get the processed directory path from first processed file
        first_processed = Path(processed_files[0])
        output_dir = str(first_processed.parent)
    else:
        output_dir = str(source_dir / "processed")

    result = {
        "status": "success" if processed_files else "partial_failure",
        "output_directory": output_dir,
        "processed_count": len(processed_files),
        "failed_count": len(failed_files),
        "processed_files": processed_files[:10],  # Limit to first 10 for readability
        "failed_files": failed_files[:10] if failed_files else [],
        "message": (
            f"Processed {len(processed_files)} files. "
            f"Failed: {len(failed_files)}. "
            f"Output saved to: {output_dir}"
        ),
    }

    return mcp.types.TextContent(type="text", text=json.dumps(result, indent=2))


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
