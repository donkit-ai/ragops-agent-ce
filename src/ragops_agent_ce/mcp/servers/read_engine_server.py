from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Literal

import mcp
from donkit.read_engine.read_engine import DonkitReader
from loguru import logger
from pydantic import BaseModel
from pydantic import Field


class ProcessDocumentsArgs(BaseModel):
    source_path: str = Field(
        description=(
            "Path to source: directory, single file, or comma-separated list of files. "
            "Examples: '/path/to/folder', '/path/to/file.pdf', "
            "'/path/file1.pdf,/path/file2.docx'"
        )
    )
    project_id: str = Field(
        description="Project ID to store processed documents in projects/<project_id>/processed/"
    )
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
        "Can process: directory (recursively), single file, or comma-separated file list. "
        "Output is saved to projects/<project_id>/processed/ directory. "
        "Returns the path to the processed directory which can be used by chunk_documents tool. "
        "Don't use this tool to get documents content! It returns only path to processed directory."
    ).strip(),
)
async def process_documents(args: ProcessDocumentsArgs) -> mcp.types.TextContent:
    """Process documents from source directory, file, or file list using DonkitReader.

    This tool converts various document formats to text-based formats that can be
    processed by the chunker. It creates a 'processed/' subdirectory with converted files.
    """
    # Log raw arguments for debugging
    logger.debug(f"Raw args: {args}")
    logger.debug(f"source_path type: {type(args.source_path)}, value: {repr(args.source_path)}")
    logger.debug(f"project_id: {args.project_id}")

    reader = DonkitReader()
    logger.debug(reader.readers)
    supported_extensions = set(reader.readers.keys())

    # Determine files to process based on source_path
    files_to_process: list[Path] = []
    source_path_str = args.source_path.strip()
    # Normalize Unicode for macOS compatibility (NFD normalization)
    source_path_str = unicodedata.normalize("NFD", source_path_str)
    logger.debug(f"Processing source_path after strip and normalization: {repr(source_path_str)}")
    source_path = Path(source_path_str)
    logger.debug(f"Created Path object: {source_path}, exists: {source_path.exists()}")

    # Check if it's a single file (prioritize over comma-separated list)
    if source_path.is_file():
        if source_path.suffix.lower() in supported_extensions:
            files_to_process.append(source_path)
        else:
            return mcp.types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "status": "error",
                        "message": f"File format not supported: {source_path.suffix}. "
                        f"Supported: {list(supported_extensions)}",
                    },
                    ensure_ascii=False,
                ),
            )
    # Check if it's a directory
    elif source_path.is_dir():
        files_to_process = [
            f
            for f in source_path.rglob("*")
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
    # Check if it's a comma-separated list of files
    elif "," in args.source_path:
        file_paths = [p.strip() for p in args.source_path.split(",")]
        for file_path_str in file_paths:
            # Normalize Unicode for macOS compatibility
            file_path_str = unicodedata.normalize("NFD", file_path_str)
            file_path = Path(file_path_str)
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files_to_process.append(file_path)
            else:
                logger.warning(
                    f"File not supported or not found: {file_path}. "
                    f"Supported: {list(supported_extensions)}"
                )
    # Try fuzzy match: find similar files and suggest to user
    elif source_path.parent.exists():
        # Normalize multiple spaces in the provided path
        normalized_name = re.sub(r"\s+", " ", source_path.name)
        logger.info(f"Looking for similar files. Normalized name: {repr(normalized_name)}")

        similar_files = []
        for file_path in source_path.parent.iterdir():
            if not file_path.is_file():
                continue
            # Normalize spaces in actual filename
            actual_normalized = re.sub(r"\s+", " ", file_path.name)
            if actual_normalized == normalized_name:
                similar_files.append(str(file_path))

        if similar_files:
            return mcp.types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "status": "error",
                        "message": f"File not found: {source_path.name}\n\n"
                        f"Found similar file(s):\n"
                        + "\n".join(f"- {f}" for f in similar_files)
                        + "\n\n"
                        "Please provide the exact file path from the list above.",
                        "similar_files": similar_files,
                    },
                    ensure_ascii=False,
                ),
            )

    # Final check: if still no files found
    if not files_to_process:
        logger.error(
            f"No files to process. "
            f"Checked: is_file={source_path.is_file()}, "
            f"is_dir={source_path.is_dir()}, "
            f"exists={source_path.exists()}, "
            f"path={source_path}"
        )
        return mcp.types.TextContent(
            type="text",
            text=json.dumps(
                {
                    "status": "error",
                    "message": f"No supported files found in {source_path}. "
                    f"Supported: {list(supported_extensions)}",
                },
                ensure_ascii=False,
            ),
        )

    # Create project output directory
    project_output_dir = Path(f"projects/{args.project_id}/processed").resolve()
    project_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {project_output_dir}")

    processed_files: list[str] = []
    failed_files: list[dict[str, str]] = []

    # Process each file - DonkitReader will save directly to project directory
    for file_path in files_to_process:
        try:
            logger.info(f"Processing file: {file_path}")
            # Pass output_dir to save directly to project directory (no moving needed)
            output_path = reader.read_document(
                str(file_path),
                output_type=args.output_type,  # type: ignore
                output_dir=str(project_output_dir),
            )
            logger.debug(f"DonkitReader saved to: {output_path}")
            processed_files.append(output_path)
            logger.info(f"✓ Processed: {file_path.name} -> {output_path}")
        except Exception as e:
            error_msg = str(e)
            failed_files.append({"file": str(file_path), "error": error_msg})
            logger.error(f"✗ Failed to process {file_path.name}: {error_msg}", exc_info=True)

    # Clean up empty temp directories (safe for Windows)
    for file_path in files_to_process:
        temp_dir = file_path.parent / "processed"
        try:
            if temp_dir.exists() and temp_dir.is_dir() and not list(temp_dir.iterdir()):
                temp_dir.rmdir()
                logger.info(f"Cleaned up empty temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not clean up temp directory {temp_dir}: {e}")

    output_dir = str(project_output_dir)

    # Determine status based on results
    if processed_files and not failed_files:
        status = "success"
    elif processed_files and failed_files:
        status = "partial_success"
    else:
        status = "error"

    result = {
        "status": status,
        "output_directory": output_dir,
        "processed_count": len(processed_files),
        "failed_count": len(failed_files),
        "processed_files": processed_files[:10],  # Limit to first 10 for readability
        "failed_files": failed_files[:10] if failed_files else [],
        "message": (
            f"Processed {len(processed_files)} files successfully. "
            + (f"Failed: {len(failed_files)} files. " if failed_files else "")
            + f"Output saved to: {output_dir}"
        ),
    }

    return mcp.types.TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
