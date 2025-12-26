"""File upload utilities for enterprise mode.

Handles uploading files to cloud storage via presigned URLs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx
from loguru import logger


class FileUploadError(Exception):
    """Error during file upload."""

    pass


class FileUploader:
    """Handles file uploads to cloud storage via presigned URLs."""

    def __init__(
        self,
        api_client,
        on_progress: Callable[[str, int], None] | None = None,
        on_success: Callable[[str, str], None] | None = None,
        on_error: Callable[[str, str], None] | None = None,
    ):
        """
        Initialize file uploader.

        Args:
            api_client: RagopsAPIGatewayClient or similar with get_presigned_urls method
            on_progress: Callback for progress updates (file_path, percent)
            on_success: Callback for successful upload (file_path, s3_path)
            on_error: Callback for upload errors (file_path, error_message)
        """
        self.api_client = api_client
        self.on_progress = on_progress or (lambda *_: None)
        self.on_success = on_success or (lambda *_: None)
        self.on_error = on_error or (lambda *_: None)

        # Upload state
        self.upload_folder: str | None = None
        self.upload_root_path: Path | None = None
        self.uploaded_files: dict[str, str] = {}  # local_path -> s3_path

    def generate_upload_folder(self) -> str:
        """Generate a unique folder name for this upload session."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"upload-{timestamp}"

    def _determine_upload_root_path(self, file_paths: list[str]) -> Path | None:
        """Determine common root path for all files."""
        if not file_paths:
            return None

        paths = [Path(p).resolve() for p in file_paths]

        # If only one file, use its parent directory
        if len(paths) == 1:
            return paths[0].parent

        # Find common parent directory
        try:
            # Get all parent parts for each path
            all_parts = [list(p.parts) for p in paths]

            # Find common prefix
            common_parts = []
            for parts in zip(*all_parts):
                if len(set(parts)) == 1:
                    common_parts.append(parts[0])
                else:
                    break

            if common_parts:
                return Path(*common_parts)
        except Exception:
            pass

        return None

    async def upload_single_file(
        self,
        file_path_str: str,
        project_id: str,
    ) -> str | None:
        """
        Upload a single file to cloud storage.

        Args:
            file_path_str: Local path to the file
            project_id: Project ID to upload to

        Returns:
            S3 path if successful, None otherwise
        """
        file_path = Path(file_path_str)
        file_name = file_path.name

        if not file_path.exists():
            self.on_error(file_path_str, f"File not found: {file_path_str}")
            return None

        if file_path.is_dir():
            self.on_error(file_path_str, f"Cannot upload directory: {file_path_str}")
            return None

        try:
            file_size = file_path.stat().st_size

            if file_size == 0:
                self.on_error(file_path_str, f"Skipping empty file: {file_name}")
                return None

            # Initialize upload folder if not set
            if self.upload_folder is None:
                self.upload_folder = self.generate_upload_folder()

            # Progress: getting presigned URL
            self.on_progress(file_path_str, 10)

            # Calculate relative path for S3
            file_path_resolved = file_path.resolve()
            if self.upload_root_path:
                try:
                    relative_path = file_path_resolved.relative_to(self.upload_root_path)
                    s3_object_name = relative_path.as_posix()
                except ValueError:
                    s3_object_name = file_name
            else:
                s3_object_name = file_name

            # Construct folder path: projects/{project_id}/documents/{upload_folder}
            folder = f"projects/{project_id}/documents/{self.upload_folder}"

            # Get presigned URL
            async with self.api_client:
                presigned_urls_response = await self.api_client.get_presigned_urls(
                    [{"name": s3_object_name, "size": file_size}],
                    folder=folder,
                )

            if not presigned_urls_response:
                self.on_error(file_path_str, f"Failed to get presigned URL for {file_name}")
                return None

            presigned_info = presigned_urls_response[0]
            presigned_url = presigned_info["url"]
            s3_path = presigned_info["path"]

            # Progress: reading file
            self.on_progress(file_path_str, 30)

            with open(file_path, "rb") as f:
                file_data = f.read()

            if len(file_data) != file_size:
                self.on_error(
                    file_path_str,
                    f"File size mismatch: expected {file_size}, got {len(file_data)}",
                )
                return None

            # Progress: uploading
            self.on_progress(file_path_str, 50)

            # Upload to presigned URL
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    presigned_url,
                    content=file_data,
                    timeout=120.0,
                )

            if response.status_code not in (200, 204):
                self.on_error(
                    file_path_str,
                    f"Upload failed: HTTP {response.status_code}",
                )
                return None

            # Progress: completed
            self.on_progress(file_path_str, 100)
            self.uploaded_files[file_path_str] = s3_path
            self.on_success(file_path_str, s3_path)

            logger.info(f"Uploaded {file_name} to {s3_path}")
            return s3_path

        except Exception as e:
            error_msg = str(e)
            self.on_error(file_path_str, error_msg)
            logger.error(f"Error uploading {file_name}: {e}")
            return None

    async def upload_files(
        self,
        file_paths: list[str],
        project_id: str,
    ) -> dict[str, str]:
        """
        Upload multiple files sequentially.

        Args:
            file_paths: List of local file paths
            project_id: Project ID to upload to

        Returns:
            Dict mapping local paths to S3 paths for successful uploads
        """
        if not file_paths:
            return {}

        # Determine common root path
        self.upload_root_path = self._determine_upload_root_path(file_paths)
        self.upload_folder = self.generate_upload_folder()

        results: dict[str, str] = {}
        failed: set[str] = set()

        for file_path_str in file_paths:
            if file_path_str in failed:
                continue

            s3_path = await self.upload_single_file(file_path_str, project_id)
            if s3_path:
                results[file_path_str] = s3_path
            else:
                failed.add(file_path_str)

        return results

    def get_s3_paths(self) -> list[str]:
        """Get list of S3 paths for all uploaded files."""
        return list(self.uploaded_files.values())

    def reset(self) -> None:
        """Reset upload state for a new session."""
        self.upload_folder = None
        self.upload_root_path = None
        self.uploaded_files.clear()


async def upload_files_to_cloud(
    api_client,
    file_paths: list[str],
    project_id: str,
    on_progress: Callable[[str, int], None] | None = None,
) -> list[str]:
    """
    Convenience function to upload files and return S3 paths.

    Args:
        api_client: API client with get_presigned_urls method
        file_paths: List of local file paths
        project_id: Project ID

    Returns:
        List of S3 paths for successfully uploaded files
    """
    uploader = FileUploader(api_client, on_progress=on_progress)
    await uploader.upload_files(file_paths, project_id)
    return uploader.get_s3_paths()
