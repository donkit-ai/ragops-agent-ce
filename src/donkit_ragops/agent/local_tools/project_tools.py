from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal

from donkit_ragops.db import close
from donkit_ragops.db import kv_all_by_prefix
from donkit_ragops.db import kv_delete
from donkit_ragops.db import kv_get
from donkit_ragops.db import kv_set
from donkit_ragops.db import open_db
from donkit_ragops.schemas.config_schemas import RagConfig
from .tools import AgentTool


def _project_key(project_id: str) -> str:
    return f"project_{project_id}"


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> None:
    """Recursively update base dict with values from update dict.

    Examples:
        base = {"a": {"b": 1}, "c": 2}
        update = {"a": {"b": 3}}
        Result: {"a": {"b": 3}, "c": 2}  # Only "b" updated, "c" preserved

        base = {"embedder": {"embedder_type": "vertex"}}
        update = {"embedder": {"embedder_type": "openai"}}
        Result: {"embedder": {"embedder_type": "openai"}}  # embedder_type updated
    """
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            _deep_update(base[key], value)
        else:
            # Replace or add the value
            base[key] = value


def tool_create_project() -> AgentTool:
    def handler(payload: dict[str, Any]) -> str:
        project_id = payload.get("project_id") or uuid.uuid4().hex
        checklist = payload.get("checklist")
        if not project_id:
            return "Error: project_id is required."
        if not checklist:
            return "Error: checklist is required."
        db = open_db()
        try:
            key = _project_key(project_id)
            if kv_get(db, key) is not None:
                return f"Error: Project '{project_id}' already exists."

            project_state = {
                "project_id": project_id,
                "checklist": checklist,
                "status": "new",
                "configuration": None,
                "chunks_path": None,
                "collection_name": None,
                "loaded_files": [],  # List of files loaded into vectorstore with metadata
            }
            kv_set(db, key, json.dumps(project_state))
            return f"Successfully created project '{project_id}'."
        finally:
            close(db)

    return AgentTool(
        name="create_project",
        description=(
            "Creates a new RAG project with a given ID"
            "initializing its state in the database."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "A unique identifier for the project. "
                    "If not provided, a random ID will be generated.",
                },
                "checklist": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of tasks to complete the project.",
                },
            },
            "required": ["checklist"],
        },
        handler=handler,
    )


def tool_get_project() -> AgentTool:
    def handler(payload: dict[str, Any]) -> str:
        project_id = payload.get("project_id")
        if not project_id:
            return "Error: project_id is required."

        db = open_db()
        try:
            key = _project_key(project_id)
            state_raw = kv_get(db, key)
            if state_raw is None:
                return f"Error: Project '{project_id}' not found."

            return state_raw  # Return the raw JSON string
        finally:
            close(db)

    return AgentTool(
        name="get_project",
        description=(
            "Retrieves the current state of a RAG project from the database as a JSON string."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "The ID of the project to retrieve.",
                },
            },
            "required": ["project_id"],
        },
        handler=handler,
    )


def tool_list_projects() -> AgentTool:
    def handler(payload: dict[str, Any]) -> str:
        db = open_db()
        try:
            all_projects_raw = kv_all_by_prefix(db, "project_")
            # Parse JSON and collect into a list of project states
            projects = [json.loads(value) for _, value in all_projects_raw]
            return json.dumps(projects, indent=2)
        finally:
            close(db)

    return AgentTool(
        name="list_projects",
        description="Lists all RAG projects currently stored in the database.",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


def tool_save_rag_config() -> AgentTool:
    def handler(payload: dict[str, Any]) -> str:
        project_id = payload.get("project_id")
        rag_config_update = payload.get("rag_config")

        # Validate required parameters FIRST
        if not project_id:
            return "Error: project_id is required."
        if rag_config_update is None:
            return (
                "Error: rag_config is required. "
                "You can pass either: (1) the COMPLETE JSON object from rag_config_plan, "
                "or (2) partial updates like {'embedder': {'embedder_type': 'openai'}}. "
                "If passing partial updates, existing config will be merged."
            )

        # Validate that rag_config is a dict/object
        if not isinstance(rag_config_update, dict):
            return (
                f"Error: rag_config must be a dictionary/object, but got {type(rag_config_update).__name__}. "
                "Pass either full config or partial updates."
            )

        db = open_db()
        try:
            key = _project_key(project_id)
            current_state_raw = kv_get(db, key)
            if current_state_raw is None:
                return f"Error: Project '{project_id}' not found."

            current_state = json.loads(current_state_raw)
            existing_config = current_state.get("configuration")

            # Check if this looks like a complete config (has all required fields)
            required_fields = {
                "files_path",
                "generation_model_type",
                "generation_model_name",
                "database_uri",
            }
            has_all_required = all(field in rag_config_update for field in required_fields)

            # If no existing config, use update as-is (must be complete)
            if existing_config is None:
                try:
                    # Validate as complete config
                    RagConfig.model_validate(rag_config_update, extra="forbid")
                    current_state["configuration"] = rag_config_update
                except ValueError as e:
                    return (
                        f"Error: No existing config found. You must provide a COMPLETE config. {e}\n\n"
                        "Required fields: files_path, generation_model_type, generation_model_name, database_uri."
                    )
            elif has_all_required:
                # Looks like a complete config - replace entirely (agent might want to reset)
                try:
                    RagConfig.model_validate(rag_config_update, extra="forbid")
                    current_state["configuration"] = rag_config_update
                except ValueError as e:
                    return (
                        f"Error: Invalid complete config. {e}\n\n"
                        "Make sure all required fields are present and valid."
                    )
            else:
                # Partial update - merge with existing config
                merged_config = json.loads(json.dumps(existing_config))  # Deep copy via JSON
                _deep_update(merged_config, rag_config_update)

                # Validate merged config
                try:
                    RagConfig.model_validate(merged_config, extra="forbid")
                    current_state["configuration"] = merged_config
                except ValueError as e:
                    return (
                        f"Error: Invalid config after merge. {e}\n\n"
                        "Make sure partial updates are valid and don't conflict with existing values. "
                        "Use get_rag_config to see current configuration."
                    )

            kv_set(db, key, json.dumps(current_state))
            return f"Successfully saved RAG configuration for project '{project_id}'."
        finally:
            close(db)

    return AgentTool(
        name="save_rag_config",
        description=(
            "Saves or updates RAG configuration for the project. "
            "You can pass either (1) a COMPLETE config object from rag_config_plan, "
            "or (2) PARTIAL updates like {'embedder': {'embedder_type': 'openai'}}. "
            "Partial updates will be merged with existing config. "
            "If no existing config exists, you MUST provide a complete config."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "The ID of the project to save configuration for.",
                },
                "rag_config": {
                    "type": "object",
                    "description": (
                        "Either the COMPLETE RAG configuration object from rag_config_plan tool, "
                        "OR partial updates to merge with existing config. "
                        "Examples: Full config from rag_config_plan, or partial: {'embedder': {'embedder_type': 'openai'}}, "
                        "{'db_type': 'qdrant'}, {'chunking_options': {'chunk_size': 500}}. "
                        "If project has no existing config, you must provide complete config with all required fields."
                    ),
                },
            },
            "required": ["project_id", "rag_config"],
        },
        handler=handler,
    )


def tool_get_rag_config() -> AgentTool:
    def handler(payload: dict[str, Any]) -> str:
        project_id = payload.get("project_id")
        if not project_id:
            return "Error: project_id is required."

        db = open_db()
        try:
            key = _project_key(project_id)
            state_raw = kv_get(db, key)
            if state_raw is None:
                return f"Error: Project '{project_id}' not found."

            state = json.loads(state_raw)
            config = state.get("configuration")

            if config is None:
                return f"No RAG configuration found for project '{project_id}'."

            return json.dumps(config, indent=2)
        finally:
            close(db)

    return AgentTool(
        name="get_rag_config",
        description="Retrieves the saved RAG configuration for a project as JSON.",
        parameters={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "The ID of the project to get configuration from.",
                },
            },
            "required": ["project_id"],
        },
        handler=handler,
    )


def tool_add_loaded_files() -> AgentTool:
    def handler(payload: dict[str, Any]) -> str:
        project_id = payload.get("project_id")
        files = payload.get("files")  # List of file paths or dicts with metadata

        if not project_id:
            return "Error: project_id is required."
        if not files:
            return "Error: files list is required."

        db = open_db()
        try:
            key = _project_key(project_id)
            state_raw = kv_get(db, key)
            if state_raw is None:
                return f"Error: Project '{project_id}' not found."

            state = json.loads(state_raw)
            loaded_files = state.get("loaded_files", [])

            # Add new files
            added_count = 0
            for file_item in files:
                # Ensure file_item is a dict with at least 'path'
                if isinstance(file_item, str):
                    file_info = {"path": file_item, "status": "loaded"}
                else:
                    file_info = dict(file_item)
                    # Set default status if not provided
                    if "status" not in file_info:
                        file_info["status"] = "loaded"

                file_path = file_info.get("path")
                if not file_path:
                    continue

                # Check if already exists
                if not any(f.get("path") == file_path for f in loaded_files):
                    loaded_files.append(file_info)
                    added_count += 1

            state["loaded_files"] = loaded_files
            kv_set(db, key, json.dumps(state))

            return f"Added {added_count} file(s) to loaded files for project '{project_id}'."
        finally:
            close(db)

    return AgentTool(
        name="add_loaded_files",
        description=(
            "Add files to the list of loaded files for a project. "
            "This tracks which files have been loaded into vectorstore. "
            "Call this AFTER successfully loading chunks into vector database. "
            "IMPORTANT: Pass specific file paths (e.g., ['file1.json', 'file2.json']), "
            "NOT directory paths. Get file list from chunked directory first."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "The ID of the project.",
                },
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path (required)",
                            },
                            "status": {
                                "type": "string",
                                "description": "Status (optional, default: 'loaded')",
                            },
                            "chunks_count": {
                                "type": "number",
                                "description": "Number of chunks (optional)",
                            },
                        },
                        "required": ["path"],
                    },
                    "description": "List of file metadata objects with at least 'path' field.",
                },
            },
            "required": ["project_id", "files"],
        },
        handler=handler,
    )


def tool_list_loaded_files() -> AgentTool:
    def handler(payload: dict[str, Any]) -> str:
        project_id = payload.get("project_id")
        if not project_id:
            return "Error: project_id is required."

        db = open_db()
        try:
            key = _project_key(project_id)
            state_raw = kv_get(db, key)
            if state_raw is None:
                return f"Error: Project '{project_id}' not found."

            state = json.loads(state_raw)
            loaded_files = state.get("loaded_files", [])

            return json.dumps({"loaded_files": loaded_files}, indent=2)
        finally:
            close(db)

    return AgentTool(
        name="list_loaded_files",
        description=(
            "Get the list of files loaded into vectorstore for a project. "
            "Use this to check which files are already in the RAG system "
            "before loading new files incrementally."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "The ID of the project.",
                },
            },
            "required": ["project_id"],
        },
        handler=handler,
    )


def tool_delete_project() -> AgentTool:
    def handler(payload: dict[str, Any]) -> str:
        project_id = payload.get("project_id")
        if not project_id:
            return "Error: project_id is required."

        db = open_db()
        try:
            key = _project_key(project_id)
            state_raw = kv_get(db, key)
            if state_raw is None:
                return f"Error: Project '{project_id}' not found."

            # Delete from database
            deleted = kv_delete(db, key)
            if not deleted:
                return f"Error: Failed to delete project '{project_id}' from database."

            # Delete project directory
            project_dir = Path(f"projects/{project_id}").resolve()
            if project_dir.exists():
                try:
                    shutil.rmtree(project_dir)
                except Exception as e:
                    return f"Warning: Deleted from DB but failed to delete directory: {e}"

            # Delete checklist from database (if exists)
            checklist_key = _checklist_key(f"checklist_{project_id}")
            try:
                kv_delete(db, checklist_key)
            except Exception:
                pass  # Optional cleanup - checklist might not exist

            return f"Successfully deleted project '{project_id}' and all related data."
        finally:
            close(db)

    return AgentTool(
        name="delete_project",
        description=(
            "Deletes a RAG project and all its related files "
            "(database entry, project directory, checklist). "
            "WARNING: This operation cannot be undone!"
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "The ID of the project to delete.",
                },
            },
            "required": ["project_id"],
        },
        handler=handler,
    )
