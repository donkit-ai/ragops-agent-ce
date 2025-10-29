from __future__ import annotations

import json
import uuid
from typing import Any

from ..db import close, kv_all_by_prefix, kv_get, kv_set, open_db
from ..schemas.config_schemas import RagConfig
from .tools import AgentTool


def _project_key(project_id: str) -> str:
    return f"project_{project_id}"


def tool_create_project() -> AgentTool:
    def handler(payload: dict[str, Any]) -> str:
        project_id = payload.get("project_id") or uuid.uuid4().hex
        goal = payload.get("goal")
        checklist = payload.get("checklist")
        if not project_id or not goal:
            return "Error: project_id and goal are required."
        if not checklist:
            return "Error: checklist is required."
        db = open_db()
        try:
            key = _project_key(project_id)
            if kv_get(db, key) is not None:
                return f"Error: Project '{project_id}' already exists."

            project_state = {
                "project_id": project_id,
                "goal": goal,
                "checklist": checklist,
                "status": "new",
                "configuration": None,
                "chunks_path": None,
                "collection_name": None,
            }
            kv_set(db, key, json.dumps(project_state))
            return f"Successfully created project '{project_id}'."
        finally:
            close(db)

    return AgentTool(
        name="create_project",
        description=(
            "Creates a new RAG project with a given ID and goal, "
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
                "goal": {
                    "type": "string",
                    "description": "The main objective of the RAG pipeline.",
                },
                "checklist": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of tasks to complete the project.",
                },
            },
            "required": ["goal", "checklist"],
        },
        handler=handler,
    )


#
# def tool_update_project() -> AgentTool:
#     def handler(payload: dict[str, Any]) -> str:
#         project_id = payload.get("project_id")
#         update_data = payload.get("update_data")
#         if not project_id or not update_data:
#             return "Error: project_id and update_data are required."
#
#         db = open_db()
#         try:
#             key = _project_key(project_id)
#             current_state_raw = kv_get(db, key)
#             if current_state_raw is None:
#                 return f"Error: Project '{project_id}' not found."
#
#             current_state = json.loads(current_state_raw)
#             current_state.update(update_data)
#             kv_set(db, key, json.dumps(current_state))
#             return f"Successfully updated project '{project_id}'."
#         finally:
#             close(db)
#
#     return AgentTool(
#         name="update_project",
#         description=(
#             "Updates the state of an existing RAG project with new data "
#             "(e.g., adding a plan, chunks_path, or status)."
#         ),
#         parameters={
#             "type": "object",
#             "properties": {
#                 "project_id": {
#                     "type": "string",
#                     "description": "The ID of the project to update.",
#                 },
#                 "update_data": {
#                     "type": "object",
#                     "description": "A dictionary of fields to update in the project's state.",
#                 },
#             },
#             "required": ["project_id", "update_data"],
#         },
#         handler=handler,
#     )


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
        rag_config = payload.get("rag_config")
        try:
            RagConfig.model_validate(rag_config, extra="forbid")
        except ValueError as e:
            return f"Error: Invalid rag_config. {e}"
        if not project_id:
            return "Error: project_id is required."
        if not rag_config:
            return "Error: rag_config is required."

        db = open_db()
        try:
            key = _project_key(project_id)
            current_state_raw = kv_get(db, key)
            if current_state_raw is None:
                return f"Error: Project '{project_id}' not found."

            current_state = json.loads(current_state_raw)
            current_state["configuration"] = rag_config
            kv_set(db, key, json.dumps(current_state))

            return f"Successfully saved RAG configuration for project '{project_id}'."
        finally:
            close(db)

    return AgentTool(
        name="save_rag_config",
        description=(
            "Saves RAG configuration (from planner) to the project. "
            "This configuration includes embedder type, chunking settings, retrieval options, etc."
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
                        "The RAG configuration object from "
                        "rag_config_plan tool (as a dict/JSON object)."
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
