from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import json
import os
import tempfile
from collections.abc import Iterator

# Google Gen AI SDK
from google import genai
from google.genai import types
from google.oauth2 import service_account
from loguru import logger

from ...config import Settings
from ...config import load_settings
from ...logging_config import setup_logging
from ..base import LLMProvider
from ..types import LLMResponse
from ..types import Message
from ..types import ToolCall
from ..types import ToolFunctionCall
from ..types import ToolSpec


class VertexProvider(LLMProvider):
    name: str = "vertex"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        credentials_data: dict[str, str] | None = None,
        model_name: str | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        # Ensure logging is configured according to .env / settings
        try:
            setup_logging(self.settings)
        except Exception:
            pass

        if not credentials_data:
            raise ValueError("Vertex credentials are not set")

        service_account.Credentials.from_service_account_info(
            credentials_data, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        # Set up environment for Vertex AI with cross-platform temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            json.dump(credentials_data, temp_file)
            temp_file_path = temp_file.name

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path

        # Create client for Vertex AI
        self._client = genai.Client(
            # TODO: move proj and loc to settings before release
            vertexai=True,
            project=credentials_data.get("project_id"),
            location="us-central1",
        )

        self._model_name = model_name or self.settings.llm_model or "gemini-2.5-flash"

    @staticmethod
    def _clean_json_schema(schema: dict | None) -> types.Schema:
        """
        Transform an arbitrary JSON Schema-like dict (possibly produced by Pydantic)
        into a google.genai.types.Schema instance by:
        - Inlining $ref by replacing references with actual schemas from $defs
        - Removing $defs after inlining all references
        - Renaming unsupported keys to the SDK's expected snake_case
        - Recursively converting nested schemas (properties, items, anyOf)
        - Preserving fields supported by the SDK Schema model
        """
        if not isinstance(schema, dict):
            # Fallback to an object schema when input is not a dict
            return types.Schema()

        # Step 1: Inline $ref references before any conversion
        defs = schema.get("$defs", {})

        def inline_refs(obj, definitions):
            """Recursively inline $ref references."""
            if isinstance(obj, dict):
                # If this object has a $ref, replace it with the referenced schema
                if "$ref" in obj:
                    ref_path = obj["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        ref_name = ref_path.replace("#/$defs/", "")
                        if ref_name in definitions:
                            # Get the referenced schema and inline it recursively
                            referenced = definitions[ref_name].copy()
                            # Preserve description and default from the referencing object
                            if "description" in obj and "description" not in referenced:
                                referenced["description"] = obj["description"]
                            if "default" in obj:
                                referenced["default"] = obj["default"]
                            return inline_refs(referenced, definitions)
                    # If can't resolve, remove the $ref
                    return {k: v for k, v in obj.items() if k != "$ref"}

                # Recursively process all properties
                result = {}
                for key, value in obj.items():
                    if key == "$defs":
                        continue  # Remove $defs after inlining
                    # Skip additionalProperties: true as it's not well supported
                    if key == "additionalProperties" and value is True:
                        continue
                    result[key] = inline_refs(value, definitions)
                return result
            elif isinstance(obj, list):
                return [inline_refs(item, definitions) for item in obj]
            else:
                return obj

        # Inline all references
        schema = inline_refs(schema, defs)

        # Debug: Log the inlined schema

        # Step 2: Convert to SDK schema format
        # Mapping from common JSON Schema/OpenAPI keys to google-genai Schema fields
        key_map = {
            "anyOf": "any_of",
            "additionalProperties": "additional_properties",
            "maxItems": "max_items",
            "maxLength": "max_length",
            "maxProperties": "max_properties",
            "minItems": "min_items",
            "minLength": "min_length",
            "minProperties": "min_properties",
            "propertyOrdering": "property_ordering",
        }

        def convert(obj):
            if isinstance(obj, dict):
                out: dict[str, object] = {}
                for k, v in obj.items():
                    if k == "const":
                        out["enum"] = [v]
                        continue

                    kk = key_map.get(k, k)
                    if kk == "properties" and isinstance(v, dict):
                        # properties: dict[str, Schema]
                        out[kk] = {pk: convert(pv) for pk, pv in v.items()}
                    elif kk == "items":
                        # items: Schema (treat list as first item schema)
                        if isinstance(v, list) and v:
                            out[kk] = convert(v[0])
                        else:
                            out[kk] = convert(v)
                    elif kk == "any_of" and isinstance(v, list):
                        out[kk] = [convert(iv) for iv in v]
                    else:
                        out[kk] = convert(v)
                return out
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            else:
                return obj

        converted = convert(schema)

        # Debug: Log the converted schema

        try:
            result = types.Schema(**converted)  # type: ignore[arg-type]
            return result
        except Exception as e:
            logger.error(
                f"Failed to construct types.Schema from converted schema: {e}. "
                f"Schema was: {json.dumps(converted, default=str, indent=2)}"
            )
            return types.Schema()

    def generate(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        def _safe_text(text: str) -> str:
            try:
                return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            except Exception:
                return ""

        contents: list[types.Content] = []
        system_instruction = ""

        # Group consecutive tool messages into single Content
        i = 0
        while i < len(messages):
            m = messages[i]

            if m.role == "tool":
                # Collect all consecutive tool messages
                tool_parts = []
                while i < len(messages) and messages[i].role == "tool":
                    tool_msg = messages[i]
                    part = types.Part.from_function_response(
                        name=tool_msg.name or "",
                        response={"result": _safe_text(tool_msg.content or "")},
                    )
                    tool_parts.append(part)
                    i += 1
                # Add all tool responses as a single Content
                if tool_parts:
                    contents.append(types.Content(role="tool", parts=tool_parts))
                continue
            elif m.role == "system":
                system_instruction += _safe_text(m.content).strip()
                i += 1
            elif m.role == "assistant":
                if m.tool_calls:
                    # Assistant message with tool calls
                    parts_list = []
                    for tc in m.tool_calls:
                        if not tc.function.name:
                            logger.warning("Skipping tool call without name in history")
                            continue
                        args = (
                            tc.function.arguments if isinstance(tc.function.arguments, dict) else {}
                        )
                        part = types.Part.from_function_call(name=tc.function.name, args=args)
                        parts_list.append(part)
                    if parts_list:
                        contents.append(types.Content(role="assistant", parts=parts_list))
                elif m.content:
                    # Regular assistant text response
                    part = types.Part(text=_safe_text(m.content))
                    contents.append(types.Content(role="assistant", parts=[part]))
                # Skip empty assistant messages without tool_calls or content
                i += 1
            else:
                part = types.Part(text=_safe_text(m.content))
                contents.append(types.Content(role=m.role, parts=[part]))
                i += 1
        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else 0.2,
            top_p=top_p if top_p is not None else 0.95,
            max_output_tokens=max_tokens if max_tokens is not None else 8192,
            system_instruction=system_instruction if system_instruction else None,
        )
        if tools:
            function_declarations: list[types.FunctionDeclaration] = []
            for t in tools:
                schema_obj = self._clean_json_schema(t.function.parameters or {})
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=t.function.name,
                        description=t.function.description or "",
                        parameters=schema_obj,
                    )
                )
            gen_tools = [types.Tool(function_declarations=function_declarations)]
            config.tools = gen_tools

        model_name = model or self._model_name

        try:
            response = self._client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            text, tool_calls = self._parse_response(response)
            raw = response.model_dump()
            # If no text and no tool calls, check for errors in response
            if not text and not tool_calls:
                try:
                    # Check for blocking reasons or errors
                    if hasattr(response, "candidates") and response.candidates:
                        cand = response.candidates[0]
                        if hasattr(cand, "finish_reason") and cand.finish_reason:
                            finish_reason = cand.finish_reason
                            if finish_reason not in ("STOP", None):
                                error_msg = f"Model finished with reason: {finish_reason}"
                                logger.warning(error_msg)
                                return LLMResponse(content=f"Warning: {error_msg}")
                    # Check for safety ratings that might block content
                    if hasattr(response, "candidates") and response.candidates:
                        cand = response.candidates[0]
                        if hasattr(cand, "safety_ratings"):
                            blocked = any(
                                getattr(r, "blocked", False)
                                for r in getattr(cand, "safety_ratings", [])
                            )
                            if blocked:
                                error_msg = "Response was blocked by safety filters"
                                logger.warning(error_msg)
                                return LLMResponse(content=f"Warning: {error_msg}")
                except Exception:
                    pass  # If we can't check, just return empty
            return LLMResponse(content=text, tool_calls=tool_calls, raw=raw)
        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Error generating content with model '{model_name}': {error_msg}", exc_info=True
            )
            # Return error message instead of empty response so user can see what went wrong
            return LLMResponse(content=f"Error: {error_msg}")

    @staticmethod
    def _parse_response(response) -> tuple[str | None, list[ToolCall] | None]:
        """Parse a genai response (or stream chunk) into plain text and tool calls."""
        calls: list[ToolCall] = []

        try:
            cand_list = response.candidates
        except AttributeError:
            cand_list = None
        if not cand_list:
            return None, None

        cand = cand_list[0]

        # Log finish reason if response was cut off
        try:
            finish_reason = cand.finish_reason
            if finish_reason and finish_reason != "STOP":
                logger.warning(f"Response finished with reason: {finish_reason}")
        except AttributeError:
            pass

        try:
            parts = cand.content.parts or []
        except AttributeError:
            parts = []

        # Collect text and tool calls in a single pass
        collected_text: list[str] = []
        for p in parts:
            # Try to get text from this part
            try:
                t = p.text
                if t:
                    collected_text.append(t)
            except AttributeError:
                pass

            # Try to get function_call from this part
            try:
                fc = p.function_call
                if fc:
                    # Extract function name and arguments
                    try:
                        name = fc.name
                    except AttributeError:
                        name = ""

                    if not name:
                        logger.warning(f"Skipping function call without name: {fc}")
                        continue

                    try:
                        args = dict(fc.args) if fc.args else {}
                    except (AttributeError, TypeError):
                        args = {}

                    calls.append(
                        ToolCall(
                            id=name,
                            function=ToolFunctionCall(
                                name=name,
                                arguments=args,
                            ),
                        )
                    )
            except AttributeError:
                pass

        text = "".join(collected_text)
        return text or None, calls or None

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        """Get list of available models from Vertex AI."""
        try:
            # Try to list models from the client
            # Note: Google Gen AI SDK may not have a direct list_models method
            # We'll try to access it if available
            if hasattr(self._client, "models") and hasattr(self._client.models, "list"):
                models = self._client.models.list()
                return [
                    model.name.split("/")[-1] if "/" in model.name else model.name
                    for model in models
                ]
            # Fallback: try to get models from the client's models attribute
            if hasattr(self._client, "models") and hasattr(self._client.models, "list_models"):
                models = self._client.models.list_models()
                return [
                    model.name.split("/")[-1] if "/" in model.name else model.name
                    for model in models
                ]
        except Exception:
            pass
        # Fallback to common Vertex AI models
        return [
            "gemini-2.5-flash",
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-pro",
            "gemini-pro-vision",
        ]

    def list_chat_models(self) -> list[str]:
        """Get list of chat models with tool calling support (Gemini models)."""
        # Vertex AI Gemini models support tool calling
        all_models = self.list_models()
        # Filter to Gemini models (exclude embedding models if any)
        chat_models = [
            m for m in all_models if "gemini" in m.lower() and "embedding" not in m.lower()
        ]
        if not chat_models:
            # Fallback to common Gemini models
            return [
                "gemini-2.5-flash",
                "gemini-2.0-flash-exp",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-pro",
            ]
        return chat_models

    def list_embedding_models(self) -> list[str]:
        """Get list of embedding models."""
        # Vertex AI uses default embedding model, but we can list if available
        all_models = self.list_models()
        embedding_models = [m for m in all_models if "embedding" in m.lower()]
        # Vertex typically uses default embedding, so return empty if none found
        # The system will use the default embedding model
        return embedding_models

    def generate_stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[LLMResponse]:
        """Stream generation yielding partial responses."""

        def _safe_text(text: str) -> str:
            try:
                return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            except Exception:
                return ""

        contents: list[types.Content] = []
        system_instruction = ""

        # Convert messages to genai format (same logic as generate())
        i = 0
        while i < len(messages):
            m = messages[i]

            if m.role == "tool":
                # Collect all consecutive tool messages
                tool_parts = []
                while i < len(messages) and messages[i].role == "tool":
                    tool_msg = messages[i]
                    part = types.Part.from_function_response(
                        name=tool_msg.name or "",
                        response={"result": _safe_text(tool_msg.content or "")},
                    )
                    tool_parts.append(part)
                    i += 1
                if tool_parts:
                    contents.append(types.Content(role="tool", parts=tool_parts))
                continue
            elif m.role == "system":
                system_instruction += _safe_text(m.content).strip()
                i += 1
            elif m.role == "assistant":
                if m.tool_calls:
                    parts_list = []
                    for tc in m.tool_calls:
                        if not tc.function.name:
                            logger.warning("Skipping tool call without name in history")
                            continue
                        args = (
                            tc.function.arguments if isinstance(tc.function.arguments, dict) else {}
                        )
                        part = types.Part.from_function_call(name=tc.function.name, args=args)
                        parts_list.append(part)
                    if parts_list:
                        contents.append(types.Content(role="assistant", parts=parts_list))
                elif m.content:
                    part = types.Part(text=_safe_text(m.content))
                    contents.append(types.Content(role="assistant", parts=[part]))
                i += 1
            else:
                part = types.Part(text=_safe_text(m.content))
                contents.append(types.Content(role=m.role, parts=[part]))
                i += 1

        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else 0.2,
            top_p=top_p if top_p is not None else 0.95,
            max_output_tokens=max_tokens if max_tokens is not None else 8192,
            system_instruction=system_instruction if system_instruction else None,
        )

        if tools:
            function_declarations: list[types.FunctionDeclaration] = []
            for t in tools:
                schema_obj = self._clean_json_schema(t.function.parameters or {})
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=t.function.name,
                        description=t.function.description or "",
                        parameters=schema_obj,
                    )
                )
            gen_tools = [types.Tool(function_declarations=function_declarations)]
            config.tools = gen_tools

        model_name = model or self._model_name

        try:
            # Use generate_content_stream for streaming
            stream = self._client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )

            for chunk in stream:
                text, tool_calls = self._parse_response(chunk)

                # Yield text chunks as they come
                if text:
                    yield LLMResponse(content=text, tool_calls=None, raw=chunk.model_dump())

                # Tool calls come in final chunk - yield them separately
                if tool_calls:
                    yield LLMResponse(
                        content=None,
                        tool_calls=tool_calls,
                        raw=chunk.model_dump(),
                    )

        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Error streaming content with model '{model_name}': {error_msg}", exc_info=True
            )
            # Yield error message instead of empty response
            yield LLMResponse(content=f"Error: {error_msg}")
