"""Async client for ragops-api-gateway."""

import json
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import UUID

import aiohttp
from websockets.asyncio.client import connect

from .errors import (
    RagopsAPIGatewayConnectionError,
    RagopsAPIGatewayError,
    RagopsAPIGatewayMaxAttemptsExceededError,
    RagopsAPIGatewayResponseError,
)
from .schemas import ProjectInfo
from .ws_handler import WsHandler


class RagopsAPIGatewayClient:
    """Async client for ragops-api-gateway."""

    def __init__(
        self,
        base_url: str,
        ws_handler: WsHandler | None = None,
        api_token: str | None = None,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self.ws_handler = ws_handler

    def set_token(self, api_token: str):
        self.api_token = api_token

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            headers={"X-API-Token": self.api_token or ""},
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    def _ensure_session(self):
        if self._session is None:
            raise RagopsAPIGatewayError(
                "Client session is not initialized. Use 'async with RagopsAPIGatewayClient(...)'."
            )

    async def _handle_error_response(self, response: aiohttp.ClientResponse):
        try:
            error_data = await response.json()
            detail = error_data.get("detail", "Unknown error")
        except Exception:
            detail = await response.text()
        raise RagopsAPIGatewayResponseError(response.status, detail)

    async def get_history(self, project_id: UUID) -> List[Dict[str, Any]]:
        """
        Get chat history for a specific project.
        Returns: List of message dicts for the project.
        """
        self._ensure_session()
        params = {"project_id": str(project_id)}
        async with self._session.get(f"{self.base_url}/history", params=params) as resp:
            if resp.status != 200:
                await self._handle_error_response(resp)
            return await resp.json()

    async def chat_stream(
        self, message: str, project_id: UUID, message_history: list[dict]
    ) -> AsyncIterator[str]:
        """
        Stream chat responses from the ragops gateway.
        Args:
            message: User message to send.
            project_id: Project UUID.
            message_history: List of previous messages in the chat.
        Yields:
            Text responses from the agent as they arrive.
        """
        self._ensure_session()
        payload = {
            "message": message,
            "project_id": str(project_id),
            "message_history": message_history,
        }
        async with self._session.post(
            f"{self.base_url}/chat/stream",
            json=payload,
        ) as resp:
            if resp.status != 200:
                await self._handle_error_response(resp)
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        if data.get("error"):
                            raise RagopsAPIGatewayError(data["error"])
                        yield data.get("text", "")
                    except json.JSONDecodeError:
                        continue

    async def get_balance(self) -> int:
        self._ensure_session()
        async with self._session.get(f"{self.base_url}/credits/balance") as resp:
            if resp.status != 200:
                await self._handle_error_response(resp)
            data = await resp.json()
            return data.get("balance", 0)

    async def create_project(self) -> ProjectInfo:
        """
        Create a new project.
        Returns: Project details.
        """
        self._ensure_session()
        async with self._session.post(f"{self.base_url}/projects") as resp:
            if resp.status != 200:
                await self._handle_error_response(resp)
            data = await resp.json()
            return ProjectInfo.model_validate(data)

    async def get_project(self, project_id: UUID) -> ProjectInfo:
        """
        Get project details.
        Returns: Project details.
        """
        self._ensure_session()
        async with self._session.get(f"{self.base_url}/projects/{project_id}") as resp:
            if resp.status != 200:
                await self._handle_error_response(resp)
            data = await resp.json()
            return ProjectInfo.model_validate(data)

    async def poll_experiment_updates(self, project_id: UUID, updated_after: datetime):
        self._ensure_session()
        params = {
            "project_id": str(project_id),
            "updated_after": updated_after.isoformat(),
        }
        async with self._session.get(
            f"{self.base_url}/experiment-updates", params=params
        ) as resp:
            if resp.status != 200:
                await self._handle_error_response(resp)
            data = await resp.json()
            return data

    def register_ws_handler(self, ws_handler: WsHandler):
        self.ws_handler = ws_handler

    async def connect_ws(self, project_id: UUID):
        if not self.ws_handler:
            return

        url = f"{self.base_url.replace('http', 'ws')}/agent/ws?project_id={project_id}"
        connection_attempts = 0
        try:
            async with connect(
                url, additional_headers={"X-API-Token": self.api_token}
            ) as ws:
                connection_attempts = 0
                await self.ws_handler.handle_ws_connection(ws)
        except Exception as e:
            connection_attempts += 1
            if connection_attempts > 3:
                raise RagopsAPIGatewayMaxAttemptsExceededError(max_attempts=3) from e
            raise RagopsAPIGatewayConnectionError() from e

    async def generate(
        self,
        messages: list[dict],
        provider: str = "default",
        model_name: str = "default",
        temperature: float = 1.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        response_format: str | None = None,
        project_id: str | None = None,
    ) -> dict:
        """
        Проксирует генерацию текста через LLM Gate.
        """
        self._ensure_session()
        payload = {
            "provider": provider,
            "model_name": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop,
            "tools": tools,
            "tool_choice": tool_choice,
            "response_format": response_format,
            "project_id": project_id,
        }
        async with self._session.post(
            f"{self.base_url}/v1/generate", json=payload
        ) as resp:
            if resp.status != 200:
                await self._handle_error_response(resp)
            return await resp.json()

    async def generate_stream(
        self,
        provider: str,
        model_name: str,
        messages: list[dict],
        temperature: float = 1.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        response_format: str | None = None,
        project_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """
        Проксирует стриминговую генерацию текста через LLM Gate.
        """
        self._ensure_session()
        payload = {
            "provider": provider,
            "model_name": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop,
            "tools": tools,
            "tool_choice": tool_choice,
            "response_format": response_format,
            "project_id": project_id,
        }
        async with self._session.post(
            f"{self.base_url}/v1/generate/stream", json=payload
        ) as resp:
            if resp.status != 200:
                await self._handle_error_response(resp)
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        yield data
                    except json.JSONDecodeError:
                        continue

    async def embeddings(
        self,
        input_text: str,
        provider: str = "default",
        model_name: str | None = None,
        credentials: dict | None = None,
        dimensions: int | None = None,
        project_id: str | None = None,
    ) -> dict:
        """
        Проксирует генерацию эмбеддингов через LLM Gate.
        """
        self._ensure_session()
        payload = {
            "provider": provider,
            "input": input_text,
            "model_name": model_name,
            "credentials": credentials,
            "dimensions": dimensions,
            "project_id": project_id,
        }
        async with self._session.post(
            f"{self.base_url}/v1/embeddings", json=payload
        ) as resp:
            if resp.status != 200:
                await self._handle_error_response(resp)
            return await resp.json()
