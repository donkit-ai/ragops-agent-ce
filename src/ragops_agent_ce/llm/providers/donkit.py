from __future__ import annotations

from ..base import LLMProvider
from ..types import LLMResponse
from ..types import Message
from ..types import ToolSpec
from ...config import Settings
from ...config import load_settings

from donkit.ragops_api_gateway_client.client import RagopsAPIGatewayClient


class DonkitProvider(LLMProvider):
    name: str = "donkit"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        if not self.settings.donkit_api_key:
            raise ValueError("DONKIT_API_KEY is not set")
        if not self.settings.donkit_base_url:
            raise ValueError("DONKIT_BASE_URL is not set")
        self._client = RagopsAPIGatewayClient(
            base_url=self.settings.donkit_base_url,
            api_token=self.settings.donkit_api_key,
        )

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
        project_id: str | None = None,
        tool_choice: str | None = None,
        response_format: str | None = None,
    ) -> LLMResponse:
        import asyncio

        tools_payload = [t.model_dump() for t in tools] if tools else None
        messages_payload = [m.model_dump() for m in messages]

        async def _run():
            async with self._client as client:
                return await client.generate(
                    messages=messages_payload,
                    temperature=temperature if temperature is not None else 1.0,
                    max_tokens=max_tokens or 1024,
                    top_p=top_p if top_p is not None else 1.0,
                    tools=tools_payload,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    project_id=project_id,
                )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Running inside an event loop (e.g. Jupyter, FastAPI, etc.)
            import nest_asyncio
            nest_asyncio.apply()
            data = loop.run_until_complete(_run())
        else:
            data = asyncio.run(_run())

        content = data.get("content")
        tool_calls = data.get("tool_calls")
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            raw=data,
        )

    def generate_stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        project_id: str | None = None,
        tool_choice: str | None = None,
        response_format: str | None = None,
    ):
        import asyncio

        tools_payload = [t.model_dump() for t in tools] if tools else None
        messages_payload = [m.model_dump() for m in messages]

        async def _stream():
            async with self._client as client:
                async for chunk in client.generate_stream(
                    messages=messages_payload,
                    temperature=temperature if temperature is not None else 1.0,
                    max_tokens=max_tokens or 1024,
                    top_p=top_p if top_p is not None else 1.0,
                    tools=tools_payload,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    project_id=project_id,
                ):
                    content = chunk.get("content")
                    tool_calls = chunk.get("tool_calls")
                    yield LLMResponse(
                        content=content,
                        tool_calls=tool_calls,
                        raw=chunk,
                    )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            gen = _stream()
            while True:
                try:
                    item = loop.run_until_complete(gen.__anext__())
                except StopAsyncIteration:
                    break
                yield item
        else:
            loop = asyncio.new_event_loop()
            try:
                gen = _stream()
                while True:
                    try:
                        item = loop.run_until_complete(gen.__anext__())
                    except StopAsyncIteration:
                        break
                    yield item
            finally:
                loop.close()

    def list_chat_models(self) -> list[str]:
        return self.list_models()

    def list_embedding_models(self) -> list[str]:
        return []
