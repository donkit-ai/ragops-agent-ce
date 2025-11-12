from typing import Protocol

from typing import runtime_checkable
from websockets import ClientConnection


@runtime_checkable
class WsHandler(Protocol):
    async def handle_ws_connection(self, ws_connection: ClientConnection) -> None:
        """Обработка WebSocket соединения"""
        ...
