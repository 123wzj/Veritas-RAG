"""Persistent LangGraph checkpoint lifecycle for resumable ReAct runs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.core.config import settings


class ReactCheckpointManager:
    """Own one async SQLite saver and one compiled graph per app event loop."""

    def __init__(self) -> None:
        self._lock: Optional[asyncio.Lock] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connection: Optional[aiosqlite.Connection] = None
        self._saver: Optional[AsyncSqliteSaver] = None
        self._graph: Any = None

    async def start(self) -> Any:
        loop = asyncio.get_running_loop()
        if self._graph is not None:
            if self._loop is not loop:
                raise RuntimeError("Checkpoint manager cannot be shared across event loops")
            return self._graph
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._graph is not None:
                return self._graph
            checkpoint_path = Path(settings.AGENT_CHECKPOINT_PATH)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            connection = await aiosqlite.connect(str(checkpoint_path))
            await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute("PRAGMA synchronous=NORMAL")
            await connection.execute("PRAGMA busy_timeout=5000")
            saver = AsyncSqliteSaver(connection)
            await saver.setup()

            # Deferred import avoids a graph <-> checkpoint lifecycle cycle.
            from backend.agent.graph import create_react_graph

            self._loop = loop
            self._connection = connection
            self._saver = saver
            self._graph = create_react_graph(checkpointer=saver)
            return self._graph

    async def get_graph(self) -> Any:
        return await self.start()

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
        self._connection = None
        self._saver = None
        self._graph = None
        self._loop = None
        self._lock = None


react_checkpoint_manager = ReactCheckpointManager()

