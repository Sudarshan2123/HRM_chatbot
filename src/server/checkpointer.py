from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
import psycopg
import os

DB_URI = os.getenv("DB_URI")

_pool: AsyncConnectionPool = None
_checkpointer: AsyncPostgresSaver = None


async def init_checkpointer() -> AsyncPostgresSaver:
    global _pool, _checkpointer

    # setup() uses CREATE INDEX CONCURRENTLY which cannot run inside a transaction
    async with await psycopg.AsyncConnection.connect(DB_URI, autocommit=True) as conn:
        await AsyncPostgresSaver(conn).setup()

    _pool = AsyncConnectionPool(
        conninfo=DB_URI,
        min_size=2,
        max_size=10,
        max_idle=300,
        reconnect_timeout=30,
        timeout=10,
        open=False
    )
    await _pool.open()

    _checkpointer = AsyncPostgresSaver(_pool)
    return _checkpointer


async def close_checkpointer():
    global _pool
    if _pool:
        await _pool.close()
