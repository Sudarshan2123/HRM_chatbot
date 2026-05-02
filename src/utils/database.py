"""
Database connection utilities with retry logic.
"""
import os
import asyncio
import logging
from typing import Optional, Any
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row

load_dotenv()

logger = logging.getLogger(__name__)

_DB_URI = os.getenv("DB_URI")


async def execute_query(query: str, params: tuple = None, fetch: bool = True) -> Any:
    max_retries = 3
    retry_delay = 1.0

    for attempt in range(max_retries):
        conn = None
        try:
            conn = await psycopg.AsyncConnection.connect(_DB_URI)
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, params or ())
                if fetch:
                    result = await cur.fetchall()
                    return result
                else:
                    await conn.commit()   # ← was missing, writes never persisted
                    return None

        except Exception as e:
            logger.warning(
                "DB attempt %d failed | query=%s | params=%s | error=%s",
                attempt + 1, query[:80], params, e
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("All DB attempts failed | query=%s | error=%s", query[:80], e)
                raise
        finally:
            if conn:
                await conn.close()


async def execute_query_single(query: str, params: tuple = None) -> Optional[dict]:
    results = await execute_query(query, params, fetch=True)
    return results[0] if results else None


async def init_db_pool():
    pass


async def close_db_pool():
    pass