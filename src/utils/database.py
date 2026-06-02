"""
Database connection pool using psycopg v3 (async).
Uses a persistent connection pool — one pool for the app lifetime.
All queries use %s placeholders (psycopg3 standard).
"""
import os
import asyncio
import logging
from typing import Any, Optional
from contextlib import asynccontextmanager

import psycopg_pool
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()

logger  = logging.getLogger(__name__)
_DB_URI = os.getenv("DB_URI")
_pool: Optional[psycopg_pool.AsyncConnectionPool] = None


# ── Pool Lifecycle ────────────────────────────────────────────────────────────

async def init_db_pool() -> None:
    """
    Create the async connection pool.
    Called once at app startup in lifespan.
    min_size=2  — always keep 2 connections warm
    max_size=10 — never exceed 10 connections to PostgreSQL
    """
    global _pool

    if _pool is not None:
        logger.warning("[DB] Pool already initialized — skipping.")
        return

    if not _DB_URI:
        raise RuntimeError("DB_URI environment variable is not set.")

    _pool = psycopg_pool.AsyncConnectionPool(
        conninfo   = _DB_URI,
        min_size   = 2,
        max_size   = 10,
        kwargs     = {"row_factory": dict_row},   # all results as dicts
        open       = False,                        # we open manually below
    )
    await _pool.open(wait=True)
    logger.info("[DB] Connection pool initialized (min=2 max=10).")


async def close_db_pool() -> None:
    """Close the pool gracefully at shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("[DB] Connection pool closed.")


def get_db_pool() -> Optional[psycopg_pool.AsyncConnectionPool]:
    """Return the pool for direct transaction use in SessionManager."""
    return _pool


# ── Transaction Context Manager ───────────────────────────────────────────────

@asynccontextmanager
async def db_transaction():
    """
    Async context manager for atomic transactions.

    Usage:
        async with db_transaction() as conn:
            await conn.execute("UPDATE ...")
            await conn.execute("INSERT ...")
        # auto-commits on exit, rolls back on exception

    Example in SessionManager:
        async with db_transaction() as conn:
            await conn.execute("UPDATE employee_sessions ...", (emp_code,))
            await conn.execute("INSERT INTO auth_audit_log ...", (...))
    """
    if _pool is None:
        raise RuntimeError("[DB] Pool not initialized. Call init_db_pool() first.")

    async with _pool.connection() as conn:
        try:
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise


# ── Query Helpers ─────────────────────────────────────────────────────────────

async def execute_query(
    query:  str,
    params: tuple = None,
    fetch:  bool  = True,
) -> Any:
    """
    Run a single query using a pooled connection.
    Uses %s placeholders (psycopg3 standard).

    fetch=True  → returns list of dicts (SELECT)
    fetch=False → commits and returns None (INSERT/UPDATE/DELETE)
    """
    if _pool is None:
        raise RuntimeError("[DB] Pool not initialized. Call init_db_pool() first.")

    max_retries = 3
    retry_delay = 1.0

    for attempt in range(max_retries):
        try:
            async with _pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, params or ())

                    if fetch:
                        return await cur.fetchall()
                    else:
                        await conn.commit()
                        return None

        except psycopg.OperationalError as e:
            # Connection-level error — worth retrying
            logger.warning(
                "[DB] Attempt %d/%d failed | error=%s | query=%s",
                attempt + 1, max_retries, e, query[:80],
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("[DB] All retries exhausted | query=%s | error=%s", query[:80], e)
                raise

        except Exception as e:
            # Non-connection error (syntax, constraint) — don't retry
            logger.error("[DB] Query failed | query=%s | params=%s | error=%s", query[:80], params, e)
            raise


async def execute_query_single(
    query:  str,
    params: tuple = None,
) -> Optional[dict]:
    """Return first row as dict or None."""
    results = await execute_query(query, params, fetch=True)
    return results[0] if results else None