import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import anyio
import pydantic
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.server.zoho_key_store import get_zoho_key

logger = logging.getLogger(__name__)

_SCHEMA_TTL = 900.0


@dataclass
class _SchemaCache:
    url:        str
    schemas:    list         = field(default_factory=list)
    fetched_at: float        = 0.0
    lock:       asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


_cache: dict[int, _SchemaCache] = {}
_cache_lock: Optional[asyncio.Lock] = None


def _get_cache_lock() -> asyncio.Lock:
    global _cache_lock
    if _cache_lock is None:
        _cache_lock = asyncio.Lock()
    return _cache_lock


# ── Both functions use anyio.to_thread.run_sync for isolation ─────────────

async def _do_list_tools(url: str) -> list:
    def _sync():
        async def _inner():
            async with streamablehttp_client(url) as (read, write, *_):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return result.tools
        return anyio.run(_inner)
    return await anyio.to_thread.run_sync(_sync)


async def _do_call_tool(url: str, name: str, arguments: dict) -> str:
    def _sync():
        async def _inner():
            async with streamablehttp_client(url) as (read, write, *_):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments=arguments)
                    texts = [
                        block.text if hasattr(block, "text")
                        else json.dumps(block.data) if hasattr(block, "data")
                        else str(block)
                        for block in result.content
                    ]
                    return "\n".join(texts) if texts else "(no output)"
        return anyio.run(_inner)
    return await anyio.to_thread.run_sync(_sync)


# ── Schema fetch + cache ───────────────────────────────────────────────────

async def _fetch_schemas(emp_code: int, url: str) -> list[dict]:
    logger.info("[zoho:%s] Fetching schemas from MCP endpoint", emp_code)  # SECURITY: Don't log URL
    try:
        tools = await _do_list_tools(url)
        schemas = [
            {
                "name":        t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema if isinstance(t.inputSchema, dict) else {},
            }
            for t in tools
        ]
        for s in schemas:
            if s["name"] == "ZohoMail_sendEmail":
                logger.debug("[MCP_SCHEMA] ZohoMail_sendEmail schema loaded")  # SECURITY: Redacted details
        logger.info("[zoho:%s] Got %d schemas", emp_code, len(schemas))
        return schemas
    except BaseException as exc:
        if hasattr(exc, 'exceptions'):
            for i, sub in enumerate(exc.exceptions):
                logger.error("[zoho:%s] Sub-exception #%d: [%s]", emp_code, i, type(sub).__name__)
        else:
            logger.error("[zoho:%s] Schema fetch failed: [%s]", emp_code, type(exc).__name__)
        return []


async def _get_schemas(emp_code: int, url: str) -> list[dict]:
    async with _get_cache_lock():
        entry = _cache.get(emp_code)
        if entry is None or entry.url != url:
            entry = _SchemaCache(url=url)
            _cache[emp_code] = entry

    async with entry.lock:
        now = time.monotonic()
        if entry.schemas and (now - entry.fetched_at) < _SCHEMA_TTL:
            logger.debug("[zoho:%s] Cache hit — %d tools", emp_code, len(entry.schemas))
            return entry.schemas

        schemas = await _fetch_schemas(emp_code, url)
        if schemas:
            entry.schemas    = schemas
            entry.fetched_at = time.monotonic()

        return entry.schemas


# ── LangChain tool wrapper ─────────────────────────────────────────────────

class _ZohoTool(BaseTool):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    zoho_url: str

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")

    async def _arun(self, **kwargs: Any) -> str:
        print(f"[ZOHO_ARUN] {self.name} kwargs={json.dumps(kwargs, default=str)[:200]}")
        try:
            # Pass kwargs as-is — MCP expects the nested structure
            response = await _do_call_tool(self.zoho_url, self.name, kwargs)
            print(f"[ZOHO_ARUN] {self.name} response={str(response)[:200]}")
            return response
        except Exception as exc:
            print(f"[ZOHO_ARUN] {self.name} EXCEPTION={exc}")
            logger.error("[ZohoTool] %s failed: [%s] %s", self.name, type(exc).__name__, exc)
            return f"Tool call failed: {exc}"

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> ToolMessage:
        if isinstance(input, dict):
            args         = input.get("args", {}) or {}
            tool_call_id = input.get("id", "")
        else:
            args         = {}
            tool_call_id = ""

        logger.info("[ZohoTool.ainvoke] %s args=%s", self.name, json.dumps(args, default=str)[:200])

        # Pass entire args dict as-is — DynamicToolNode already structured it correctly
        result = await self._arun(**args)
        return ToolMessage(content=result, tool_call_id=tool_call_id, name=self.name)


def _schemas_to_tools(schemas: list[dict], url: str) -> list[BaseTool]:
    tools = []
    for s in schemas:
        try:
            tools.append(_ZohoTool(
                name=s["name"],
                description=s["description"],
                zoho_url=url,
            ))
        except Exception as exc:
            logger.warning("Could not wrap tool %s: %s", s["name"], exc)
    return tools


# ── Public API ─────────────────────────────────────────────────────────────

async def get_zoho_tools_for_user(emp_code: int) -> list[BaseTool]:
    url = await get_zoho_key(emp_code)
    logger.debug("[zoho:%s] MCP URL retrieved from key store", emp_code)  # SECURITY: Don't log URL
    if not url:
        logger.warning("[zoho:%s] No MCP URL configured", emp_code)
        return []

    url = url.strip().rstrip("/")
    schemas = await _get_schemas(emp_code, url)
    if not schemas:
        return []

    tools = _schemas_to_tools(schemas, url)
    logger.info("[zoho:%s] Returning %d tools", emp_code, len(tools))  # SECURITY: Redacted tool names
    return tools


async def invalidate_zoho_session(emp_code: int) -> None:
    async with _get_cache_lock():
        _cache.pop(emp_code, None)
    logger.info("[zoho:%s] Cache cleared", emp_code)


async def close_all_zoho_sessions() -> None:
    async with _get_cache_lock():
        _cache.clear()
    logger.info("All Zoho caches cleared")