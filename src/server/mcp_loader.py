"""
mcp_loader.py
-------------
Loads SHARED MCP tools that are the same for all users.
Zoho tools are per-user and handled separately by zoho_session.py.
"""

import asyncio
import logging
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools

# Only shared internal server — Zoho is per-user via zoho_session.py
MCP_CONFIGS = [
    {
        "name": "remote_internal_server",
        "transport": "streamable-http",
        "url": "http://10.192.5.51:6000/mcp",
        # "url": "http://nginx-mcp-server-1:6000/mcp"
    },
]

_tools        = []
_init_lock    = None
_shutdown_event: asyncio.Event = None


def _get_lock() -> asyncio.Lock:
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock


def _get_shutdown_event() -> asyncio.Event:
    global _shutdown_event
    if _shutdown_event is None:
        _shutdown_event = asyncio.Event()
    return _shutdown_event


async def _wait_for_shutdown():
    await _get_shutdown_event().wait()


async def _init_single_server(config: dict):
    global _tools
    name = config["name"]

    try:
        logging.info(f"[{name}] Connecting via streamable-http...")

        async with streamablehttp_client(config["url"]) as streams:
            read, write = streams[0], streams[1]

            async with ClientSession(read, write) as session:
                await session.initialize()
                server_tools = await load_mcp_tools(session)

                existing_names = {t.name for t in _tools}
                new_tools = [t for t in server_tools if t.name not in existing_names]
                _tools.extend(new_tools)

                logging.info(f"[{name}] Loaded {len(new_tools)} tools.")
                await _wait_for_shutdown()

    except* Exception as eg:
        for exc in eg.exceptions:
            logging.error(f"[{name}] Failed: {exc}")


async def _wait_for_tools_loaded(timeout: float = 30.0):
    elapsed = 0.0
    while elapsed < timeout:
        await asyncio.sleep(0.5)
        elapsed += 0.5
        if _tools:
            await asyncio.sleep(2.0)
            break


async def init_mcp_session():
    global _tools
    if _tools:
        return _tools

    async with _get_lock():
        if _tools:
            return _tools

        _get_shutdown_event().clear()

        for config in MCP_CONFIGS:
            asyncio.create_task(
                _init_single_server(config),
                name=f"mcp-{config['name']}"
            )

        await _wait_for_tools_loaded(timeout=30.0)
        logging.info(f"Shared MCP tools ready: {len(_tools)}")

    return _tools


def get_mcp_tools() -> list:
    return _tools or []


async def close_mcp_session():
    global _tools
    _get_shutdown_event().set()
    await asyncio.sleep(0.5)
    _tools = []
    logging.info("MCP sessions closed.")