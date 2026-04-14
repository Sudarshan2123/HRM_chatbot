# import os 
# import asyncio
# import logging
# from typing import List, Optional
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# from mcp.client.sse import sse_client
# from mcp.client.streamable_http import streamable_http_client 
# from langchain_mcp_adapters.tools import load_mcp_tools

# # --- Configuration ---
# ZOHO_KEY = os.getenv("Zoho_MCP_KEY")

# MCP_CONFIGS = [
#     {
#         "name": "zoho_mail_server",
#         "transport": "stdio", # Changed from streamable-http to stdio
#         "command": "cmd",
#         "args": [
#             "-y", 
#             "mcp-remote", 
#             f"https://mail-sending-replies-60069513271.zohomcp.in/mcp/{ZOHO_KEY}/message",
#             "--transport",
#             "http-only"
#         ],
#         "cwd": "." 
#     },
#     {
#         "name": "remote_internal_server",
#         "transport": "streamable-http",
#         "url": "http://10.192.5.51:6000/mcp"
#     }
# ]

# # --- Global State Management ---
# _tools = []
# _sessions: List[ClientSession] = []
# _cm_outers = [] 
# _cm_inners = [] 
# _init_lock = None

# def _get_lock():
#     global _init_lock
#     if _init_lock is None:
#         _init_lock = asyncio.Lock()
#     return _init_lock

# async def init_single_server(config):
#     """
#     Logic for a single server handshake. 
#     Parallelizing this prevents one slow server from blocking others.
#     """
#     global _tools, _sessions, _cm_outers, _cm_inners
#     transport_type = config.get("transport", "").lower()
    
#     try:
#         logging.info(f"Connecting to {config['name']} via {transport_type}...")
        
#         # 1. Setup Transport
#         if transport_type == "stdio":
#             server_params = StdioServerParameters(
#                 command=config["command"],
#                 args=config["args"],
#                 cwd=config["cwd"]
#             )
#             cm_outer = stdio_client(server_params)
#             read, write = await cm_outer.__aenter__()

#         elif transport_type == "sse":
#             cm_outer = sse_client(config["url"])
#             read, write = await cm_outer.__aenter__()

#         elif transport_type == "streamable-http":
#             cm_outer = streamable_http_client(config["url"])
#             result = await cm_outer.__aenter__()
#             read, write = result[0], result[1]
#         else:
#             logging.warning(f"Unknown transport type for {config['name']}")
#             return

#         # 2. Setup Session
#         cm_inner = ClientSession(read, write)
#         session = await cm_inner.__aenter__()
        
#         # This is usually the bottleneck - initialize() waits for a server ping
#         await session.initialize()

#         # 3. Load tools
#         server_tools = await load_mcp_tools(session)
        
#         # 4. Thread-safe update of globals
#         _tools.extend(server_tools)
#         _cm_outers.append(cm_outer)
#         _cm_inners.append(cm_inner)
#         _sessions.append(session)

#         logging.info(f"Successfully loaded {len(server_tools)} tools from {config['name']}.")

#     except Exception as e:
#         logging.error(f"Failed to initialize MCP server {config['name']}: {e}")

# async def init_mcp_session():
#     global _tools

#     if _tools:
#         return _tools

#     async with _get_lock():
#         if _tools:
#             return _tools

#         await close_mcp_session()

#         # FIXED: Run all initializations in parallel using asyncio.gather
#         tasks = [init_single_server(config) for config in MCP_CONFIGS]
#         await asyncio.gather(*tasks)

#         if not _tools:
#             logging.error("Final Result: No MCP tools were loaded from any configured server.")
            
#     return _tools

# def get_mcp_tools() -> list:
#     return _tools or []

# async def close_mcp_session():
#     global _tools, _sessions, _cm_outers, _cm_inners

#     # Close sessions first
#     for cm in _cm_inners:
#         try:
#             await cm.__aexit__(None, None, None)
#         except Exception: pass
    
#     # Close transports
#     for cm in _cm_outers:
#         try:
#             await cm.__aexit__(None, None, None)
#         except Exception: pass

#     _tools, _sessions, _cm_outers, _cm_inners = [], [], [], []
#     logging.info("All MCP sessions and transports closed.")


import os
import asyncio
import logging
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools
from src.server.zoho_auth import get_zoho_read_tools

ZOHO_KEY = os.getenv("Zoho_MCP_KEY")

MCP_CONFIGS = [
    {
        "name": "zoho_mail_server",
        "transport": "streamable-http",
        "url": f"https://mail-sending-replies-60069513271.zohomcp.in/mcp/{ZOHO_KEY}/message",
    },
    {
        "name": "remote_internal_server",
        "transport": "streamable-http",
        "url": "http://10.192.5.51:6000/mcp",
    },
]

_tools = []
_init_lock = None
_shutdown_event: asyncio.Event = None


def _get_lock():
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


def _fix_integer_enums(schema: dict) -> dict:
    """
    Recursively walk a JSON schema dict.
    For any node that has an 'enum' containing integers,
    coerce those integers to strings and set type=string.

    This fixes Zoho MCP's scheduleType field (and any similar fields)
    that ships integer enums, which LangChain's validator can't handle.
    """
    if not isinstance(schema, dict):
        return schema

    # Check for integer enums BEFORE modifying
    if "enum" in schema:
        original_enum = schema["enum"]
        has_int = any(isinstance(v, int) for v in original_enum)
        if has_int:
            schema["enum"] = [str(v) for v in original_enum]
            schema["type"] = "string"
            logging.debug(f"Fixed integer enum: {original_enum} -> {schema['enum']}")

    # Recurse into all child values
    for key, value in list(schema.items()):
        if key == "enum":
            continue  # already handled above
        if isinstance(value, dict):
            schema[key] = _fix_integer_enums(value)
        elif isinstance(value, list):
            schema[key] = [
                _fix_integer_enums(item) if isinstance(item, dict) else item
                for item in value
            ]

    return schema


def _patch_tools(tools: list, server_name: str) -> list:
    """
    After load_mcp_tools(), some tools carry their schema as a raw dict
    on args_schema instead of a Pydantic class. Walk and fix those in-place.
    """
    for tool in tools:
        schema = getattr(tool, "args_schema", None)
        if isinstance(schema, dict):
            _fix_integer_enums(schema)
            logging.info(f"[{server_name}] Patched dict args_schema for: {tool.name}")
        # Also patch the underlying MCP tool's inputSchema if present
        # (covers tools where LangChain wraps but preserves the raw object)
        inner = getattr(tool, "_tool", None) or getattr(tool, "tool", None)
        if inner:
            raw = getattr(inner, "inputSchema", None)
            if isinstance(raw, dict):
                _fix_integer_enums(raw)
                logging.info(f"[{server_name}] Patched inputSchema for: {tool.name}")
    return tools


async def init_single_server(config: dict):
    global _tools
    name = config["name"]
    transport_type = config.get("transport", "").lower()

    try:
        logging.info(f"[{name}] Connecting via {transport_type}...")

        if transport_type == "streamable-http":
            async with streamablehttp_client(config["url"]) as streams:
                read, write = streams[0], streams[1]

                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Load tools then patch integer enums in their schemas
                    server_tools = await load_mcp_tools(session)
                    server_tools = _patch_tools(server_tools, name)

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
                init_single_server(config), name=f"mcp-{config['name']}"
            )

        await _wait_for_tools_loaded(timeout=30.0)

        read_tools = get_zoho_read_tools()
        _tools.extend(read_tools)

        logging.info(f"Total tools ready: {len(_tools)}")

    return _tools


def get_mcp_tools() -> list:
    return _tools or []


async def close_mcp_session():
    global _tools
    _get_shutdown_event().set()
    await asyncio.sleep(0.5)
    _tools = []
    logging.info("MCP sessions closed.")