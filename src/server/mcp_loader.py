import os 
import asyncio
import logging
from typing import List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client 
from langchain_mcp_adapters.tools import load_mcp_tools

# --- Configuration ---
ZOHO_KEY = os.getenv("Zoho_MCP_KEY")

MCP_CONFIGS = [
    {
        "name": "zoho_mail_server",
        "transport": "streamable-http",
        # FIXED: Use f-string to properly inject the key into the URL
        "url": f"https://mail-sending-replies-60069513271.zohomcp.in/mcp/{ZOHO_KEY}/message"
    },
    {
        "name": "remote_internal_server",
        "transport": "streamable-http",
        "url": "http://10.192.5.51:6000/mcp"
    }
]

# --- Global State Management ---
_tools = []
_sessions: List[ClientSession] = []
_cm_outers = [] 
_cm_inners = [] 
_init_lock = None

def _get_lock():
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock

async def init_single_server(config):
    """
    Logic for a single server handshake. 
    Parallelizing this prevents one slow server from blocking others.
    """
    global _tools, _sessions, _cm_outers, _cm_inners
    transport_type = config.get("transport", "").lower()
    
    try:
        logging.info(f"Connecting to {config['name']} via {transport_type}...")
        
        # 1. Setup Transport
        if transport_type == "stdio":
            server_params = StdioServerParameters(
                command=config["command"],
                args=config["args"],
                cwd=config["cwd"]
            )
            cm_outer = stdio_client(server_params)
            read, write = await cm_outer.__aenter__()

        elif transport_type == "sse":
            cm_outer = sse_client(config["url"])
            read, write = await cm_outer.__aenter__()

        elif transport_type == "streamable-http":
            cm_outer = streamable_http_client(config["url"])
            result = await cm_outer.__aenter__()
            read, write = result[0], result[1]
        else:
            logging.warning(f"Unknown transport type for {config['name']}")
            return

        # 2. Setup Session
        cm_inner = ClientSession(read, write)
        session = await cm_inner.__aenter__()
        
        # This is usually the bottleneck - initialize() waits for a server ping
        await session.initialize()

        # 3. Load tools
        server_tools = await load_mcp_tools(session)
        
        # 4. Thread-safe update of globals
        _tools.extend(server_tools)
        _cm_outers.append(cm_outer)
        _cm_inners.append(cm_inner)
        _sessions.append(session)

        logging.info(f"Successfully loaded {len(server_tools)} tools from {config['name']}.")

    except Exception as e:
        logging.error(f"Failed to initialize MCP server {config['name']}: {e}")

async def init_mcp_session():
    global _tools

    if _tools:
        return _tools

    async with _get_lock():
        if _tools:
            return _tools

        await close_mcp_session()

        # FIXED: Run all initializations in parallel using asyncio.gather
        tasks = [init_single_server(config) for config in MCP_CONFIGS]
        await asyncio.gather(*tasks)

        if not _tools:
            logging.error("Final Result: No MCP tools were loaded from any configured server.")
            
    return _tools

def get_mcp_tools() -> list:
    return _tools or []

async def close_mcp_session():
    global _tools, _sessions, _cm_outers, _cm_inners

    # Close sessions first
    for cm in _cm_inners:
        try:
            await cm.__aexit__(None, None, None)
        except Exception: pass
    
    # Close transports
    for cm in _cm_outers:
        try:
            await cm.__aexit__(None, None, None)
        except Exception: pass

    _tools, _sessions, _cm_outers, _cm_inners = [], [], [], []
    logging.info("All MCP sessions and transports closed.")