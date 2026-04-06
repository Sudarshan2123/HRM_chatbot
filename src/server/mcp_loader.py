from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
# ADD: Import for Streamable HTTP
from mcp.client.streamable_http import streamable_http_client 
from langchain_mcp_adapters.tools import load_mcp_tools
import asyncio
import logging

# Updated Transport string
TRANSPORT = "Streamable-http"
MCP_SERVER_PATH = "E:\\Agentic_Chatbot\\Mcp"
MCP_SSE_URL = "http://localhost:8000/sse"
# Streamable HTTP usually hits the base URL or /message depending on your server config
MCP_HTTP_URL = "http://10.192.5.51:6000/mcp" 

_tools = None
_session = None
_cm_outer = None 
_cm_inner = None 
_init_lock = None

def _get_lock():
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock

async def _is_session_alive() -> bool:
    global _session
    if _session is None:
        return False
    try:
        # Pings the server to see if the session is valid
        await _session.list_tools()
        return True
    except Exception:
        return False

async def init_mcp_session():
    global _tools, _session, _cm_outer, _cm_inner

    if _session is not None and await _is_session_alive():
        return _tools

    async with _get_lock():
        if _session is not None and await _is_session_alive():
            return _tools

        await close_mcp_session()

        try:
            if TRANSPORT.lower() == "stdio":
                server_params = StdioServerParameters(
                    command="python",
                    args=["mcp_server.py"],
                    cwd=MCP_SERVER_PATH
                )
                _cm_outer = stdio_client(server_params)
                read, write = await _cm_outer.__aenter__()

            elif TRANSPORT.lower() == "sse":
                _cm_outer = sse_client(MCP_SSE_URL)
                read, write = await _cm_outer.__aenter__()

            else:
                # Streamable HTTP returns (read, write, get_session_id)
                logging.info(f"Connecting via Streamable HTTP to {MCP_HTTP_URL}")
                _cm_outer = streamable_http_client(MCP_HTTP_URL)
                result = await _cm_outer.__aenter__()

                # Safely unpack regardless of tuple length
                read, write = result[0], result[1]
                logging.info("Streamable HTTP context entered successfully.")

            _cm_inner = ClientSession(read, write)
            _session = await _cm_inner.__aenter__()
            await _session.initialize()

            _tools = await load_mcp_tools(_session)
            logging.info(f"[{TRANSPORT}] MCP tools loaded: {[t.name for t in _tools]}")

        except Exception as e:
            logging.error(f"MCP server unavailable ({TRANSPORT}): {e}")
            await close_mcp_session()
            _tools = []

    return _tools


def get_mcp_tools() -> list:
    return _tools or []

async def close_mcp_session():
    global _tools, _session, _cm_outer, _cm_inner

    if _cm_inner is not None:
        try:
            await _cm_inner.__aexit__(None, None, None)
        except Exception: pass
        _cm_inner = None

    if _cm_outer is not None:
        try:
            await _cm_outer.__aexit__(None, None, None)
        except Exception: pass
        _cm_outer = None

    _session = None
    _tools = None
    logging.info("MCP session closed.")
    