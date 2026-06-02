import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import os
import uuid
import urllib
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
from pydantic import ValidationError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from src.constants.token import Token
from src.entity._init_ import TextRequest, TextToSpeechRequest
from src.entity.responseHelper import handle_result, read_conversation
from src.entity.interrupt_helpers import _extract_interrupt_payload
from src.entity.DBHelper import (
    init_db_pool, close_db_pool,
    execute_query, execute_query_single,
    save_user_thread, update_thread_active,
    verify_thread_ownership,
)
from src.entity.tokenHelper import verify_token, assert_token_matches_emp
from src.server.zoho_session import close_all_zoho_sessions, invalidate_zoho_session
from src.server.session_manager import SessionManager
from src.pipeline.Login import Login
from src.server.checkpointer import close_checkpointer, init_checkpointer
from src.utils.common import decrypt_credentials
from src.utils.security import check_no_query_params
from src.server.zoho_key_store import save_zoho_key, get_zoho_key, has_zoho_key
from src.utils.Text_To_Speach import TextToSpeach
from src.logging import logger
from graphm import create_intent_driven_agent
from src.server.mcp_loader import init_mcp_session, close_mcp_session
from src.pipeline.agent_registry import AGENT_REGISTRY   # ← registry import
from statenode import *
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from langgraph.errors import GraphRecursionError
# ── Env ────────────────────────────────────────────────────────────────────────
os.environ["FASTEMBED_CACHE_PATH"] = os.getenv("FASTEMBED_CACHE_PATH", "/tmp/fastembed")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"]       = "error"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]   = "1"
os.environ["HF_HUB_LOCAL_FILES_ONLY"]         = "0"
os.environ["HUGGINGFACE_HUB_NO_SYMLINK"]      = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"]         = "1"

_token_handler = Token()
load_dotenv()
def _rate_limit_key(request: Request) -> str:
    return _token_handler.get_emp_code_from_request(request)
limiter = Limiter(key_func=_rate_limit_key)
_graph = None

# ── Streaming helpers ──────────────────────────────────────────────────────────
def _extract_content(content) -> str:
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content or ""


def _sse(payload: dict) -> str:
    """Format a dict as a Server-Sent Event string."""
    return f"data: {json.dumps(payload)}\n\n"


def _is_error_text(text: str) -> bool:
    ERROR_KEYWORDS = {"error", "failed", "not found", "404", "500", "sorry", "unavailable"}
    lower = text.lower()
    return any(kw in lower for kw in ERROR_KEYWORDS)


# ── Registry-driven node sets (rebuilt at import time) ────────────────────────
# Agents that stream token-by-token via on_chat_model_stream
_STREAMING_NODES: set[str] = {
    cfg.name for cfg in AGENT_REGISTRY.values() if cfg.streams_tokens
}

_CHAIN_END_NODES: set[str] = {
    cfg.name for cfg in AGENT_REGISTRY.values() if not cfg.streams_tokens
} | {
    cfg.tool_node for cfg in AGENT_REGISTRY.values() if cfg.tool_node
}


async def _check_for_interrupt(graph, config: dict) -> dict | None:
    try:
        state = await graph.aget_state(config)
        if state and state.tasks:
            for task in state.tasks:
                interrupts = getattr(task, "interrupts", None) or []
                if interrupts:
                    return _extract_interrupt_payload(interrupts[0].value)
    except Exception as e:
        logger.warning(f"[_check_for_interrupt] aget_state failed (non-fatal): {e}")
    return None


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    try:
        logger.info("Lifespan: Initializing database pool...")
        await init_db_pool()

        logger.info("Lifespan: Initializing MCP sessions...")
        await init_mcp_session()

        logger.info("Lifespan: Initializing checkpointer...")
        checkpointer = await init_checkpointer()

        logger.info("Lifespan: Creating agentic graph...")
        _graph = create_intent_driven_agent(checkpointer=checkpointer)

        # ── Log registered agents at startup ─────────────────────────
        logger.info(
            f"Lifespan: {len(AGENT_REGISTRY)} agents registered: "
            f"{list(AGENT_REGISTRY.keys())}"
        )
        logger.info(f"  Streaming nodes : {_STREAMING_NODES}")
        logger.info(f"  Chain-end nodes : {_CHAIN_END_NODES}")

        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            SessionManager.cleanup_expired_sessions,
            trigger="cron", hour=2, minute=0, id="cleanup_sessions",
        )
        scheduler.start()
        app.state.scheduler  = scheduler
        app.state.tts_engine = TextToSpeach()

        logger.info("Lifespan: Startup complete.")
        yield

    finally:
        logger.info("Lifespan: Shutting down...")
        if hasattr(app.state, "scheduler"):
            app.state.scheduler.shutdown(wait=False)
        await close_mcp_session()
        await close_all_zoho_sessions()
        await close_checkpointer()
        await close_db_pool()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "HRMS AI Assistant",
    description = "LangGraph multi-agent system for HRMS queries",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5000,http://localhost:3000"
    ).split(",")
    if o.strip()
]

# SECURITY: Validate CORS origins — reject wildcards in production
if "*" in ALLOWED_ORIGINS:
    logger.warning("WARNING: Wildcard (*) CORS origin detected — not recommended for production")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["GET", "POST", "OPTIONS"],
    allow_headers     = ["Authorization", "Content-Type"],
    expose_headers    = [],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    registered = {
        name: {
            "display_name": cfg.display_name,
            "has_tools":    cfg.has_tools,
            "tool_node":    cfg.tool_node,
            "keywords":     cfg.keywords,
        }
        for name, cfg in AGENT_REGISTRY.items()
    }
    return {"status": "ok", "agents": registered}


@app.post("/login")
async def login(request: Request, data: EncryptedLoginData = Body(...)):
    try:
        check_no_query_params(request)

        username = urllib.parse.unquote(data.username)
        password = data.password

        if not username or not password:
            raise HTTPException(status_code=400, detail="Employee code and password are required.")

        decrypted = decrypt_credentials({"username": username, "password": password})
        if not decrypted.get("username") or not decrypted.get("password"):
            raise HTTPException(status_code=400, detail="Employee code and password are required.")

        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")

        result = await Login().login_user(
            decrypted["username"], decrypted["password"],
            ip_address=ip_address, user_agent=user_agent,
        )

        if result.get("status") == "success":
            response = JSONResponse(
                content={
                    "status": "success",
                    "user_Status": result.get("user_Status"),
                    "access_token": result.get("access_token"),
                    "token_type": "bearer",
                }
            )
            response.headers["Authorization"] = f"Bearer {result.get('access_token')}"
            return response

        return JSONResponse({"status": "error", "message": "Invalid credentials."}, status_code=401)

    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        logger.error(f"[login] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Login service unavailable.")


@app.post("/logout")
async def logout(request: Request, token_payload: dict = Depends(verify_token)):
    session_id = token_payload.get("session_id")
    ip_address = request.client.host if request.client else None
    await SessionManager.revoke_session(session_id=session_id, ip_address=ip_address, reason="logout")
    return {"status": "success", "message": "Logged out successfully."}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(req: ChatRequest,  request: Request,token_payload: dict = Depends(verify_token)):
    assert_token_matches_emp(token_payload, req.emp_code)

    if req.thread_id:
        is_owner = await verify_thread_ownership(req.emp_code, req.thread_id)
        if not is_owner:
            raise HTTPException(status_code=403, detail="Access denied.")

    is_new_thread = not req.thread_id
    thread_id     = req.thread_id or str(uuid.uuid4())
    config        = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}

    await save_user_thread(req.emp_code, thread_id)

    try:
        result = await _graph.ainvoke(
            {
                "messages":  [HumanMessage(content=req.message)],
                "emp_code":  req.emp_code,
                "firm_id":   req.firm_id,
                "emp_name":  req.emp_name,
                "role_id":   req.role_id,
                "firm_name": req.firm_name,
            },
            config=config,
        )
    except GraphRecursionError:
        raise HTTPException(status_code=200, detail="Request too complex, please simplify.")
    except Exception as e:
        logger.error(f"[chat] Graph invocation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat service error.")

    
    await update_thread_active(thread_id)

    interrupt_payload = await _check_for_interrupt(_graph, config)
    if interrupt_payload:
        msg = interrupt_payload.get("message") or "Awaiting your input."
        logger.info(f"[chat] interrupt detected action='{interrupt_payload.get('action')}'")
        return JSONResponse(content={
            "thread_id":     thread_id,
            "is_new_thread": is_new_thread,
            "response":      msg,
            "status":        "waiting_for_input",
            "action":        interrupt_payload.get("action"),
            "options":       interrupt_payload.get("options"),
            "values":        interrupt_payload.get("values"),
            "intent":        "general",
        })

    response_dict                  = handle_result(result, thread_id).model_dump()
    response_dict["is_new_thread"] = is_new_thread
    return JSONResponse(content=response_dict)


@app.post("/chat/resume", response_model=ChatResponse)
@limiter.limit("10/minute")
async def resume(req: ResumeRequest, request: Request, token_payload: dict = Depends(verify_token)):
    assert_token_matches_emp(token_payload, req.emp_code)

    is_owner = await verify_thread_ownership(req.emp_code, req.thread_id)
    if not is_owner:
        raise HTTPException(status_code=403, detail="Access denied.")

    config   = {"configurable": {"thread_id": req.thread_id}, "recursion_limit": 25}
    decision = req.decision.strip().lower()

    await update_thread_active(req.thread_id)
    logger.info(f"[resume] emp={req.emp_code} decision='{decision}'")

    try:
        result = await _graph.ainvoke(Command(resume=decision), config=config)
    except PermissionError as e:
        return ChatResponse(thread_id=req.thread_id, intent="leave", response=str(e), status="completed")
    except ValueError as e:
        return ChatResponse(thread_id=req.thread_id, intent="leave", response=str(e), status="completed")
    except GraphRecursionError:
        raise HTTPException(status_code=200, detail="Request too complex, please simplify.")
    except Exception as e:
        logger.error(f"[resume] Graph invocation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Resume service error.")


    # Check for chained interrupt (next step in any multi-step agent flow)
    interrupt_payload = await _check_for_interrupt(_graph, config)
    if interrupt_payload:
        msg = interrupt_payload.get("message") or "Awaiting your input."
        logger.info(f"[resume] chained interrupt action='{interrupt_payload.get('action')}'")
        return ChatResponse(
            thread_id = req.thread_id,
            intent    = "general",
            response  = msg,
            status    = "waiting_for_input",
            action    = interrupt_payload.get("action"),
            options   = interrupt_payload.get("options"),
            values    = interrupt_payload.get("values"),
        )

    return handle_result(result, req.thread_id)


@app.post("/chat/stream")
@limiter.limit("10/minute")
async def chat_stream(req: ChatRequest, request: Request, token_payload: dict = Depends(verify_token)):
    assert_token_matches_emp(token_payload, req.emp_code)

    thread_id = req.thread_id or str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}

    thread_in_db = await execute_query_single(
        "SELECT thread_id FROM user_conversations WHERE thread_id = %s", (thread_id,)
    )

    if thread_in_db:
        is_owner = await verify_thread_ownership(req.emp_code, thread_id)
        if not is_owner:
            raise HTTPException(status_code=403, detail="Access denied.")

    is_new_thread = not thread_in_db
    await save_user_thread(req.emp_code, thread_id)

    async def event_stream():
        seen_content:    set[str]   = set()
        last_tool_error: str | None = None

        def _emit_token(text: str) -> str | None:
            """Emit a token only if not already seen."""
            clean = text.strip()
            if clean and clean not in seen_content:
                seen_content.add(clean)
                return _sse({"type": "token", "content": clean, "thread_id": thread_id})
            return None

        def _emit_agent_message(msgs: list, node: str) -> str | None:
            """
            Extract and emit the last meaningful AIMessage from any agent's
            on_chain_end output. Works for ALL agents — registry-driven.
            """
            if not msgs:
                return None
            last = msgs[-1]

            if isinstance(last, AIMessage):
                text           = _extract_content(last.content)
                has_tool_calls = bool(getattr(last, "tool_calls", None))
                if text and not has_tool_calls:
                    return _emit_token(text)

            elif isinstance(last, ToolMessage):
                # Surface tool errors when LLM hasn't replied yet
                text = last.content or ""
                if _is_error_text(text) and text not in seen_content:
                    seen_content.add(text)
                    return _sse({"type": "error_message", "content": text.strip(), "thread_id": thread_id})

            return None

        yield _sse({"type": "thread_init", "thread_id": thread_id, "is_new_thread": is_new_thread})

        try:
            async for event in _graph.astream_events(
                {
                    "messages":  [HumanMessage(content=req.message)],
                    "emp_code":  req.emp_code,
                    "firm_id":   req.firm_id,
                    "emp_name":  req.emp_name,
                    "role_id":   req.role_id,
                    "firm_name": req.firm_name,
                },
                config  = config,
                version = "v2",
            ):
                kind = event["event"]
                name = event.get("name", "")
                node = event.get("metadata", {}).get("langgraph_node", "")

                # ══════════════════════════════════════════════════════════════
                # 1. TOKEN STREAMING
                #    Streaming LLM agents (no dedicated tool node) emit tokens
                #    via on_chat_model_stream. Registry-driven — any new agent
                #    without a tool_node automatically streams here.
                # ══════════════════════════════════════════════════════════════
                if kind == "on_chat_model_stream" and node in _STREAMING_NODES:
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        text = _extract_content(chunk.content)
                        if text:
                            yield _sse({"type": "token", "content": text, "thread_id": thread_id})

                # ══════════════════════════════════════════════════════════════
                # 2. CHAIN-END AGENT MESSAGES
                #    Agents with dedicated tool nodes (e.g. leave_agent) are
                #    plain async fns — their final output comes via on_chain_end.
                #    Registry-driven — any new agent with tool_node set lands here.
                # ══════════════════════════════════════════════════════════════
                elif kind == "on_chain_end" and node in _CHAIN_END_NODES:
                    output = event.get("data", {}).get("output")
                    if not isinstance(output, dict):
                        continue
                    result = _emit_agent_message(output.get("messages", []), node)
                    if result:
                        yield result

                # ══════════════════════════════════════════════════════════════
                # 3. SILENT LLM AFTER TOOL ERROR
                #    Streaming agents that go quiet after a tool failure —
                #    surface the last tool error as an error_message instead.
                # ══════════════════════════════════════════════════════════════
                elif kind == "on_chain_end" and node in _STREAMING_NODES:
                    if not last_tool_error:
                        continue
                    output = event.get("data", {}).get("output") or {}
                    msgs   = output.get("messages", []) if isinstance(output, dict) else []
                    if not msgs:
                        continue
                    last = msgs[-1]
                    if isinstance(last, AIMessage):
                        text           = _extract_content(last.content)
                        has_tool_calls = bool(getattr(last, "tool_calls", None))
                        if not text and not has_tool_calls:
                            error_msg = f"An error occurred: {last_tool_error}"
                            if error_msg not in seen_content:
                                seen_content.add(error_msg)
                                yield _sse({"type": "error_message", "content": error_msg, "thread_id": thread_id})

                # ══════════════════════════════════════════════════════════════
                # 4. TOOL END — track errors for silent-LLM detection
                #    Works for ALL tool nodes — shared and dedicated.
                # ══════════════════════════════════════════════════════════════
                elif kind == "on_tool_end":
                    output_str = event["data"].get("output", "")
                    if not isinstance(output_str, str):
                        output_str = str(output_str)
                    if _is_error_text(output_str):
                        last_tool_error = output_str
                        logger.warning(f"[stream] tool error from '{name}' node='{node}': {output_str[:120]}")
                    else:
                        last_tool_error = None

                # ══════════════════════════════════════════════════════════════
                # 5. INTERRUPT — any agent can raise an interrupt
                #    (leave confirmation, email send confirmation, etc.)
                # ══════════════════════════════════════════════════════════════
                elif kind == "on_chain_stream":
                    chunk = event["data"].get("chunk", {})
                    if "__interrupt__" in chunk:
                        interrupt_data = chunk["__interrupt__"][0].value
                        payload = (
                            interrupt_data
                            if isinstance(interrupt_data, dict)
                            else {"message": str(interrupt_data)}
                        )
                        logger.info(f"[stream] interrupt action='{payload.get('action')}' node='{node}'")
                        yield _sse({
                            "type":      "interrupt",
                            "content":   payload.get("message", ""),
                            "action":    payload.get("action"),
                            "options":   payload.get("options"),
                            "values":    payload.get("values"),
                            "thread_id": thread_id,
                        })

                # ══════════════════════════════════════════════════════════════
                # 6. GUARD BLOCKED — guardrails rejected the message
                # ══════════════════════════════════════════════════════════════
                elif kind == "on_chain_end" and node == "guard":
                    output = event.get("data", {}).get("output") or {}
                    if isinstance(output, dict) and output.get("responded"):
                        msgs = output.get("messages", [])
                        if msgs:
                            text = _extract_content(msgs[-1].content)
                            if text and text not in seen_content:
                                seen_content.add(text)
                                yield _sse({"type": "guardrail", "content": text, "thread_id": thread_id})

                # ══════════════════════════════════════════════════════════════
                # 7. SUPERVISOR — internal routing, never sent to client
                #    Logged for observability / debugging only.
                # ══════════════════════════════════════════════════════════════
                elif kind == "on_chain_end" and node == "supervisor":
                    output = event.get("data", {}).get("output") or {}
                    if isinstance(output, dict):
                        next_agent = output.get("next_agent")
                        agent_queue = output.get("agent_queue")
                        reason = output.get("reason")
                    else:
                        next_agent = getattr(output, "next_agent", None)
                        agent_queue = getattr(output, "agent_queue", None)
                        reason = getattr(output, "reason", None)
                    logger.info(
                        f"[stream:supervisor] next='{next_agent}' "
                        f"queue={agent_queue} "
                        f"reason='{reason}'"
                    )
        except GraphRecursionError:
            yield _sse({"type": "error", "content": "Request too complex, please simplify."})
            return  # stop the generator cleanly
        except Exception as e:
            logger.error(f"[chat_stream] Stream error: {e}", exc_info=True)
            yield _sse({"type": "error", "content": "Stream error occurred."})

        finally:
            await update_thread_active(thread_id)
            yield _sse({"type": "done", "thread_id": thread_id})

    return StreamingResponse(
        event_stream(),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── DB Endpoints ───────────────────────────────────────────────────────────────

@app.post("/db/conversation")
async def get_conversation_from_db(req: ConversationRequest, token_payload: dict = Depends(verify_token)):
    assert_token_matches_emp(token_payload, req.emp_code)

    is_owner = await verify_thread_ownership(req.emp_code, req.thread_id)
    if not is_owner:
        raise HTTPException(status_code=403, detail="Access denied.")

    result             = await read_conversation(req.thread_id, _graph)
    result["emp_code"] = req.emp_code
    return result


@app.post("/db/conversations/user")
async def get_user_conversations(req: UserConversationsRequest, token_payload: dict = Depends(verify_token)):
    assert_token_matches_emp(token_payload, req.emp_code)

    rows = await execute_query(
        """
        SELECT
            uc.thread_id,
            uc.created_at,
            uc.last_active,
            COUNT(c.checkpoint_id) AS total_checkpoints
        FROM user_conversations uc
        LEFT JOIN checkpoints c ON uc.thread_id = c.thread_id
        WHERE uc.emp_code = %s
        GROUP BY uc.thread_id, uc.created_at, uc.last_active
        ORDER BY uc.last_active DESC
        LIMIT 10
        """,
        (req.emp_code,)
    )
    return {
        "emp_code":            req.emp_code,
        "total_conversations": len(rows) if rows else 0,
        "conversations": [
            {
                "thread_id":         r["thread_id"],
                "created_at":        str(r["created_at"]),
                "last_active":       str(r["last_active"]),
                "total_checkpoints": r["total_checkpoints"],
            }
            for r in rows
        ] if rows else [],
    }


@app.get("/db/conversations")
async def list_all_conversations(token_payload: dict = Depends(verify_token)):
    role_id = token_payload.get("role_id")
    if role_id not in {1, 2}:
        raise HTTPException(status_code=403, detail="Access denied.")

    rows = await execute_query(
        """
        SELECT
            thread_id,
            COUNT(*)           AS total_checkpoints,
            MAX(checkpoint_id) AS latest_checkpoint
        FROM checkpoints
        GROUP BY thread_id
        ORDER BY latest_checkpoint DESC
        LIMIT 10
        """
    )
    return {
        "total_conversations": len(rows),
        "conversations": [
            {
                "thread_id":         r["thread_id"],
                "total_checkpoints": r["total_checkpoints"],
                "latest_checkpoint": r["latest_checkpoint"],
            }
            for r in rows
        ],
    }


# ── Zoho Key Management ────────────────────────────────────────────────────────

@app.post("/user/zoho-key")
async def save_user_zoho_key(req: ZohoKeyRequest, token_payload: dict = Depends(verify_token)):
    assert_token_matches_emp(token_payload, req.emp_code)
    try:
        await save_zoho_key(req.emp_code, req.zoho_mcp_key)
        await invalidate_zoho_session(req.emp_code)
        return {"status": "success", "emp_code": req.emp_code, "message": "Zoho MCP URL saved."}
    except Exception as e:
        logger.error(f"[zoho-key] save failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to save Zoho key.")


@app.get("/user/has-api-key/{emp_code}")
async def check_user_has_zoho_key(emp_code: int, token_payload: dict = Depends(verify_token)):
    assert_token_matches_emp(token_payload, emp_code)
    try:
        has_key = await has_zoho_key(emp_code)
        return {
            "status":            "success",
            "emp_code":          emp_code,
            "has_key":           has_key,
            "connection_status": "connected" if has_key else None,
        }
    except Exception as e:
        logger.error(f"[has-api-key] Error: {e}")
        raise HTTPException(status_code=500, detail="Service unavailable.")


@app.get("/user/zoho-key/{emp_code}")
async def get_user_zoho_key(emp_code: int, token_payload: dict = Depends(verify_token)):
    assert_token_matches_emp(token_payload, emp_code)
    try:
        url = await get_zoho_key(emp_code)
        if not url:
            raise HTTPException(status_code=404, detail="Zoho key not found.")

        parts        = url.rstrip("/").split("/")
        hash_segment = parts[-1] if parts else ""
        preview      = (
            "/".join(parts[:-1]) + "/" + hash_segment[:8] + "***"
            if len(hash_segment) > 8
            else url[:20] + "***"
        )
        return {
            "status":            "success",
            "emp_code":          emp_code,
            "has_key":           True,
            "url_preview":       preview,
            "connection_status": "connected",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[zoho-key] get failed: {e}")
        raise HTTPException(status_code=500, detail="Service unavailable.")


# ── Text to Speech ─────────────────────────────────────────────────────────────

@app.post("/text-to-speech")
async def text_to_speech(
    req:             TextRequest,
    fastapi_request: Request,
    data:            TextToSpeechRequest = Body(...),
    token_payload:   dict = Depends(verify_token),
):
    try:
        assert_token_matches_emp(token_payload, req.emp_code)
        tts_engine = fastapi_request.app.state.tts_engine
        return tts_engine.Text_to_speech_process(data, token_payload)
    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        logger.error(f"[text-to-speech] Error: {e}")
        raise HTTPException(status_code=500, detail="Text-to-speech service error.")

@app.post("/user/feedback")
@limiter.limit("5/minute")
async def submit_feedback(
    req:           FeedbackRequest,
    request:       Request,
    token_payload: dict = Depends(verify_token),
):
    assert_token_matches_emp(token_payload, req.emp_code)

    try:
        await execute_query(
            """
            INSERT INTO user_feedback
                (emp_code, rating, category, comments, created_at)
            VALUES
                (%s, %s, %s, %s, NOW())
            RETURNING id
            """,
            (
                req.emp_code,
                req.rating,
                req.category,
                req.comments,
            ),
        )
    except Exception as e:
        logger.error(f"[feedback] DB insert failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save feedback.")

    logger.info(
        f"[feedback] emp={req.emp_code} rating={req.rating} "
        f"category={req.category}"
    )
    return {"status": "success", "message": "Thank you for your feedback!"}
# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(
        "app:app",
        host               = "0.0.0.0",
        port               = 8000,
        reload             = False,
        lifespan           = "on",
        timeout_keep_alive = 60,
    )