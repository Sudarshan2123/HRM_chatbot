import json
import os
import psycopg
from fastapi.middleware.cors import CORSMiddleware
import urllib
from src.server.zoho_session import close_all_zoho_sessions, invalidate_zoho_session
from src.pipeline.Login import Login
from src.server.checkpointer import close_checkpointer, init_checkpointer
from src.utils.common import decrypt_credentials
from src.utils.security import check_no_query_params
from src.server.zoho_key_store import save_zoho_key, get_zoho_key, has_zoho_key
from src.utils.database import init_db_pool, close_db_pool, execute_query, execute_query_single

os.environ["FASTEMBED_CACHE_PATH"] = "C:/fastembed_models"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_LOCAL_FILES_ONLY"] = "0"
os.environ["HUGGINGFACE_HUB_NO_SYMLINK"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import uuid
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from pydantic import BaseModel, ValidationError
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.constants.token import Token          
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from app import create_intent_driven_agent
from src.server.mcp_loader import init_mcp_session, close_mcp_session

load_dotenv()
security = HTTPBearer(auto_error=False)
_token_service = Token()
_graph = None

def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Validates the Bearer JWT from the Authorization header.
    Returns the decoded payload (includes userName, session_id, exp).
    Raises 401 if missing, expired, or invalid.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing or malformed."
        )

    token = credentials.credentials

    # Strip "Bearer " prefix if the frontend accidentally double-wraps it
    if token.lower().startswith("bearer "):
        token = token[7:]

    payload = _token_service.validate_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Token is expired or invalid."
        )

    return payload

def assert_token_matches_emp(token_payload: dict, emp_code: int):
    """
    Compares token's userName against the emp_code in the request body.
    Strips whitespace and normalises to string to avoid type/format mismatches.
    Raises 403 immediately if they don't match.
    """
    token_user   = str(token_payload.get("userName", "")).strip()
    request_user = str(emp_code).strip()

    if not token_user:
        raise HTTPException(
            status_code=403,
            detail="Token does not contain a userName claim."
        )

    if token_user != request_user:
        raise HTTPException(
            status_code=403,
            detail=f"Token identity '{token_user}' does not match emp_code '{request_user}'."
        )
# ── Lifespan ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    try:
        print("Starting Lifespan: Initializing Database Pool...")
        await init_db_pool()

        print("Starting Lifespan: Initializing MCP Sessions...")
        await init_mcp_session()

        print("Initializing Checkpointer...")
        checkpointer = await init_checkpointer()

        print("Creating Agentic Graph...")
        _graph = create_intent_driven_agent(checkpointer=checkpointer)

        yield
    finally:
        print("Shutting down: Closing sessions...")
        await close_mcp_session()
        await close_all_zoho_sessions()
        await close_checkpointer()
        await close_db_pool()


# ── App ────────────────────────────────────────────────────

app = FastAPI(
    title="HRMS AI Assistant",
    description="LangGraph agent with NeMo Guardrails for HRMS queries",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS must be added before including routers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Authorization"],
)


# ── Request / Response Models ──────────────────────────────

class ChatRequest(BaseModel):
    message:   str
    thread_id: Optional[str] = None
    emp_code:  int  = 100606
    firm_id:   int  = 3
    emp_name:  str  = "Anika"
    role_id:   int  = 5
    firm_name: str  = "Macom"


class ChatResponse(BaseModel):
    thread_id: str
    intent:    str
    response:  str
    status:    str           = "completed"
    action:    Optional[str] = None
    feedback_url: Optional[str] = None


class ResumeRequest(BaseModel):
    thread_id: str
    decision:  str
    emp_code:  int = 1203


class ConversationRequest(BaseModel):
    emp_code:  int
    thread_id: str


class UserConversationsRequest(BaseModel):
    emp_code: int


class EncryptedLoginData(BaseModel):
    username: str
    password: str

    class Config:
        extra = 'forbid'

class ZohoKeyRequest(BaseModel):
    emp_code: int
    zoho_mcp_key: str


# ── DB Helpers ─────────────────────────────────────────────

async def save_user_thread(emp_code: int, thread_id: str):
    await execute_query(
        """
        INSERT INTO user_conversations (emp_code, thread_id)
        VALUES (%s, %s)
        ON CONFLICT (thread_id) DO UPDATE
            SET last_active = NOW()
        """,
        (emp_code, thread_id),
        fetch=False
    )


async def update_thread_active(thread_id: str):
    await execute_query(
        """
        UPDATE user_conversations
        SET last_active = NOW()
        WHERE thread_id = %s
        """,
        (thread_id,),
        fetch=False
    )


async def verify_thread_ownership(emp_code: int, thread_id: str) -> bool:
    result = await execute_query_single(
        """
        SELECT thread_id FROM user_conversations
        WHERE emp_code = %s AND thread_id = %s
        """,
        (emp_code, thread_id)
    )
    return result is not None


# ── Response Helpers ───────────────────────────────────────

def extract_last_message(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "No response"
    content = messages[-1].content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip() or "No response"
    elif isinstance(content, str):
        return content
    return str(content)


def handle_result(result: dict, thread_id: str) -> ChatResponse:
    intent = result.get("intent", "general")

    if result.get("__interrupt__"):
        interrupt_data = result["__interrupt__"][0].value
        return ChatResponse(
            thread_id=thread_id,
            intent=intent,
            response=interrupt_data.get("message", "Awaiting your input."),
            status="waiting_for_input",
            action=interrupt_data.get("action")
        )

    return ChatResponse(
        thread_id=thread_id,
        intent=intent,
        response=extract_last_message(result),
        status="completed"
    )


async def read_conversation(thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = await _graph.aget_state(config)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Thread not found: {e}")

    if not state or not state.values:
        raise HTTPException(status_code=404, detail="No data for this thread")

    messages = state.values.get("messages", [])
    conversation = []

    for msg in messages:
        if isinstance(msg, HumanMessage) or msg.type == "human":
            role = "user"
        elif isinstance(msg, AIMessage) or msg.type == "ai":
            role = "assistant"
        else:
            continue

        content = msg.content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )

        if content and role in ("user", "assistant"):
            conversation.append({"role": role, "content": content})

    return {
        "thread_id":     thread_id,
        "intent":        state.values.get("intent", "unknown"),
        "message_count": len(conversation),
        "conversation":  conversation
    }


# ── Endpoints ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/login")
async def login(request: Request, data: EncryptedLoginData = Body(...)):
    try:
        check_no_query_params(request)

        username = urllib.parse.unquote(data.username)
        password = data.password
        encrypted = {"username": username, "password": password}
        decrypted = decrypt_credentials(encrypted)

        Login_chat = Login()
        if username and password:
            result = Login_chat.login_user(decrypted['username'], decrypted['password'])
            user_Status = result.get("user_Status")

            if result.get("status") == "success":
                access_token = result.get("access_token")
                response = JSONResponse(content={
                    'status': 'success',
                    'user_Status': user_Status
                })
                response.headers['Authorization'] = f"Bearer {access_token}"
                return response
            else:
                return JSONResponse(result, status_code=400)
        else:
            raise HTTPException(
                status_code=400,
                detail="Employee code and password required for login"
            )
    except ValidationError as e:
        print(e)
        raise HTTPException(status_code=422, detail=e.errors())


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest,token_payload: dict = Depends(verify_token)):
    # Generate thread_id if not provided (new chat)
    is_new_thread = not req.thread_id
    thread_id = req.thread_id or str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    # Always save/touch the thread BEFORE invoking so it exists in DB immediately
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Touch last_active after successful response
    await update_thread_active(thread_id)

    response = handle_result(result, thread_id)
    # Tell the frontend whether this was a newly created thread
    # so it knows to refresh the sidebar
    response_dict = response.dict()
    response_dict["is_new_thread"] = is_new_thread
    return JSONResponse(content=response_dict)


@app.post("/chat/resume", response_model=ChatResponse)
async def resume(req: ResumeRequest,token_payload: dict = Depends(verify_token)):
    config = {"configurable": {"thread_id": req.thread_id}}

    await update_thread_active(req.thread_id)
    token_user = str(token_payload.get("userName", ""))
    if token_user != str(req.emp_code):
        raise HTTPException(
            status_code=403,
            detail="Token identity does not match emp_code in request."
        )
    
    try:
        result = await _graph.ainvoke(
            Command(resume=req.decision.strip().lower()),
            config=config,
        )
    except PermissionError as e:
        return ChatResponse(
            thread_id=req.thread_id, intent="leave",
            response=str(e), status="completed"
        )
    except ValueError as e:
        return ChatResponse(
            thread_id=req.thread_id, intent="leave",
            response=str(e), status="completed"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return handle_result(result, req.thread_id)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, token_payload: dict = Depends(verify_token)):
    assert_token_matches_emp(token_payload, req.emp_code)

    thread_id = req.thread_id or str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    # ── Ownership check ───────────────────────────────────────────────
    # A thread is "new" if it has no row in user_conversations yet.
    # We cannot rely on is_new_thread = not req.thread_id because the
    # frontend always generates and sends a thread_id from page load.
    # Instead: if the thread already exists in DB, verify it belongs to
    # this user. If it doesn't exist yet, it's a genuinely new thread —
    # skip the check and let save_user_thread establish ownership below.
    thread_exists = await verify_thread_ownership(req.emp_code, thread_id)
    thread_in_db  = await execute_query_single(
        "SELECT thread_id FROM user_conversations WHERE thread_id = %s",
        (thread_id,)
    )

    if thread_in_db and not thread_exists:
        # Thread exists in DB but belongs to a different user
        raise HTTPException(
            status_code=403,
            detail=f"Thread {thread_id} does not belong to emp_code {req.emp_code}."
        )

    is_new_thread = not thread_in_db   # accurate flag based on DB state

    # ── Save thread AFTER all checks pass ─────────────────────────────
    await save_user_thread(req.emp_code, thread_id)

    async def event_stream():
        yield f"data: {json.dumps({'type': 'thread_init', 'thread_id': thread_id, 'is_new_thread': is_new_thread})}\n\n"

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
                config=config,
                version="v2",
            ):
                kind = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")

                if kind == "on_chat_model_stream" and node == "assistant":
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        content = chunk.content
                        if isinstance(content, list):
                            content = "".join(
                                block.get("text", "")
                                for block in content
                                if isinstance(block, dict) and block.get("type") == "text"
                            )
                        if content:
                            yield f"data: {json.dumps({'type': 'token', 'content': content, 'thread_id': thread_id})}\n\n"

                elif kind == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': event.get('name', '')})}\n\n"

                elif kind == "on_tool_end":
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': event.get('name', '')})}\n\n"

                elif kind == "on_chain_stream" and node == "":
                    chunk = event["data"].get("chunk", {})
                    if "__interrupt__" in chunk:
                        interrupt_data = chunk["__interrupt__"][0].value
                        msg = interrupt_data.get("message", "") if isinstance(interrupt_data, dict) else str(interrupt_data)
                        yield f"data: {json.dumps({'type': 'interrupt', 'content': msg, 'thread_id': thread_id})}\n\n"

                elif kind == "on_chain_end":
                    output = event["data"].get("output")
                    if not isinstance(output, dict):
                        continue
                    if node == "orchestrator" and output.get("responded"):
                        messages = output.get("messages", [])
                        if messages:
                            content = messages[-1].content
                            if isinstance(content, list):
                                content = "".join(
                                    block.get("text", "")
                                    for block in content
                                    if isinstance(block, dict) and block.get("type") == "text"
                                )
                            if content:
                                yield f"data: {json.dumps({'type': 'guardrail', 'content': content})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        finally:
            await update_thread_active(thread_id)
            yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── DB Read Endpoints ──────────────────────────────────────

@app.post("/db/conversation")
async def get_conversation_from_db(req: ConversationRequest):
    is_owner = await verify_thread_ownership(req.emp_code, req.thread_id)
    if not is_owner:
        raise HTTPException(
            status_code=403,
            detail=f"Thread {req.thread_id} does not belong to emp_code {req.emp_code}"
        )
    result = await read_conversation(req.thread_id)
    result["emp_code"] = req.emp_code
    return result


@app.post("/db/conversations/user")
async def get_user_conversations(req: UserConversationsRequest):
    rows = await execute_query(
        """
        SELECT
            uc.thread_id,
            uc.created_at,
            uc.last_active,
            COUNT(c.checkpoint_id) AS total_checkpoints
        FROM user_conversations uc
        LEFT JOIN checkpoints c
            ON uc.thread_id = c.thread_id
        WHERE uc.emp_code = %s
        GROUP BY uc.thread_id, uc.created_at, uc.last_active
        ORDER BY uc.last_active DESC
        """,
        (req.emp_code,)
    )

    # FIX: Return empty list instead of 404 so sidebar renders correctly
    # for new users or when no conversations exist yet
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
        ] if rows else []
    }


@app.get("/db/conversations")
async def list_all_conversations():
    rows = await execute_query(
        """
        SELECT
            thread_id,
            COUNT(*)           AS total_checkpoints,
            MAX(checkpoint_id) AS latest_checkpoint
        FROM checkpoints
        GROUP BY thread_id
        ORDER BY latest_checkpoint DESC
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
        ]
    }


# ── Zoho Key Management Endpoints ──────────────────────────
@app.post("/user/zoho-key")
async def save_user_zoho_key(req: ZohoKeyRequest):
    try:
        await save_zoho_key(req.emp_code, req.zoho_mcp_key)
        await invalidate_zoho_session(req.emp_code)
        return {
            "status":   "success",
            "emp_code": req.emp_code,
            "message":  "Zoho MCP URL saved. Connecting in background..."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save Zoho URL: {str(e)}")


@app.get("/user/zoho-key/{emp_code}")
async def get_user_zoho_key(emp_code: int):
    try:
        url = await get_zoho_key(emp_code)
        if not url:
            raise HTTPException(
                status_code=404,
                detail=f"No Zoho MCP URL found for emp_code {emp_code}"
            )

        # Show domain + first 8 chars of the hash segment — never expose full URL
        # e.g. "https://mail-sending-replies-60069513271.zohomcp.in/mcp/abc123de***"
        parts = url.rstrip("/").split("/")
        hash_segment = parts[-1] if parts else ""
        preview = (
            "/".join(parts[:-1]) + "/" + hash_segment[:8] + "***"
            if len(hash_segment) > 8
            else url[:20] + "***"
        )

        # Also report live session status
        from src.server.zoho_session import _sessions
        session = _sessions.get(emp_code)
        if session is None:
            connection_status = "not_started"
        elif session.ready.is_set():
            connection_status = "connected"
            tool_count = len(session.tools)
        elif session.task and not session.task.done():
            connection_status = "connecting"
            tool_count = 0
        else:
            connection_status = "disconnected"
            tool_count = 0

        return {
            "status":            "success",
            "emp_code":          emp_code,
            "has_key":           True,
            "url_preview":       preview,
            "connection_status": connection_status,
            "tool_count":        len(session.tools) if session and session.ready.is_set() else 0,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve Zoho key: {str(e)}")


@app.get("/user/has-api-key/{emp_code}")
async def check_user_has_zoho_key(emp_code: int):
    try:
        has_key = await has_zoho_key(emp_code)

        # If key exists, also report connection status
        status_detail = None
        if has_key:
            from src.server.zoho_session import _sessions
            session = _sessions.get(emp_code)
            if session is None:
                status_detail = "not_started"
            elif session.ready.is_set():
                status_detail = "connected"
            elif session.task and not session.task.done():
                status_detail = "connecting"
            else:
                status_detail = "disconnected"

        return {
            "status":            "success",
            "emp_code":          emp_code,
            "has_key":           has_key,
            "connection_status": status_detail,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check Zoho key: {str(e)}")
# ---------- Run ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        lifespan="on",
        timeout_keep_alive=60
    )