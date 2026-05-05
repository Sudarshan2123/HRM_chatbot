import json
import os
from fastapi.middleware.cors import CORSMiddleware
import urllib

from urllib3 import request
from src.entity._init_ import TextRequest, TextToSpeechRequest
from src.entity.responseHelper import *
from src.entity.DBHelper import *
from src.entity.tokenHelper import *
from src.server.zoho_session import close_all_zoho_sessions, invalidate_zoho_session
from src.pipeline.Login import Login
from src.server.checkpointer import close_checkpointer, init_checkpointer
from src.utils.common import decrypt_credentials
from src.utils.security import check_no_query_params
from src.server.zoho_key_store import save_zoho_key, get_zoho_key, has_zoho_key
from statenode import *
from src.utils.Text_To_Speach import TextToSpeach

os.environ["FASTEMBED_CACHE_PATH"] = "C:/fastembed_models"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_LOCAL_FILES_ONLY"] = "0"
os.environ["HUGGINGFACE_HUB_NO_SYMLINK"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import uuid
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from pydantic import ValidationError
from fastapi import Depends, HTTPException, Request
from src.constants.token import Token          
from graphm import create_intent_driven_agent
from src.server.mcp_loader import init_mcp_session, close_mcp_session

load_dotenv()
_graph = None
_sessions = {} 
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
        app.state.tts_engine = TextToSpeach()   
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
    config    = {"configurable": {"thread_id": thread_id},"recursion_limit": 5}

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
    config = {"configurable": {"thread_id": req.thread_id},"recursion_limit": 5}

    await update_thread_active(req.thread_id)
    print(f"[RESUME] decision='{req.decision}' → sending='{req.decision.strip().lower()}'")
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
    config    = {"configurable": {"thread_id": thread_id},"recursion_limit": 5}

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

@app.post("/text-to-speech")
async def text_to_speech(req: TextRequest, data: TextToSpeechRequest = Body(...),token_payload: dict = Depends(verify_token)):
    try:
        assert_token_matches_emp(token_payload, req.emp_code)
        tts_engine = request.app.state.tts_engine
        response=tts_engine.Text_to_speech_process(data,token_payload)
        return response
                 
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())  
    
    except Exception as e:
        logger.error(f"Error in /text-to-speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── DB Read Endpoints ──────────────────────────────────────

@app.post("/db/conversation")
async def get_conversation_from_db(req: ConversationRequest):
    is_owner = await verify_thread_ownership(req.emp_code, req.thread_id)
    if not is_owner:
        return {
            "thread_id":    req.thread_id,
            "emp_code":     req.emp_code,
            "conversation": []
        }
    result = await read_conversation(req.thread_id, _graph)  # ← pass _graph
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
        limit 10
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
        limit 10
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


@app.get("/user/has-api-key/{emp_code}")
async def check_user_has_zoho_key(emp_code: int):
    try:
        has_key = await has_zoho_key(emp_code)
        return {
            "status":            "success",
            "emp_code":          emp_code,
            "has_key":           has_key,
            "connection_status": "connected" if has_key else None,
        }
    except Exception as e:
        logger.error(f"has-api-key error: {e}")
        return {
            "status":            "error",
            "emp_code":          emp_code,
            "has_key":           False,
            "connection_status": None,
        }


@app.get("/user/zoho-key/{emp_code}")
async def get_user_zoho_key(emp_code: int):
    try:
        url = await get_zoho_key(emp_code)
        if not url:
            raise HTTPException(
                status_code=404,
                detail=f"No Zoho MCP URL found for emp_code {emp_code}"
            )

        parts = url.rstrip("/").split("/")
        hash_segment = parts[-1] if parts else ""
        preview = (
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
            "tool_count":        0,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve Zoho key: {str(e)}")
# ---------- Run ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        lifespan="on",
        timeout_keep_alive=60
    )