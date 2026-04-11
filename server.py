import json
import os
import psycopg

from src.server.checkpointer import close_checkpointer, init_checkpointer

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
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from pydantic import BaseModel

from app import create_intent_driven_agent
from src.server.mcp_loader import init_mcp_session, close_mcp_session
from test_endpoint import router as ui_router

load_dotenv()

_graph = None


# ---------- Lifespan ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    try:
        # Increase internal timeout awareness or just log the start
        print("Starting Lifespan: Initializing MCP Sessions...")
        
        # This now handles multiple servers internally based on your new loader
        await init_mcp_session() 
        
        print("Initializing Checkpointer...")
        checkpointer = await init_checkpointer()
        
        print("Creating Agentic Graph...")
        _graph = create_intent_driven_agent(checkpointer=checkpointer)
        
        yield
    finally:
        # Ensure cleanup happens even if startup fails halfway
        print("Shutting down: Closing sessions...")
        await close_mcp_session()
        await close_checkpointer()



# ---------- App ----------

app = FastAPI(
    title="HRMS AI Assistant",
    description="LangGraph agent with NeMo Guardrails for HRMS queries",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ui_router)


# ---------- Request / Response Models ----------

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
    emp_code:  int  = 1203  # needed to update last_active


class ConversationRequest(BaseModel):
    """
    Fetch a single conversation.
    emp_code fetched from request body — same pattern as ChatRequest.
    """
    emp_code:  int
    thread_id: str


class UserConversationsRequest(BaseModel):
    """
    List all threads for a user.
    emp_code fetched from request body — same pattern as ChatRequest.
    """
    emp_code: int


# ---------- DB Helpers ----------

async def save_user_thread(emp_code: int, thread_id: str):
    """Insert or update user → thread mapping."""
    async with await psycopg.AsyncConnection.connect(os.getenv("DB_URI")) as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                INSERT INTO user_conversations (emp_code, thread_id)
                VALUES (%s, %s)
                ON CONFLICT (thread_id) DO UPDATE
                    SET last_active = NOW()
            """, (emp_code, thread_id))
        await conn.commit()


async def update_thread_active(thread_id: str):
    """Update last_active timestamp for a thread."""
    async with await psycopg.AsyncConnection.connect(os.getenv("DB_URI")) as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                UPDATE user_conversations
                SET last_active = NOW()
                WHERE thread_id = %s
            """, (thread_id,))
        await conn.commit()


async def verify_thread_ownership(emp_code: int, thread_id: str) -> bool:
    """Check if a thread belongs to a specific emp_code."""
    async with await psycopg.AsyncConnection.connect(os.getenv("DB_URI")) as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT thread_id FROM user_conversations
                WHERE emp_code = %s AND thread_id = %s
            """, (emp_code, thread_id))
            row = await cur.fetchone()
    return row is not None


# ---------- Response Helpers ----------

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
    """Read and deserialize full conversation from checkpointer."""
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
            conversation.append({
                "role":    role,
                "content": content
            })

    return {
        "thread_id":     thread_id,
        "intent":        state.values.get("intent", "unknown"),
        "message_count": len(conversation),
        "conversation":  conversation
    }


# ---------- Chat Endpoints ----------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    # emp_code fetched from ChatRequest → saved to user_conversations
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

    return handle_result(result, thread_id)


@app.post("/chat/resume", response_model=ChatResponse)
async def resume(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    # emp_code fetched from ResumeRequest → update last_active
    await update_thread_active(req.thread_id)

    try:
        result = await _graph.ainvoke(
            Command(resume=req.decision.strip().lower()),
            config=config,
        )
    except PermissionError as e:
        return ChatResponse(
            thread_id=req.thread_id,
            intent="leave",
            response=str(e),
            status="completed"
        )
    except ValueError as e:
        return ChatResponse(
            thread_id=req.thread_id,
            intent="leave",
            response=str(e),
            status="completed"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return handle_result(result, req.thread_id)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    # emp_code fetched from ChatRequest → saved to user_conversations
    await save_user_thread(req.emp_code, thread_id)

    async def event_stream():
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

                # Token-by-token from assistant LLM only
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

                # Tool call signals
                elif kind == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': event.get('name', '')})}\n\n"

                elif kind == "on_tool_end":
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': event.get('name', '')})}\n\n"

                # Interrupt
                elif kind == "on_chain_stream" and node == "":
                    chunk = event["data"].get("chunk", {})
                    if "__interrupt__" in chunk:
                        interrupt_data = chunk["__interrupt__"][0].value
                        msg = interrupt_data.get("message", "") if isinstance(interrupt_data, dict) else str(interrupt_data)
                        yield f"data: {json.dumps({'type': 'interrupt', 'content': msg, 'thread_id': thread_id})}\n\n"
                        return

                # Guardrail
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

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ---------- DB Read Endpoints ----------

@app.post("/db/conversation")
async def get_conversation_from_db(req: ConversationRequest):
    """
    Fetch full conversation for a thread.
    emp_code comes from request body — same pattern as ChatRequest.

    Request body:
        { "emp_code": 1203, "thread_id": "abc-123" }
    """
    # emp_code from request body used to verify ownership
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
    """
    List all conversation threads for a user.
    emp_code comes from request body — same pattern as ChatRequest.

    Request body:
        { "emp_code": 1203 }
    """
    async with await psycopg.AsyncConnection.connect(os.getenv("DB_URI")) as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
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
            """, (req.emp_code,))
            rows = await cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No conversations found for emp_code {req.emp_code}"
        )

    return {
        "emp_code":            req.emp_code,
        "total_conversations": len(rows),
        "conversations": [
            {
                "thread_id":         r[0],
                "created_at":        str(r[1]),
                "last_active":       str(r[2]),
                "total_checkpoints": r[3],
            }
            for r in rows
        ]
    }


@app.get("/db/conversations")
async def list_all_conversations():
    """List all conversations stored in DB (admin use)."""
    async with await psycopg.AsyncConnection.connect(os.getenv("DB_URI")) as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT
                    thread_id,
                    COUNT(*)           AS total_checkpoints,
                    MAX(checkpoint_id) AS latest_checkpoint
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY latest_checkpoint DESC
            """)
            rows = await cur.fetchall()

    return {
        "total_conversations": len(rows),
        "conversations": [
            {
                "thread_id":         r[0],
                "total_checkpoints": r[1],
                "latest_checkpoint": r[2],
            }
            for r in rows
        ]
    }


# ---------- Run ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        lifespan="on",
        timeout_keep_alive=60 # Adjust based on network latency
    )