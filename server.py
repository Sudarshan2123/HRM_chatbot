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
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from app import create_intent_driven_agent
from src.server.mcp_loader import init_mcp_session, close_mcp_session
from test_endpoint import router as ui_router  # ← import here, before app

load_dotenv()

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    await init_mcp_session()
    checkpointer = await init_checkpointer()
    _graph = create_intent_driven_agent(checkpointer=checkpointer)
    yield
    await close_mcp_session()
    await close_checkpointer()


# ── App created first ──
app = FastAPI(
    title="HRMS AI Assistant",
    description="LangGraph agent with NeMo Guardrails for HRMS queries",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Router registered AFTER app is created ──
app.include_router(ui_router)


# ---------- Request / Response Models ----------

class ChatRequest(BaseModel):
    message:   str
    thread_id: Optional[str] = None
    emp_code:  int   = 1203
    firm_id:   int   = 3
    emp_name:  str   = "Anika"
    role_id:   int   = 5
    firm_name: str   = "Macom"


class ChatResponse(BaseModel):
    thread_id: str
    intent: str
    response: str
    status: str = "completed"
    action: Optional[str] = None


class ResumeRequest(BaseModel):
    thread_id: str
    decision: str


# ---------- Helper ----------

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


# ---------- Endpoints ----------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await _graph.ainvoke(
            {
                "messages": [HumanMessage(content=req.message)],
                "emp_code": req.emp_code,
                "firm_id":  req.firm_id,
                "emp_name": req.emp_name,
                "role_id":  req.role_id,
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

    try:
        result = await _graph.ainvoke(
            Command(resume=req.decision.strip().lower()),
            config=config,
        )
    except PermissionError as e:
        return ChatResponse(thread_id=req.thread_id, intent="leave", response=str(e), status="completed")
    except ValueError as e:
        return ChatResponse(thread_id=req.thread_id, intent="leave", response=str(e), status="completed")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return handle_result(result, req.thread_id)

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def event_stream():
        try:
            async for event in _graph.astream_events(
                {
                    "messages": [HumanMessage(content=req.message)],
                    "emp_code": req.emp_code,
                    "firm_id":  req.firm_id,
                    "emp_name": req.emp_name,
                    "role_id":  req.role_id,
                    "firm_name": req.firm_name,
                },
                config=config,
                version="v2",
            ):
                kind = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")

                # ── Token-by-token from assistant LLM only ──
                if kind == "on_chat_model_stream" and node == "assistant":
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        content = chunk.content

                        # Vertex AI returns a list of dicts, other LLMs return a string
                        if isinstance(content, list):
                            content = "".join(
                                block.get("text", "")
                                for block in content
                                if isinstance(block, dict) and block.get("type") == "text"
                            )

                        if content:
                            yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

                # ── Tool call signals ──
                elif kind == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': event.get('name', '')})}\n\n"

                elif kind == "on_tool_end":
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': event.get('name', '')})}\n\n"

                # ── Interrupt (emitted as on_chain_stream with __interrupt__) ──
                elif kind == "on_chain_stream" and node == "":
                    chunk = event["data"].get("chunk", {})
                    if "__interrupt__" in chunk:
                        interrupt_data = chunk["__interrupt__"][0].value
                        msg = interrupt_data.get("message", "") if isinstance(interrupt_data, dict) else str(interrupt_data)
                        yield f"data: {json.dumps({'type': 'interrupt', 'content': msg, 'thread_id': thread_id})}\n\n"
                        return

                # ── Guardrail ──
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

@app.get("/db/conversation/{thread_id}")
async def get_conversation_from_db(thread_id: str):
    """Read conversation directly from postgres via checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # This reads from postgres automatically — no raw SQL needed
        state = await _graph.aget_state(config)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Thread not found: {e}")

    if not state or not state.values:
        raise HTTPException(status_code=404, detail="No data for this thread")

    messages = state.values.get("messages", [])

    conversation = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = msg.type  # tool, system etc

        content = msg.content
        if isinstance(content, list):
            # Vertex AI returns list of dicts
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )

        if content and role in ("user", "assistant"):  # skip tool/system messages
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

@app.get("/db/conversations")
async def list_all_conversations():
    """List all thread IDs from postgres."""
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
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)