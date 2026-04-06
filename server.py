import json
import os

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
    checkpointer = MemorySaver()
    _graph = create_intent_driven_agent(checkpointer=checkpointer)
    yield
    await close_mcp_session()


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
    emp_name:  str   = "Ram"
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

                # ── Guardrail / interrupt ──
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

                    elif output.get("__interrupt__"):
                        interrupt_data = output["__interrupt__"][0].value
                        msg = interrupt_data.get("message", "") if isinstance(interrupt_data, dict) else str(interrupt_data)
                        yield f"data: {json.dumps({'type': 'interrupt', 'content': msg})}\n\n"
                        return

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

# ---------- Run ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)