from fastapi import HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from src.logging import logger
from statenode import *


# ── Content Extractor ──────────────────────────────────────────────────────────

def _extract_content(content) -> str:
    """
    Normalize message content.
    Handles both plain string and list-of-blocks format (Vertex AI).
    """
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return str(content) if content else ""


def extract_last_message(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "No response"

    # Walk backwards — skip empty messages (tool calls, empty AIMessages)
    for msg in reversed(messages):
        content = _extract_content(msg.content)
        if content and content.strip():
            return content.strip()

    return "No response"


# ── Result Handler ─────────────────────────────────────────────────────────────

def handle_result(result: dict, thread_id: str) -> ChatResponse:
    """
    Converts LangGraph result into a ChatResponse.
    Handles 3 cases:
      1. Interrupt waiting for user input
      2. Normal completed response
      3. Empty/failed response
    """
    intent = result.get("intent", "general")

    # ── Case 1: Graph interrupted — waiting for user input ────────────────────
    if result.get("__interrupt__"):
        interrupt_val = result["__interrupt__"]

        # Handle both list and direct value formats
        if isinstance(interrupt_val, list) and interrupt_val:
            interrupt_data = interrupt_val[0].value
        else:
            interrupt_data = interrupt_val

        # Normalize interrupt_data — could be dict or plain string
        if isinstance(interrupt_data, dict):
            message = interrupt_data.get("message", "Awaiting your input.")
            action  = interrupt_data.get("action")
            options = interrupt_data.get("options")     # for frontend button rendering
            values  = interrupt_data.get("values")      # for frontend button values
        else:
            message = str(interrupt_data)
            action  = None
            options = None
            values  = None

        return ChatResponse(
            thread_id = thread_id,
            intent    = intent,
            response  = message,
            status    = "waiting_for_input",
            action    = action,
            # Pass options/values if ChatResponse model supports them
            # options = options,
            # values  = values,
        )

    # ── Case 2: Normal completed response ─────────────────────────────────────
    response_text = extract_last_message(result)

    if not response_text or response_text == "No response":
        logger.warning(f"[handle_result] Empty response for thread={thread_id}")

    return ChatResponse(
        thread_id = thread_id,
        intent    = intent,
        response  = response_text,
        status    = "completed",
    )


# ── Conversation Reader ────────────────────────────────────────────────────────

async def read_conversation(thread_id: str, graph=None) -> dict:
    """
    Reads conversation history from LangGraph checkpointer.
    Only returns human and AI messages — skips tool messages.
    """
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized.")

    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = await graph.aget_state(config)
    except Exception as e:
        logger.error(f"[read_conversation] aget_state failed: {e}")
        raise HTTPException(status_code=404, detail="Thread not found.")

    if not state or not state.values:
        raise HTTPException(status_code=404, detail="No data found for this thread.")

    messages     = state.values.get("messages", [])
    conversation = []

    for msg in messages:
        # Determine role
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            role = "user"
        elif isinstance(msg, AIMessage) or getattr(msg, "type", None) == "ai":
            role = "assistant"
        else:
            continue   # skip ToolMessage, SystemMessage etc.

        content = _extract_content(msg.content)

        # Skip empty messages and tool-call-only AIMessages
        if not content or not content.strip():
            continue

        conversation.append({"role": role, "content": content.strip()})

    return {
        "thread_id":     thread_id,
        "intent":        state.values.get("intent", "unknown"),
        "message_count": len(conversation),
        "conversation":  conversation,
    }