from fastapi import HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from statenode import *

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


async def read_conversation(thread_id: str, graph=None) -> dict:
    config = {"configurable": {"thread_id": thread_id}}

    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    try:
        state = await graph.aget_state(config)
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

