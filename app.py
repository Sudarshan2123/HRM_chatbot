import datetime
import sys
import os

from src.pipeline.leave_agent_node import leave_subgraph  
os.environ["FASTEMBED_CACHE_PATH"] = "C:/fastembed_models"
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, START
from langchain_core.language_models import BaseChatModel
from typing import Callable
from nemoguardrails.rails.llm.options import GenerationOptions
# from src.pipeline.leave_agent import run_leave_agent
from src.ColdStart.singleton import get_pipeline
from src.server.mcp_loader import get_mcp_tools
from src.logging import logger
from guardrails_check import get_rails
from langgraph.prebuilt import ToolNode, tools_condition
from statenode import State
load_dotenv()

guardrails = get_rails()

def get_all_tools():
    return get_mcp_tools()


class DynamicToolNode:
    """Wraps ToolNode but rebuilds with current tools on each call."""

    async def __call__(self, state):
        node = ToolNode(get_all_tools())
        return await node.ainvoke(state)


async def Orchestrator(state: State) -> State:

    if not (state.get("messages") and isinstance(state["messages"][-1], HumanMessage)):
        return {
            "intent":    state.get("intent", "general"),
            "responded": False,
        }

    latest_user_msg = state["messages"][-1].content.strip()
    leave_step      = state.get("leave_step")
    active_intent   = state.get("active_intent", "general")

    active_leave_steps = {
        "awaiting_leave_type", "awaiting_dates",
        "awaiting_to_date", "awaiting_remarks", "awaiting_submission"
    }

    # ── Explicit leave trigger phrases ──
    EXPLICIT_LEAVE_TRIGGERS = [
        "apply leave", "apply for leave", "apply the leave",
        "i want to apply", "submit leave", "request leave",
        "book leave", "i need a day off", "need leave",
        "apply casual", "apply sick", "apply earned",
        "raise a leave", "put in leave", "request time off"
    ]

    def is_explicit_leave(msg: str) -> bool:
        msg_lower = msg.lower().strip()
        return any(phrase in msg_lower for phrase in EXPLICIT_LEAVE_TRIGGERS)

    # Case 1: Active interrupt — skip guardrail
    if leave_step in active_leave_steps:
        print(f">>> interrupt pending — skipping guardrail")
        return {"intent": "apply_leave", "responded": False}

    # Case 2: Explicit leave request — skip guardrail entirely
    # User clearly wants to apply — don't let active_intent block it
    if is_explicit_leave(latest_user_msg):
        print(f">>> explicit leave request detected — routing directly to leave_node")
        return {
            "intent":        "apply_leave",
            "active_intent": "apply_leave",
            "responded":     False,
        }

    # Case 3: Run guardrail (embeddings only — fast)
    input_messages = [{"role": "user", "content": latest_user_msg}]
    print(f"Input to Guardrail: {input_messages}")

    res = await guardrails.generate_async(
        messages=input_messages,
        options=GenerationOptions(
            output_vars=True,
            log={
                "activated_rails": False,
                "llm_calls":       False,
                "internal_events": True,
                "colang_history":  False,
            },
            rails=["input", "dialog"],
        ),
    )

    intents = [
        e for e in (res.log.internal_events or [])
        if e.get("type") == "UserIntent"
    ]
    detected_intent = intents[-1].get("intent") if intents else None
    print(f"Detected intent: {detected_intent}")

    output_text = res.response[-1].get("content", "") if res.response else ""

    PASSTHROUGH_MESSAGES = {
        "Passing your request to the assistant...",
        "Passing your request to the leave system..."
    }

    # Case 4: Guardrail blocked
    # If mid-assistant-conversation → don't block, continue to assistant
    if output_text and output_text not in PASSTHROUGH_MESSAGES:
        if active_intent == "Assistant":
            print(">>> guardrail blocked but active_intent=Assistant — continuing to assistant")
            return {
                "intent":        "Assistant",
                "active_intent": "Assistant",
                "responded":     False,
            }
        return {
            "messages":  [AIMessage(content=output_text)],
            "intent":    detected_intent or active_intent,
            "responded": True,
        }

    # Case 5: apply_leave detected but mid-assistant-conversation
    # Only override if NOT an explicit leave request (already handled in Case 2)
    if detected_intent == "apply_leave" and active_intent == "Assistant":
        print(">>> non-explicit apply_leave during assistant conversation — staying in assistant")
        return {
            "intent":        "Assistant",
            "active_intent": "Assistant",
            "responded":     False,
        }

    # Case 6: Resolve final intent
    VAGUE_INTENTS = {"follow_up", "ask off topic", None}

    if leave_step in {"cancelled", "completed", "failed"}:
        previous_active_intent = "general"
    else:
        previous_active_intent = active_intent

    if detected_intent not in VAGUE_INTENTS:
        final_intent      = detected_intent
        new_active_intent = detected_intent
    else:
        final_intent      = previous_active_intent
        new_active_intent = previous_active_intent

    print(f"Final intent: {final_intent} | active_intent: {new_active_intent}")

    return {
        "intent":        final_intent,
        "active_intent": new_active_intent,
        "responded":     False,
    }


def route_after_classification(state: State) -> str:
    print(f">>> routing: responded={state.get('responded')}, intent={state.get('intent')}")

    if state.get("responded", False):
        return END

    leave_step = state.get("leave_step")

    active_leave_steps = {
        "awaiting_leave_type",
        "awaiting_dates",
        "awaiting_to_date",
        "awaiting_remarks",
        "awaiting_submission"
    }


    if leave_step in active_leave_steps:
        return "Leave_Application"

    if leave_step in {"cancelled", "completed", "failed", None}:
        intent = state.get("intent", "general")
        if intent == "apply_leave":
            return "Leave_Application"   # fresh start — enters leave_balance_node
        if intent == "Assistant":
            return "assistant"
        return "assistant"

    return "assistant"



async def assistant_node(state: State) -> dict:
    """LLM node that binds tools and generates a response."""
    pipeline = get_pipeline()
    classification = state.get("intent", "general")

    all_tools = get_all_tools()
    print(f"Available tools: {all_tools}")
    logger.info(all_tools)
    llm = pipeline.vertex_llm.bind_tools(all_tools)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    system_content = (
        "You are MACOM AI, a dedicated HR Assistant for MACOM employees.\n\n"

        f"CURRENT DATE: {current_date}\n\n"

        "--- TOOL EXECUTION RULES ---\n"
        "1. Policy/Holiday/Leave queries → call 'Policy_RAG_Implementation' with a plain string.\n"
        "2. SYNTHESIZED SEARCH: If a user provides extra detail (e.g., 'Kerala', 'Confirmed') "
        "to a previous question, combine it with the previous context before searching. "
        "Example: Search 'Sick leave policy for confirmed employee in Kerala' not just 'Kerala'.\n"
        "3. If the answer is already in conversation history → answer directly, no tool call.\n"
        "4. Never pass dict/JSON to tools — plain string only.\n\n"

        "--- OUTPUT RULES ---\n"
        "- 'Next/Upcoming' → single closest result only.\n"
        "- 'List/All' → full list.\n"
        "- If not found in RAG → say: 'I couldn't find specific policy details for [Topic] in our database.'\n"
        "- Never use robotic phrases like 'As an AI' or 'I am programmed to'.\n"
        "- Keep responses concise and professional."
    )

    history = state["messages"][-5:]
    has_human = any(isinstance(m, HumanMessage) for m in history)
    if not has_human:
        return {"messages": [AIMessage(content="I'm here to help! What would you like to know?")]}    
    messages = [SystemMessage(content=system_content)] + history  # type: ignore # Sliding window of last 5 messages

    response = await llm.ainvoke(messages)
    return {"messages": [response]}





def create_intent_driven_agent(checkpointer=None) -> StateGraph:
    """Create a LangGraph agent with NeMo Guardrails integration.

    Graph structure:
        START -> orchestrator -> route_after_classification
        -> assistant -> tools_condition -> tools -> assistant -> ...
    """

    _leave_subgraph_compiled = leave_subgraph(checkpointer=checkpointer)

    graph = StateGraph(State)

    # Add nodes
    graph.add_node("orchestrator", Orchestrator)
    graph.add_node("assistant", assistant_node)
    graph.add_node("leave_node", _leave_subgraph_compiled)
    graph.add_node("tools", DynamicToolNode())  # Use dynamic tool node

    # Entry point
    graph.add_edge(START, "orchestrator")


    # After orchestrator, route based on intent
    graph.add_conditional_edges(
        "orchestrator",
        route_after_classification,
        {
            "assistant": "assistant",
            # Add more routes here as needed, e.g.:
            "Leave_Application": "leave_node",
            END: END,
        },
    )

    # Tool call loop: assistant -> tools -> assistant
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile(checkpointer=checkpointer)