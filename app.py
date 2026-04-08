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
    new_state = {}

    if state.get("messages") and isinstance(state["messages"][-1], HumanMessage):
        # input_messages = [
        #     {"role": "user", "content": state["messages"][-1].content}
        # ]
        history = state["messages"][-3:]
        input_messages = [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content
            }
            for m in history
            if isinstance(m, (HumanMessage, AIMessage)) and m.content
        ]
        print(f"Input to Guardrail: {input_messages}")
        res = await guardrails.generate_async(
            messages=input_messages,
            options=GenerationOptions(
                output_vars=True,
                log={
                    "activated_rails": False,
                    "llm_calls": False,
                    "internal_events": True,
                    "colang_history": False,
                },
                rails=["input", "dialog"],
            ),
        )

        print(f"Output from Guardrail: {res.response}")
        print(f"All internal events: {res.log.internal_events}")

        # ← Fix: correct event type is "UserIntent" not "intent_user_message"
        intents = [
            e for e in (res.log.internal_events or [])
            if e.get("type") == "UserIntent"
        ]
        print(f"Intents from Guardrail: {intents}")

        intent = intents[-1].get("intent") if intents else state.get("intent", "general")
        new_state["intent"] = intent
        print(f"Detected intent: {intent}")

        output = res.response[-1].get("content", "") if res.response else ""

        PASSTHROUGH_MESSAGES = {
            "Passing your request to the assistant...",
            "Passing your request to the leave system..."
            }

        if output and output not in PASSTHROUGH_MESSAGES:
            # Guardrails handled it directly (greeting, block, etc.)
            return {
                "messages": [AIMessage(content=output)],
                "intent": intent,
                "responded": True,
            }

        # Needs assistant node
        return {
            "intent": intent,
            "responded": False,
        }

    return {
        "intent": state.get("intent", "general"),
        "responded": False,
    }


def route_after_classification(state: State) -> str:
    print(f">>> routing: responded={state.get('responded')}, intent={state.get('intent')}", file=sys.stderr, flush=True)

    if state.get("responded", False):
        return END

    classification = state.get("intent", "general")

    if isinstance(classification, str):
        if classification == "apply_leave":
            return "Leave_Application"  # or "leave_node" when ready
        if classification == "Assistant":
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
        "You are MACOM AI, a helpful assistant for MACOM employees.\n\n"

        f"INTENT: {classification}\n"
        f"TODAY: {current_date}\n\n"

        "TOOLS:\n"
        "- Policy/holiday/leave queries → call 'Policy_RAG_Implementation' with plain string.\n"
        "- Include TODAY's date in query for temporal questions. Example: 'Upcoming holidays after 2026-04-07?'\n"
        "- If answer already in history → answer directly, no tool call.\n"
        "- Never pass dict/JSON to tools.\n\n"

        "OUTPUT:\n"
        "- 'next/upcoming' → single closest result only.\n"
        "- 'list/all' → full list.\n"
        "- Policy answers: use tool context only. If not found, say so.\n"
        "- Emails/date queries: answer directly, no tool needed.\n"
        "- Never add unprompted refusals.\n"
        "- Dont provide the whole information at once first provide short and ask if require detail information it should be in yes or some options but itb should be contian more words of policy related nothing else"
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