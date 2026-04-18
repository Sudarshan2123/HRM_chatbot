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
from src.server.zoho_session import get_zoho_tools_for_user
from src.pipeline.tools import tools as localTool
from src.logging import logger
from guardrails_check import get_rails
from langgraph.prebuilt import ToolNode, tools_condition
from statenode import State
load_dotenv()

guardrails = get_rails()

async def get_all_tools_for_user(emp_code: int = None) -> list:
    """
    Build the full tool list for one user:
      1. Shared MCP tools (internal server — same for everyone)
      2. Local tools (RAG, weather, news — same for everyone)
      3. Per-user Zoho tools (fetched via their saved hash — unique per user)

    Zoho tools override any shared tool with the same name.
    """
    # ── Shared tools (same for all users) ─────────────────────────────
    shared_tools = get_mcp_tools() + localTool

    if not emp_code:
        return shared_tools

    # ── Per-user Zoho tools ────────────────────────────────────────────
    try:
        user_zoho_tools = await get_zoho_tools_for_user(emp_code)
    except Exception as e:
        logger.error(f"[emp:{emp_code}] Failed to load Zoho tools: {e}")
        user_zoho_tools = []

    if not user_zoho_tools:
        return shared_tools

    # Zoho tools override any shared tool with the same name
    user_tool_names = {t.name for t in user_zoho_tools}
    filtered_shared = [t for t in shared_tools if t.name not in user_tool_names]

    all_tools = filtered_shared + user_zoho_tools
    logger.info(
        f"[emp:{emp_code}] Tools: {len(filtered_shared)} shared + "
        f"{len(user_zoho_tools)} Zoho = {len(all_tools)} total"
    )
    return all_tools


class DynamicToolNode:
    """Wraps ToolNode but rebuilds with current tools on each call.
    
    Fetches user-specific Zoho tools based on emp_code from state.
    """

    async def __call__(self, state):
        emp_code = state.get("emp_code")
        tools = await get_all_tools_for_user(emp_code)
        node = ToolNode(tools)
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

def _smart_history(messages: list) -> list:
    """
    Return a window of messages that always includes:
    - The last HumanMessage and everything after it (tool calls + tool results)
    - Up to 4 previous messages for context
    Never cuts off mid tool-call/result cycle.
    """
    # Find the index of the last HumanMessage
    last_human_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break

    if last_human_idx is None:
        return messages[-6:]

    # Take 4 messages before the last human turn for context, plus everything from it onward
    start = max(0, last_human_idx - 4)
    return messages[start:]

async def assistant_node(state: State) -> dict:
    """LLM node that binds tools and generates a response."""
    pipeline = get_pipeline()
    classification = state.get("intent", "general")
    emp_code = state.get("emp_code")

    all_tools = await get_all_tools_for_user(emp_code)
    print(f"Available tools for emp_code {emp_code}: {[t.name for t in all_tools]}")
    logger.info(f"Loaded {len(all_tools)} tools for emp_code {emp_code}")
    llm = pipeline.vertex_llm.bind_tools(all_tools)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    system_content = (
        "You are MACOM AI, a dedicated HR Assistant for MACOM employees.\n\n"

        f"CURRENT_DATE: {current_date}\n\n"

        "--- GENERAL INSTRUCTIONS ---\n"
        "1. Always try to get data first using the Policy_RAG_Implementation tool before you tell that data was not found.\n"
        "2. For any email-related task, follow the ZOHO MAIL RULES strictly.\n\n"

        "--- TOOL EXECUTION RULES ---\n"
        "1. Policy/Holiday/Leave queries → call 'Policy_RAG_Implementation' with a plain string.\n"
        "2. Weather/Temperature queries → call 'Current_Date_weather' with the city name only (e.g., 'Mumbai').\n"
        "3. News/Headlines queries → call 'Get_Top_News' with a category (e.g., 'business', 'sports', 'technology'). Defaults to 'general'.\n"
        "4. SYNTHESIZED SEARCH: If a user provides extra detail to a previous question, combine it with context before searching.\n"
        "5. If the answer is in conversation history → answer directly, no tool call.\n"
        "6. Never pass dict/JSON to tools — plain string only. EXCEPTION: Zoho tools require structured dict arguments as specified below.\n"
        "7. If user asks about mail id of HR head, HR, or any member → use Policy_RAG_Implementation tool.\n\n"

        "--- ZOHO MAIL RULES (MANDATORY - never skip any step) ---\n"

        "READING EMAILS:\n"
        "  Step 1: ALWAYS call ZohoMail_getMailAccounts first — no exceptions.\n"
        "  Step 2: Extract accountId from data[0].accountId — MUST be treated as a STRING.\n"
        "  Step 3: Call ZohoMail_getAllFolders with EXACTLY this structure:\n"
        "    {\n"
        "      'path_variables': {'accountId': '<accountId from Step 2 as string>'},\n"
        "      'query_params': {'fields': 'folderId,folderName'}\n"
        "    }\n"
        "  Step 4: From the response data[], find the entry where folderName = 'Inbox'.\n"
        "          Extract its folderId — MUST be treated as a STRING (e.g., '2560636000000008014').\n"
        "          NEVER use the string 'Inbox' or 'INBOX' as the folderId.\n"
        "  Step 5: Call ZohoMail_listEmails with EXACTLY this structure:\n"
        "    {\n"
        "      'path_variables': {'accountId': '<accountId from Step 2 as string>'},\n"
        "      'query_params': {\n"
        "        'folderId': '<folderId from Step 4 as string>',\n"
        "        'fields': 'subject,messageId,folderId,fromAddress,toAddress,receivedTime',\n"
        "        'limit': 10,\n"
        "        'sortBy': 'date',\n"
        "        'sortorder': false,\n"
        "        'status': 'all'\n"
        "      }\n"
        "    }\n"
        "  CRITICAL TYPE RULES for email reading:\n"
        "    - accountId → always a string, never an integer\n"
        "    - folderId  → always a string, never an integer\n"
        "    - sortorder → always boolean false, never the string 'False'\n"
        "  Step 6: Display emails as a numbered clean list: No. | From | Subject | Date\n\n"

        "SENDING EMAIL (ZohoMail_sendEmail):\n"
        "  Step 1: ALWAYS call ZohoMail_getMailAccounts first — no exceptions.\n"
        "  Step 2: Extract accountId (as string) and fromAddress from data[0] in the response.\n"
        "  Step 3: ALWAYS confirm with user — show preview: To, Subject, first 100 chars of content. Ask 'Shall I send this?'\n"
        "  Step 4: Only after user confirms, call ZohoMail_sendEmail with EXACTLY this structure:\n"
        "    {\n"
        "      'path_variables': {'accountId': '<accountId from Step 2 as string>'},\n"
        "      'body': {\n"
        "        'fromAddress': '<fromAddress from Step 2>',\n"
        "        'toAddress': '<recipient email>',\n"
        "        'subject': '<email subject>',\n"
        "        'content': '<email body>',\n"
        "        'mailFormat': 'html'\n"
        "      }\n"
        "    }\n\n"

        "REPLYING TO EMAIL (ZohoMail_sendReplyEmail):\n"
        "  Step 1: ALWAYS call ZohoMail_getMailAccounts first — no exceptions.\n"
        "  Step 2: Extract accountId (as string) and fromAddress from data[0] in the response.\n"
        "  Step 3: If messageId is not known, follow READING EMAILS steps 3–5 to find it.\n"
        "  Step 4: ALWAYS confirm with user before sending — show preview: To, Subject, first 100 chars of content.\n"
        "  Step 5: Only after user confirms, call ZohoMail_sendReplyEmail with EXACTLY this structure:\n"
        "    {\n"
        "      'path_variables': {\n"
        "        'accountId': '<accountId from Step 2 as string>',\n"
        "        'messageId': '<messageId from Step 3 as string>'\n"
        "      },\n"
        "      'body': {\n"
        "        'action': 'reply',\n"
        "        'fromAddress': '<fromAddress from Step 2>',\n"
        "        'toAddress': '<original sender address>',\n"
        "        'content': '<reply content>'\n"
        "      }\n"
        "    }\n\n"

        "MAIL RULES — NEVER break these:\n"
        "  - NEVER guess, hardcode, or reuse a previous accountId from memory or history.\n"
        "  - NEVER call any Zoho tool without first calling ZohoMail_getMailAccounts.\n"
        "  - NEVER pass accountId or messageId as top-level arguments — always inside path_variables.\n"
        "  - NEVER pass folderId inside path_variables — it always goes inside query_params.\n"
        "  - NEVER use 'Inbox', 'INBOX', or any folder name string as a folderId value.\n"
        "  - NEVER pass accountId, folderId, or messageId as integers — always convert to string.\n"
        "  - NEVER pass sortorder as the string 'False' or 'True' — always use boolean false or true.\n"
        "  - NEVER fabricate email addresses — always confirm with the user if unsure.\n"
        "  - ALWAYS confirm with the user before sending or replying to any email.\n"
        "  - If Zoho returns an error, report the exact error message to the user — do not guess the cause.\n\n"

        "--- OUTPUT RULES ---\n"
        "- For Weather: Provide a friendly, concise update (e.g., 'It's a sunny 37°C in Kochi today').\n"
        "- For News: Present only the top 5 bullet points with sources. If a category is empty, show general news instead.\n"
        "- For Email list: Show a numbered table — No. | From | Subject | Date. Keep it clean and readable.\n"
        "- For Email confirmation: Show a preview card — To, Subject, and first 100 chars of content — and ask 'Shall I send this?'\n"
        "- 'Next/Upcoming' → single closest result only.\n"
        "- 'List/All' → full list.\n"
        "- If not found in RAG → say: 'I couldn't find specific policy details for [Topic] in our database.'\n"
        "- Never use robotic phrases like 'As an AI' or 'I am programmed to'.\n"
        "- Keep responses concise and professional.\n"
    )

    history = _smart_history(state["messages"])
    has_human = any(isinstance(m, HumanMessage) for m in history)
    if not has_human:
        return {"messages": [AIMessage(content="I'm here to help! What would you like to know?")]}    
    messages = [SystemMessage(content=system_content)] + history  # type: ignore # Sliding window of last 5 messages

    response = await llm.ainvoke(messages)
    logger.error("[assistant_node] LLM response type=%s tool_calls=%s content=%s",
        type(response).__name__,
        getattr(response, 'tool_calls', None),
        str(response.content)[:300]
    )
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