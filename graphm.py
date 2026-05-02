import datetime
import json
import os

from src.pipeline.leave_agent_node import leave_subgraph  
os.environ["FASTEMBED_CACHE_PATH"] = "C:/fastembed_models"
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, START
from langchain_core.language_models import BaseChatModel
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
    async def __call__(self, state: State) -> dict:
        emp_code = state.get("emp_code")
        tools    = await get_all_tools_for_user(emp_code)
        node     = ToolNode(tools)
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

from enum import Enum

class _Context(str, Enum):
    POLICY  = "policy"
    WEATHER = "weather"
    NEWS    = "news"
    MAIL_READ  = "mail_read"
    MAIL_SEND  = "mail_send"
    MAIL_REPLY = "mail_reply"
    GENERAL = "general"

def _name_from_email(email: str) -> str:
    """sudarshan.patil@mactech.net.in → Sudarshan Patil"""
    try:
        local = email.split("@")[0]
        return " ".join(p.capitalize() for p in local.split("."))
    except Exception:
        return email
    

def _detect_context(messages: list, state: dict = None) -> set[_Context]:
    last_human = next(
        (m.content.lower() for m in reversed(messages) if isinstance(m, HumanMessage)),
        ""
    )

    # ── Look back up to 6 messages for mail trigger ──────────────────
    # Handles cases where "write a mail to X" was said earlier in the conversation
    recent_humans = [
        m.content.lower() for m in messages[-6:]
        if isinstance(m, HumanMessage)
    ]
    recent_combined = " ".join(recent_humans)

    contexts = set()

    cached_account_id = (state or {}).get("zoho_account_id")

    REPLY_TRIGGERS = ["reply", "respond to", "answer the mail", "answer the email"]
    SEND_TRIGGERS  = [
        "send mail", "send email", "compose",
        "write mail", "write email",
        "write a mail", "write a email",
        "draft mail", "draft email",
        "draft a mail", "draft a email",
        "mail to", "email to",
        "shoot a mail", "shoot an email",
    ]
    READ_TRIGGERS = [
        "inbox", "read mail", "read email",
        "show mail", "show email",
        "fetch mail", "fetch email",
        "check mail", "check email",
    ]

    # Check last message first, then fall back to recent history
    if any(w in last_human for w in REPLY_TRIGGERS):
        contexts.add(_Context.MAIL_REPLY)

    elif any(w in last_human for w in SEND_TRIGGERS):
        contexts.add(_Context.MAIL_SEND)

    elif any(w in last_human for w in READ_TRIGGERS):
        contexts.add(_Context.MAIL_READ)

    # ── Fallback: check recent history if no mail context yet ────────
    elif not contexts & {_Context.MAIL_SEND, _Context.MAIL_REPLY, _Context.MAIL_READ}:
        if any(w in recent_combined for w in REPLY_TRIGGERS):
            contexts.add(_Context.MAIL_REPLY)
        elif any(w in recent_combined for w in SEND_TRIGGERS):
            contexts.add(_Context.MAIL_SEND)
        elif any(w in recent_combined for w in READ_TRIGGERS):
            contexts.add(_Context.MAIL_READ)

    # ── Also carry forward if account already cached ─────────────────
    if cached_account_id and not contexts & {_Context.MAIL_SEND, _Context.MAIL_REPLY, _Context.MAIL_READ}:
        contexts.add(_Context.MAIL_SEND)

    # Weather
    if any(w in last_human for w in ["weather", "temperature", "rain", "climate", "hot", "cold"]):
        contexts.add(_Context.WEATHER)

    # News
    if any(w in last_human for w in ["news", "headline", "latest update", "today's news"]):
        contexts.add(_Context.NEWS)

    # Policy fallback
    if not contexts or any(w in last_human for w in [
        "policy", "leave", "holiday", "rule", "hr", "salary",
        "attendance", "appraisal", "benefit", "mail id", "contact"
    ]):
        contexts.add(_Context.POLICY)

    return contexts if contexts else {_Context.GENERAL}

def _build_system_prompt(
    messages: list,
    current_date: str,
    zoho_account_id: str = None,
    zoho_from_address: str = None,
    zoho_sender_name: str = None,
) -> str:
    contexts = _detect_context(messages)

    zoho_cache_hint = ""
    if zoho_account_id:
        zoho_cache_hint = (
            f"ZOHO SESSION:\n"
            f"  accountId   = '{zoho_account_id}'\n"
            f"  fromAddress = '{zoho_from_address}'\n"
            f"  senderName  = '{zoho_sender_name}'\n"
            f"Use these values directly — do NOT call ZohoMail_getMailAccounts.\n"
            f"ALWAYS sign emails as '{zoho_sender_name}' — never as 'MACOM HR' or 'MACOM AI'.\n\n"
        )

    base = (
        "You are MACOM AI, a dedicated HR Assistant for MACOM employees.\n"
        f"CURRENT_DATE: {current_date}\n\n"
        f"{zoho_cache_hint}"
        "Never use robotic phrases. Keep responses concise and professional.\n"
    )

    sections = []

    if _Context.POLICY in contexts or _Context.GENERAL in contexts:
        sections.append(
            "POLICY/HR: Call 'Policy_RAG_Implementation' with a plain string query.\n"
            "If not found → say: 'I couldn't find specific policy details for [Topic].'"
        )

    if _Context.WEATHER in contexts:
        sections.append(
            "WEATHER: Call 'Current_Date_weather' with city name only (e.g., 'Kochi').\n"
            "Output: friendly one-liner e.g. 'It's a sunny 37°C in Kochi today'."
        )

    if _Context.NEWS in contexts:
        sections.append(
            "NEWS: Call 'Get_Top_News' with category: business/sports/technology/general.\n"
            "Output: top 5 bullet points with sources only."
        )

    mail_needed = {_Context.MAIL_READ, _Context.MAIL_SEND, _Context.MAIL_REPLY} & contexts
    if mail_needed:
        if zoho_account_id:
            sections.append(
                "ZOHO MAIL — MANDATORY RULES:\n"
                f"- accountId='{zoho_account_id}' and fromAddress='{zoho_from_address}' are already known.\n"
                "- Do NOT call ZohoMail_getMailAccounts — use the values above directly.\n"
                "- accountId, folderId, messageId → ALWAYS strings, never integers.\n"
                "- folderId → always in query_params, NEVER in path_variables.\n"
                "- NEVER use 'Inbox'/'INBOX' as folderId — always use the numeric ID.\n"
                "- sortorder → boolean false, never string 'False'.\n"
                "- Report exact Zoho errors — never guess.\n"
            )
        else:
            sections.append(
                "ZOHO MAIL — MANDATORY RULES:\n"
                "- ALWAYS call ZohoMail_getMailAccounts FIRST before ANY mail operation — no exceptions.\n"
                "- Extract accountId AND fromAddress from the response — NEVER guess, hardcode, or reuse from memory.\n"
                "- accountId, folderId, messageId → ALWAYS strings, never integers.\n"
                "- folderId → always in query_params, NEVER in path_variables.\n"
                "- NEVER use 'Inbox'/'INBOX' as folderId — always use the numeric ID.\n"
                "- sortorder → boolean false, never string 'False'.\n"
                "- Report exact Zoho errors — never guess.\n"
                "- NEVER proceed to send/reply without a confirmed accountId from a live API call.\n"
            )

    if _Context.MAIL_READ in contexts:
        step1 = (
            f"  1. Use accountId='{zoho_account_id}' (already known — skip getMailAccounts)\n"
            if zoho_account_id else
            "  1. ZohoMail_getMailAccounts → accountId (string)\n"
        )
        sections.append(
            "READING EMAILS:\n"
            + step1 +
            "  2. ZohoMail_getAllFolders:\n"
            "     { 'path_variables': {'accountId': '<str>'},\n"
            "       'query_params': {'fields': 'folderId,folderName'} }\n"
            "  3. Find Inbox → extract folderId (string, NOT 'Inbox')\n"
            "  4. ZohoMail_listEmails:\n"
            "     { 'path_variables': {'accountId': '<str>'},\n"
            "       'query_params': {'folderId': '<str>', 'fields': 'subject,messageId,folderId,fromAddress,toAddress,receivedTime',\n"
            "                        'limit': 10, 'sortBy': 'date', 'sortorder': false, 'status': 'all'} }\n"
            "  5. For full content → ZohoMail_getMessageContent:\n"
            "     { 'path_variables': {'accountId': '<str>', 'folderId': '<str>', 'messageId': '<str>'} }\n"
            "  Output: numbered table — No. | From | Subject | Date"
        )

    if _Context.MAIL_SEND in contexts:
        step1 = (
            f"  STEP 1 — SKIP: accountId='{zoho_account_id}', fromAddress='{zoho_from_address}' already known.\n"
            if zoho_account_id else
            "  STEP 1 — MANDATORY: Call ZohoMail_getMailAccounts immediately.\n"
            "           Extract and store: accountId (string), fromAddress (string).\n"
            "           If this call fails → stop and report the error. Do NOT continue.\n"
        )
        sections.append(
            "SENDING EMAIL — STRICT STEP-BY-STEP (do NOT skip or reorder steps):\n"
            + step1 +
            "  STEP 2 — DRAFT PREVIEW: Compose a well-formatted professional email and show it as a readable preview to the user.\n"
            "           Then ask: 'Shall I send this email? (Yes / No)'\n"
            "           Do NOT call ZohoMail_sendEmail yet.\n"
            "  STEP 3 — SEND ONLY AFTER EXPLICIT USER APPROVAL:\n"
            "           Convert the email body to proper HTML and call ZohoMail_sendEmail:\n"
            "           { 'path_variables': {'accountId': '<str from STEP 1>'},\n"
            "             'body': {'fromAddress': '<str from STEP 1>', 'toAddress': '<str>',\n"
            "                      'subject': '<str>', 'content': '<HTML formatted body>', 'mailFormat': 'html'} }\n"
            "           If user says No → ask what they'd like to change.\n"
        )

    if _Context.MAIL_REPLY in contexts:
        step1 = (
            f"  STEP 1 — SKIP: accountId='{zoho_account_id}', fromAddress='{zoho_from_address}' already known.\n"
            if zoho_account_id else
            "  STEP 1 — MANDATORY: Call ZohoMail_getMailAccounts immediately.\n"
            "           Extract and store: accountId (string), fromAddress (string).\n"
            "           If this call fails → stop and report the error. Do NOT continue.\n"
        )
        sections.append(
            "REPLYING TO EMAIL — STRICT STEP-BY-STEP (do NOT skip or reorder steps):\n"
            + step1 +
            "  STEP 2 — FIND MESSAGE: If messageId unknown → follow READING steps 2-4 to find it.\n"
            "  STEP 3 — DRAFT PREVIEW: Compose a well-formatted professional reply and show it as a readable preview to the user.\n"
            "           Then ask: 'Shall I send this reply? (Yes / No)'\n"
            "           Do NOT call ZohoMail_sendReplyEmail yet.\n"
            "  STEP 4 — SEND ONLY AFTER EXPLICIT USER APPROVAL:\n"
            "           Convert the reply body to proper HTML and call ZohoMail_sendReplyEmail:\n"
            "           { 'path_variables': {'accountId': '<str from STEP 1>', 'messageId': '<str>'},\n"
            "             'body': {'action': 'reply', 'fromAddress': '<str from STEP 1>',\n"
            "                      'toAddress': '<str>', 'content': '<HTML formatted body>'} }\n"
            "           If user says No → ask what they'd like to change.\n"
        )

    return base + "\n\n" + "\n\n".join(sections)

async def assistant_node(state: State) -> dict:
    """LLM node that binds tools and generates a response."""
    pipeline     = get_pipeline()
    emp_code     = state.get("emp_code")
    all_tools    = await get_all_tools_for_user(emp_code)
    llm          = pipeline.vertex_llm.bind_tools(all_tools)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # ── Always pre-fetch Zoho account ────────────────────────────────
    zoho_account_id   = None
    zoho_from_address = None
    zoho_sender_name  = None

    try:
        zoho_tool = next(
            (t for t in all_tools if t.name == "ZohoMail_getMailAccounts"), None
        )
        if zoho_tool:
            result   = await zoho_tool.ainvoke({})
            data     = json.loads(result) if isinstance(result, str) else result
            accounts = data.get("data", [])
            if accounts:
                zoho_account_id   = str(accounts[0].get("accountId", ""))
                zoho_from_address = accounts[0].get("fromAddress", "")
                zoho_sender_name  = (
                    accounts[0].get("displayName", "")
                    or _name_from_email(zoho_from_address)
                )
                logger.info(
                    f"[assistant_node] Pre-fetched accountId={zoho_account_id}, "
                    f"fromAddress={zoho_from_address}, senderName={zoho_sender_name}"
                )
    except Exception as e:
        logger.error(f"[assistant_node] Pre-fetch getMailAccounts failed: {e}")
    # ─────────────────────────────────────────────────────────────────

    recent_messages = state["messages"][-6:]
    system_content  = _build_system_prompt(
        recent_messages,
        current_date,
        zoho_account_id=zoho_account_id,
        zoho_from_address=zoho_from_address,
        zoho_sender_name=zoho_sender_name,
    )

    history   = _smart_history(state["messages"])
    has_human = any(isinstance(m, HumanMessage) for m in history)
    if not has_human:
        return {"messages": [AIMessage(content="I'm here to help! What would you like to know?")]}

    messages = [SystemMessage(content=system_content)] + history
    response = await llm.ainvoke(messages)
    logger.error(
        "[assistant_node] LLM response type=%s tool_calls=%s content=%s",
        type(response).__name__,
        getattr(response, "tool_calls", None),
        str(response.content)[:300],
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
            "Leave_Application": "leave_node",
            END: END,
        },
    )

    # Tool call loop: assistant -> tools -> assistant
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile(checkpointer=checkpointer)