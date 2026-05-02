"""
zoho_debug.py
-------------
Run this DIRECTLY on your server to diagnose the Zoho connection issue.

Usage:
    python zoho_debug.py <emp_code>
    python zoho_debug.py 101189
"""

import asyncio
import sys
import os
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def call_tool(session, tool_name: str, arguments: dict):
    """Helper to call a tool and return parsed response."""
    print(f"\n    >>> Calling {tool_name} with args: {json.dumps(arguments, indent=6)}")
    try:
        result = await session.call_tool(tool_name, arguments=arguments)
        content = result.content
        if isinstance(content, list):
            text = "".join(c.text for c in content if hasattr(c, "text"))
        else:
            text = str(content)
        print(f"    <<< Raw response: {text[:500]}")
        try:
            return json.loads(text), text
        except json.JSONDecodeError:
            return None, text
    except Exception as e:
        print(f"    ✗ Tool call FAILED [{type(e).__name__}]: {e}")
        traceback.print_exc()
        return None, str(e)


async def main(emp_code: int):
    print(f"\n{'='*60}")
    print(f"  Zoho MCP Diagnostic  |  emp_code={emp_code}")
    print(f"{'='*60}\n")

    # ── Step 0: load .env ───────────────────────────────────────────────────
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("[0] .env loaded\n")
    except ImportError:
        print("[0] python-dotenv not installed, skipping\n")

    # ── Step 1: key lookup ──────────────────────────────────────────────────
    print("[1] Looking up Zoho key from key store...")
    try:
        from src.server.zoho_key_store import get_zoho_key
        zoho_key = await get_zoho_key(emp_code)
        if not zoho_key:
            print("    ✗ No key found for this emp_code.")
            return
        zoho_key = zoho_key.strip()
        print(f"    ✓ Key found: {zoho_key[:12]}...{zoho_key[-6:]}  (len={len(zoho_key)})")
        if '\n' in zoho_key or '\r' in zoho_key:
            print("    ⚠ WARNING: key contains newline characters!")
        if ' ' in zoho_key:
            print("    ⚠ WARNING: key contains spaces!")
    except Exception as e:
        print(f"    ✗ Key store error: {e}")
        traceback.print_exc()
        return

    zoho_url = (
        f"https://mail-sending-replies-60069513271.zohomcp.in"
        f"/mcp/{zoho_key}/message"
    )
    print(f"\n    URL: {zoho_url}\n")

    # ── Step 2: DNS ─────────────────────────────────────────────────────────
    print("[2] DNS resolution...")
    import socket
    try:
        ip = socket.gethostbyname("mail-sending-replies-60069513271.zohomcp.in")
        print(f"    ✓ Resolved to {ip}")
    except socket.gaierror as e:
        print(f"    ✗ DNS FAILED: {e}")
        return

    # ── Step 3: raw HTTPS ───────────────────────────────────────────────────
    print("\n[3] Raw HTTPS GET to Zoho URL...")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(zoho_url)
            print(f"    ✓ HTTP {r.status_code}  |  body: {r.text[:200]}")
    except ImportError:
        print("    ⚠ httpx not installed — skipping")
    except Exception as e:
        print(f"    ✗ HTTPS error: {e}")

    # ── Step 4: MCP session ─────────────────────────────────────────────────
    print("\n[4] MCP streamablehttp_client handshake...")
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(zoho_url) as (read, write, *_):
            print("    ✓ Transport opened")
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("    ✓ Session initialized")

                # ── Step 5: list tools ──────────────────────────────────────
                print("\n[5] Listing available tools...")
                result = await session.list_tools()
                names = [t.name for t in result.tools]
                print(f"    ✓ {len(names)} tools: {names}")

                # Print full schema for listEmails so we can see exact arg structure
                print("\n[5b] Schema for listEmails and getMailAccounts...")
                for t in result.tools:
                    if t.name in ("ZohoMail_listEmails", "ZohoMail_getMailAccounts",
                                  "listEmails", "getMailAccounts"):
                        print(f"\n    Tool: {t.name}")
                        print(f"    Description: {t.description}")
                        print(f"    Input Schema: {json.dumps(t.inputSchema, indent=6)}")

                # ── Step 6: call getMailAccounts ────────────────────────────
                print("\n[6] Calling getMailAccounts...")
                # Try both naming conventions
                get_accounts_name = next(
                    (n for n in names if "getmailaccounts" in n.lower()), None
                )
                if not get_accounts_name:
                    print("    ✗ getMailAccounts tool not found in tool list!")
                    return

                parsed, raw = await call_tool(session, get_accounts_name, {})
                if not parsed:
                    print(f"    ✗ getMailAccounts returned non-JSON: {raw}")
                    return

                # Extract accountId
                account_id = None
                try:
                    data = parsed.get("data", [])
                    if data:
                        account_id = data[0].get("accountId")
                        from_address = data[0].get("primaryEmailAddress") or \
                                       data[0].get("mailboxAddress")
                        print(f"\n    ✓ accountId   : {account_id}")
                        print(f"    ✓ fromAddress : {from_address}")
                    else:
                        print("    ✗ 'data' array is empty in response!")
                        return
                except Exception as e:
                    print(f"    ✗ Failed to parse accountId: {e}")
                    return

                if not account_id:
                    print("    ✗ accountId is None after parsing!")
                    return

                # ── Step 7: try listEmails with every possible arg structure ─
                print(f"\n[7] Testing listEmails with accountId={account_id}")
                print("    Trying 4 different argument structures to find what works...\n")

                list_emails_name = next(
                    (n for n in names if "listemails" in n.lower()), None
                )
                if not list_emails_name:
                    print("    ✗ listEmails tool not found!")
                    return

                test_cases = [
                    # Structure 1 — nested path_variables (what LLM was sending)
                    {
                        "label": "Nested path_variables",
                        "args":  {"path_variables": {"accountId": account_id}}
                    },
                    # Structure 2 — flat top-level
                    {
                        "label": "Flat top-level",
                        "args":  {"accountId": account_id}
                    },
                    # Structure 3 — inside body
                    {
                        "label": "Inside body",
                        "args":  {"body": {"accountId": account_id}}
                    },
                    # Structure 4 — path_variables as string (some MCPs serialize)
                    {
                        "label": "path_variables with string accountId",
                        "args":  {"path_variables": {"accountId": str(account_id)}}
                    },
                    {
                        "label": "path_variables + query_params (correct full structure)",
                        "args": {
                            "path_variables": {"accountId": account_id},
                            "query_params":   {
                                "fields": "subject,messageId,folderId,fromAddress,toAddress,receivedTime",
                                "limit":  10
                            }
                        }
                    },
                ]

                working_structure = None
                for i, tc in enumerate(test_cases, 1):
                    print(f"    [{i}] Trying: {tc['label']}")
                    parsed_resp, raw_resp = await call_tool(
                        session, list_emails_name, tc["args"]
                    )
                    if "mandatory path variable accountid" in raw_resp.lower():
                        print(f"        ✗ FAILED — accountId not recognized\n")
                    elif "mandatory query param" in raw_resp.lower():
                        print(f"        ✓ accountId ACCEPTED — but missing required query_params\n")
                        working_structure = tc
                        break
                    else:
                        print(f"        ✓ SUCCESS\n")
                        working_structure = tc
                        break

                # ── Step 8: result ──────────────────────────────────────────
                print("\n" + "="*60)
                if working_structure:
                    print(f"  ✓ WORKING ARGUMENT STRUCTURE FOUND:")
                    print(f"    {json.dumps(working_structure['args'], indent=4)}")
                    print(f"\n  → Update your system prompt to use: {working_structure['label']}")
                else:
                    print("  ✗ ALL 4 STRUCTURES FAILED")
                    print("  → The Zoho MCP tool itself is broken or the key is invalid.")
                    print("  → Ask user to regenerate Zoho MCP key.")
                print("="*60)

    except Exception as e:
        print(f"\n    ✗ MCP FAILED [{type(e).__name__}]: {e}")
        traceback.print_exc()
        msg = str(e).lower()
        if "invalid_client" in msg or "token" in msg:
            print("\n→ AUTH ERROR: Key invalid or OAuth expired.")
        elif "connection" in msg or "timeout" in msg:
            print("\n→ NETWORK ERROR: Can't reach Zoho MCP server.")
        elif "404" in msg:
            print("\n→ URL ERROR: Wrong endpoint — regenerate key.")
        elif "sse" in msg or "protocol" in msg:
            print("\n→ PROTOCOL ERROR: Run: pip install --upgrade mcp")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python zoho_debug.py <emp_code>")
        sys.exit(1)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main(int(sys.argv[1])))
















    
import datetime
import os

from src.pipeline.leave_agent_node import leave_subgraph

os.environ["FASTEMBED_CACHE_PATH"] = "C:/fastembed_models"
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, START
from nemoguardrails.rails.llm.options import GenerationOptions

from src.ColdStart.singleton import get_pipeline
from src.server.mcp_loader import get_mcp_tools, get_all_tools_for_user
from src.pipeline.tools import tools as localTool
from src.logging import logger
from guardrails_check import get_rails
from langgraph.prebuilt import ToolNode, tools_condition
from statenode import State

load_dotenv()

guardrails = get_rails()


# ---------------------------------------------------------------------------
# Tool resolution — always use emp_code from state
# ---------------------------------------------------------------------------

async def _resolve_tools_for_state(state: dict) -> list:
    """
    Returns the full tool list for the current user.
    Falls back to shared tools if emp_code is absent.
    """
    emp_code = state.get("emp_code")
    if emp_code:
        user_tools = await get_all_tools_for_user(emp_code)
    else:
        user_tools = get_mcp_tools()
    return user_tools + localTool


# ---------------------------------------------------------------------------
# Dynamic tool node — rebuilds with the correct per-user tools on every call
# ---------------------------------------------------------------------------
class DynamicToolNode:
    async def __call__(self, state: dict):
        # 1. Capture the LLM's tool call
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls"):
            for tc in last_message.tool_calls:
                # 2. DATA NORMALIZATION (The Fix)
                # If the LLM sent it flat, but the tool needs it nested (or vice versa)
                if tc['name'] == "ZohoMail_listEmails":
                    # Ensure accountId is where the tool actually wants it
                    if "accountId" in tc['args'] and "path_variables" not in tc['args']:
                        tc['args']["path_variables"] = {"accountId": tc['args'].pop("accountId")}
        
        # 3. Pass the normalized arguments to the actual tool node
        node = ToolNode(await _resolve_tools_for_state(state))
        return await node.ainvoke(state)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def Orchestrator(state) -> dict:
 
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
 
    EXPLICIT_LEAVE_TRIGGERS = [
        "apply leave", "apply for leave", "apply the leave",
        "i want to apply", "submit leave", "request leave",
        "book leave", "i need a day off", "need leave",
        "apply casual", "apply sick", "apply earned",
        "raise a leave", "put in leave", "request time off"
    ]
 
    def is_explicit_leave(msg: str) -> bool:
        return any(p in msg.lower().strip() for p in EXPLICIT_LEAVE_TRIGGERS)
 
    # Case 1: Active leave interrupt — skip guardrail
    if leave_step in active_leave_steps:
        print(">>> interrupt pending — skipping guardrail")
        return {"intent": "apply_leave", "responded": False}
 
    # Case 2: Explicit leave request
    if is_explicit_leave(latest_user_msg):
        print(">>> explicit leave request — routing to leave_node")
        return {
            "intent":        "apply_leave",
            "active_intent": "apply_leave",
            "responded":     False,
        }
 
    # ── FIX: Short-circuit guardrails entirely when already in Assistant ──
    # When the user is mid-conversation with the assistant, every follow-up
    # message hits guardrails which fires bot:stop (0 LLM calls).
    # Skip guardrails completely — we already know the intent.
    if active_intent == "Assistant":
        print(">>> active_intent=Assistant — bypassing guardrails entirely")
        return {
            "intent":        "Assistant",
            "active_intent": "Assistant",
            "responded":     False,
        }
 
    # Case 3: Run guardrail
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
 
    # Case 4: Guardrail blocked with a real message (not a passthrough)
    if output_text and output_text not in PASSTHROUGH_MESSAGES:
        return {
            "messages":  [AIMessage(content=output_text)],
            "intent":    detected_intent or active_intent,
            "responded": True,
        }
 
    # Case 5: apply_leave detected mid-assistant-conversation (already
    # handled above by the early return, but kept as safety net)
    if detected_intent == "apply_leave" and active_intent == "Assistant":
        print(">>> non-explicit apply_leave during assistant — staying in assistant")
        return {
            "intent":        "Assistant",
            "active_intent": "Assistant",
            "responded":     False,
        }
 
    # Case 6: Resolve final intent
    VAGUE_INTENTS = {"follow_up", "ask off topic", None}
 
    previous_active_intent = (
        "general"
        if leave_step in {"cancelled", "completed", "failed"}
        else active_intent
    )
 
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


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

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
            return "Leave_Application"
        return "assistant"

    return "assistant"


# ---------------------------------------------------------------------------
# Assistant node — binds per-user tools resolved from state.emp_code
# ---------------------------------------------------------------------------

async def assistant_node(state: State) -> dict:
    """LLM node that binds the correct tools for this user and generates a response."""
    pipeline = get_pipeline()

    # ── Resolve tools for this specific user ──────────────────────────────
    all_tools = await _resolve_tools_for_state(state)
    logger.info("[assistant_node] emp_code=%s  tools=%d", state.get("emp_code"), len(all_tools))

    llm = pipeline.vertex_llm.bind_tools(all_tools)

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    system_content = (
            "You are MACOM AI, a dedicated HR Assistant for MACOM employees.\n\n"
            f"CURRENT_DATE: {current_date}\n\n"

            "--- GENERAL INSTRUCTIONS ---\n"
            "1. Always try to get data first using the Policy_RAG_Implementation tool before stating data was not found.\n"
            "2. For any email-related task, follow the ZOHO MAIL RULES strictly.\n"
            "3. SYNTHESIZED SEARCH: If a user provides extra detail to a previous question, combine it with context before searching.\n"
            "4. If the answer is in conversation history, answer directly without tool calls.\n"
            "5. If user asks about mail IDs of HR head or staff, use Policy_RAG_Implementation.\n\n"

            "--- TOOL EXECUTION RULES ---\n"
            "1. Policy/Holiday/Leave queries -> call 'Policy_RAG_Implementation' with a plain string.\n"
            "2. Weather queries -> call 'Current_Date_weather' with the city name only.\n"
            "3. News queries -> call 'Get_Top_News' with a category.\n"
            "4. ABSOLUTE RULE: Never pass raw dict/JSON as the primary argument to 'Policy_RAG_Implementation', 'Current_Date_weather', or 'Get_Top_News'. Use plain strings.\n\n"

            "--- ZOHO MAIL RULES (CRITICAL) ---\n\n"

            "RULE 0: THE INITIALIZATION\n"
            "You MUST call 'ZohoMail_getMailAccounts' as the FIRST tool call for ANY email task.\n"
            "Extract 'accountId' and 'fromAddress' from data[0].\n\n"

            "1. LISTING EMAILS (ZohoMail_listEmails):\n"
            "   CRITICAL: You MUST wrap accountId inside a 'path_variables' dictionary.\n"
            "   Structure:\n"
            "   {\n"
            "     'path_variables': { 'accountId': '<retrieved_id>' },\n"
            "     'query_params': {\n"
            "        'fields': 'subject,messageId,folderId,fromAddress,toAddress,receivedTime',\n"
            "        'limit': 10\n"
            "     }\n"
            "   }\n\n"

            "2. SENDING EMAIL (ZohoMail_sendEmail):\n"
            "   Structure:\n"
            "   {\n"
            "     'path_variables': { 'accountId': '<retrieved_id>' },\n"
            "     'body': {\n"
            "        'fromAddress': '<retrieved_fromAddress>',\n"
            "        'toAddress': '<recipient>',\n"
            "        'subject': '<subject>',\n"
            "        'content': '<html_content>',\n"
            "        'mailFormat': 'html'\n"
            "     }\n"
            "   }\n\n"

            "3. REPLYING (ZohoMail_sendReplyEmail):\n"
            "   Structure:\n"
            "   {\n"
            "     'path_variables': { 'accountId': '<retrieved_id>', 'messageId': '<target_msg_id>' },\n"
            "     'body': {\n"
            "        'action': 'reply',\n"
            "        'fromAddress': '<retrieved_fromAddress>',\n"
            "        'toAddress': '<original_sender>',\n"
            "        'content': '<reply_text>'\n"
            "     }\n"
            "   }\n\n"

            "--- ABSOLUTE RESTRICTIONS ---\n"
            " - NEVER pass accountId as a top-level flat key. It MUST be inside 'path_variables'.\n"
            " - If the error 'Mandatory path variable accountId is not present' persists, it means the server is expecting 'accountId' inside the 'body' dictionary for that specific tool. In that case, move accountId into 'body'.\n"
            " - NEVER ask the user for their accountId.\n"
            " - ALWAYS show a summary (To, Subject, Preview) and ask for confirmation before sending/replying.\n"
        )
    history = state["messages"][-8:]
    has_human = any(isinstance(m, HumanMessage) for m in history)
    if not has_human:
        return {"messages": [AIMessage(content="I'm here to help! What would you like to know?")]}

    messages = [SystemMessage(content=system_content)] + history

    response = await llm.ainvoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def create_intent_driven_agent(checkpointer=None) -> StateGraph:
    _leave_subgraph_compiled = leave_subgraph(checkpointer=checkpointer)

    graph = StateGraph(State)

    graph.add_node("orchestrator", Orchestrator)
    graph.add_node("assistant", assistant_node)
    graph.add_node("leave_node", _leave_subgraph_compiled)
    graph.add_node("tools", DynamicToolNode())

    graph.add_edge(START, "orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_after_classification,
        {
            "assistant":        "assistant",
            "Leave_Application": "leave_node",
            END:                END,
        },
    )

    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile(checkpointer=checkpointer)