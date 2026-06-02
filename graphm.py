import datetime
import json
import os
import uuid
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt
from src.pipeline.leave_agent_node import LeaveToolNode, leave_agent, leave_agent_condition
from src.pipeline.agent_registry import (
    AGENT_REGISTRY,
    get_supervisor_agent_block,
    get_display_to_internal_map,
    get_all_display_names,
    keyword_route,
)
os.environ["FASTEMBED_CACHE_PATH"] = os.getenv("FASTEMBED_CACHE_PATH", "/tmp/fastembed")
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph, START
from src.ColdStart.singleton import get_pipeline
from src.server.mcp_loader import get_mcp_tools
from src.server.zoho_session import get_zoho_tools_for_user
from src.pipeline.tools import tools as localTool
from src.logging import logger
from guardrails_check import get_rails
from langgraph.prebuilt import ToolNode
from statenode import *

# ── Constants ──────────────────────────────────────────────────────────────────
_queue_supervisor = None

TOOL_REGISTRY: dict[str, ToolMetadata] = {
    # Zoho tools
    "ZohoMail_getMailAccounts":   ToolMetadata(category="zoho",  agents=["mail_agent"]),
    "ZohoMail_getAllFolders":     ToolMetadata(category="zoho",  agents=["mail_agent"]),
    "ZohoMail_listEmails":        ToolMetadata(category="zoho",  agents=["mail_agent"]),
    "ZohoMail_getMessageContent": ToolMetadata(category="zoho",  agents=["mail_agent"]),
    "ZohoMail_sendEmail":         ToolMetadata(category="zoho",  agents=["mail_agent"]),
    "ZohoMail_readMessages":      ToolMetadata(category="zoho",  agents=["mail_agent"]),
    "ZohoMail_sendReplyEmail":    ToolMetadata(category="zoho",  agents=["mail_agent"]),
    "ZohoMail_readFolder":        ToolMetadata(category="zoho",  agents=["mail_agent"]),
    # HR tools
    "Policy_RAG_Implementation":  ToolMetadata(category="hr",    agents=["hr_agent"]),
    "Current_Date_weather":       ToolMetadata(category="hr",    agents=["hr_agent"]),
    "Get_Top_News":               ToolMetadata(category="hr",    agents=["hr_agent"]),
    # Leave tools
    "leave_check_balance":        ToolMetadata(category="leave", agents=["leave_agent"]),
    "leave_get_status":           ToolMetadata(category="leave", agents=["leave_agent"]),
    "leave_get_categories":       ToolMetadata(category="leave", agents=["leave_agent"]),
    "leave_calculate_days":       ToolMetadata(category="leave", agents=["leave_agent"]),
    "leave_find_reasons":         ToolMetadata(category="leave", agents=["leave_agent"]),
    "leave_apply":                ToolMetadata(category="leave", agents=["leave_agent"]),
}

_FALLBACK_TOOLS = {
    "mail_agent":  lambda tool: "zoho"  in tool.name.lower(),
    "hr_agent":    lambda tool: any(k in tool.name.lower() for k in ["rag", "policy", "weather", "news"]),
    "leave_agent": lambda tool: "leave" in tool.name.lower(),
}

# ── Confirmation phrases per agent ─────────────────────────────────────────────
# Used by agents to detect when they are asking the user a yes/no question,
# so pending_agent can be set correctly.
AGENT_CONFIRMATION_PHRASES: dict[str, list[str]] = {
    "mail_agent": [
        "shall i send this email",
        "should i send this email",
        "is the mail format correct",
        "is the format in email correct",
        "is the format correct",
    ],
    "hr_agent": [
        "shall i proceed",
        "do you want me to",
        "would you like me to",
        "should i look up",
        "do you confirm",
    ],
    # leave_agent uses LangGraph interrupt() — no confirmation phrase detection needed
}

# ── Tool filtering ─────────────────────────────────────────────────────────────

def _fallback_category(agent_name: str, tool) -> list:
    matcher = _FALLBACK_TOOLS.get(agent_name)
    if matcher and matcher(tool):
        return [tool]
    return []


def get_tools_for_agent(agent_name: str, all_tools: list) -> list:
    result = []
    for tool in all_tools:
        meta = TOOL_REGISTRY.get(tool.name)
        if meta:
            if agent_name in meta.agents:
                result.append(tool)
        else:
            result.extend(_fallback_category(agent_name, tool))
    return result


async def get_all_tools_for_user(emp_code: int = None) -> list:
    shared_tools = get_mcp_tools() + localTool
    try:
        user_zoho_tools = await get_zoho_tools_for_user(emp_code) if emp_code else []
    except Exception as e:
        logger.error(f"[emp:{emp_code}] Failed to load Zoho tools: {e}")
        user_zoho_tools = []

    user_tool_names = {t.name for t in user_zoho_tools}
    filtered_shared = [t for t in shared_tools if t.name not in user_tool_names]
    all_tools       = filtered_shared + user_zoho_tools

    logger.info(
        f"[emp:{emp_code}] Tools: {len(filtered_shared)} shared + "
        f"{len(user_zoho_tools)} Zoho = {len(all_tools)} total"
    )
    return all_tools


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_confirmation_reply(text: str) -> bool:
    """Return True if the user's message is a bare yes/no confirmation."""
    return text.strip().lower().rstrip(".").rstrip("!") in {
        "yes", "no", "y", "n", "confirm", "cancel", "ok", "okay", "sure", "nope"
    }


def _agent_is_asking_confirmation(text: str, agent_name: str) -> bool:
    """Return True if the agent's response text contains a confirmation prompt."""
    phrases = AGENT_CONFIRMATION_PHRASES.get(agent_name, [])
    lower   = text.lower()
    return any(p in lower for p in phrases)


# ── DynamicToolNode ────────────────────────────────────────────────────────────
class DynamicToolNode:
    async def __call__(self, state: State) -> dict:
        emp_code  = state.get("emp_code")
        all_tools = state.get("all_tools") or await get_all_tools_for_user(emp_code)

        last_msg = state["messages"][-1] if state.get("messages") else None
        if not last_msg or not hasattr(last_msg, "tool_calls"):
            return await ToolNode(all_tools).ainvoke(state)

        NO_ARG_TOOLS = {"ZohoMail_getMailAccounts"}
        patched      = []
        needs_patch  = False

        for tc in (last_msg.tool_calls or []):
            name = tc.get("name", "")
            args = dict(tc.get("args", {}))

            if "Zoho" in name and name not in NO_ARG_TOOLS:

                if list(args.keys()) == ["kwargs"] and isinstance(args.get("kwargs"), dict):
                    args = dict(args["kwargs"])
                    logger.info(f"[DynamicToolNode] unwrapped kwargs for {name}")

                pv   = dict(args.get("path_variables") or {})
                qp   = dict(args.get("query_params")   or {})
                body = dict(args.get("body")            or {})

                if "accountId" in args:
                    val = args.pop("accountId")
                    if val and val != "unknown":
                        pv["accountId"] = val
                if pv.get("accountId", "unknown") in ("unknown", "", None):
                    if state.get("zoho_account_id"):
                        pv["accountId"] = state["zoho_account_id"]
                        logger.info(f"[DynamicToolNode] injected accountId from state for {name}")
                    else:
                        logger.warning(f"[DynamicToolNode] accountId missing for {name}")

                if "messageId" in args:
                    pv["messageId"] = args.pop("messageId")

                if "folderId" in args:
                    if name == "ZohoMail_getMessageContent":
                        pv["folderId"] = args.pop("folderId")
                    else:
                        qp["folderId"] = args.pop("folderId")

                if name == "ZohoMail_getMessageContent" and "folderId" not in pv:
                    folder_id = state.get("zoho_folder_id")
                    if folder_id:
                        pv["folderId"] = folder_id
                    else:
                        for msg in reversed(state.get("messages", [])):
                            if isinstance(msg, ToolMessage) and msg.name == "ZohoMail_getAllFolders":
                                try:
                                    data    = json.loads(msg.content)
                                    folders = data.get("data", [])
                                    inbox   = next(
                                        (f for f in folders if "inbox" in f.get("folderName", "").lower()),
                                        folders[0] if folders else {}
                                    )
                                    if inbox.get("folderId"):
                                        pv["folderId"] = inbox["folderId"]
                                except Exception:
                                    pass
                                break

                if name == "ZohoMail_getAllFolders":
                    qp["fields"] = "folderId,folderName"
                elif name == "ZohoMail_listEmails":
                    if "folderId" not in qp and state.get("zoho_folder_id"):
                        qp["folderId"] = state["zoho_folder_id"]
                    qp["fields"] = "messageId,subject,sender,receivedTime"
                    if "sortOrder" in qp:
                        qp["sortorder"] = qp.pop("sortOrder")
                    so = qp.get("sortorder", False)
                    if isinstance(so, str):
                        qp["sortorder"] = so.lower() not in ("false", "0", "no", "off")
                    elif isinstance(so, int):
                        qp["sortorder"] = bool(so)
                    else:
                        qp["sortorder"] = False

                BODY_FIELDS = {"fromAddress", "toAddress", "subject", "content", "mailFormat", "ccAddress", "bccAddress"}
                for f in BODY_FIELDS:
                    if f in args:
                        body[f] = args.pop(f)

                if name in ("ZohoMail_sendEmail", "ZohoMail_sendReplyEmail"):
                    if body.get("fromAddress", "unknown") in ("unknown", "", None):
                        if state.get("zoho_from_address"):
                            body["fromAddress"] = state["zoho_from_address"]

                args.clear()
                if pv:   args["path_variables"] = pv
                if qp:   args["query_params"]   = qp
                if body: args["body"]            = body

                logger.info(f"[DynamicToolNode] {name} → {json.dumps(args, default=str)[:300]}")
                tc          = {**tc, "args": args}
                needs_patch = True

            elif "Zoho" in name and name in NO_ARG_TOOLS:
                tc          = {**tc, "args": {}}
                needs_patch = True
                logger.info(f"[DynamicToolNode] cleared args for no-arg tool {name}")

            patched.append(tc)

        if needs_patch:
            state = {
                **state,
                "messages": state["messages"][:-1] + [
                    AIMessage(content=last_msg.content, tool_calls=patched)
                ],
            }

        try:
            result = await ToolNode(all_tools).ainvoke(state)
        except GraphInterrupt:
            raise
        except Exception as e:
            logger.error(f"[DynamicToolNode] Tool execution error: {e}")
            return {"messages": [AIMessage(content=f"Tool execution failed: {e}")]}

        for msg in reversed(result.get("messages", [])):
            if not isinstance(msg, ToolMessage):
                break
            try:
                data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                if not isinstance(data, dict):
                    continue
            except Exception:
                continue

            if msg.name == "ZohoMail_getMailAccounts":
                acct = (data.get("data") or [{}])[0]
                result["zoho_account_id"]   = str(acct.get("accountId", ""))
                result["zoho_from_address"] = acct.get("fromAddress") or acct.get("mailboxAddress", "")
                logger.info(f"[DynamicToolNode] cached zoho_account_id={result['zoho_account_id']}")

            elif msg.name == "ZohoMail_getAllFolders":
                folders = data.get("data", [])
                inbox   = next(
                    (f for f in folders if "inbox" in f.get("folderName", "").lower()),
                    folders[0] if folders else {}
                )
                result["zoho_folder_id"] = inbox.get("folderId")
                logger.info(f"[DynamicToolNode] cached zoho_folder_id={result['zoho_folder_id']}")

        return result


# ── System prompts ─────────────────────────────────────────────────────────────

def _build_supervisor_prompt() -> str:
    agent_block = get_supervisor_agent_block()
    valid_names = ", ".join(f'"{n}"' for n in get_all_display_names())

    return f"""
You are a supervisor for MACOM HR assistant.
Your ONLY job is to decide which agents are needed for the user's request — in order.
You do NOT answer the user. You do NOT explain. You only plan the agent queue.

AGENTS:
{agent_block}

PLANNING RULES:
  1. Most requests need exactly ONE agent — do not over-route
  2. Never repeat the same agent in the queue
  3. Order matters — put the most relevant agent first
  4. If the request clearly involves only one domain → return only that agent
  5. If unsure → ["hr_agent"]
  6.Any query about leave status, leave history, or applied leaves → always route to leave_agent, never hr_agent.

MULTI-AGENT (rare — only when request explicitly spans two domains):
  - "check my leave balance AND send an email" → ["leave_agent", "mail_agent"]
  - "what's the weather and read my emails"    → ["hr_agent", "mail_agent"]

STOP — RETURN EMPTY OR FINISH IF ANY OF THESE ARE TRUE:
  - The last AIMessage already contains the answer to the user's request
  - The last AIMessage shows email content, a list of emails, or mail details
  - The last AIMessage asks the user a follow-up question (e.g. "Shall I send?")
  - The last AIMessage confirms a completed action
  - A tool error occurred — do NOT retry the same agent
  - The same agent was already called this turn

  IF ANY ABOVE IS TRUE → return exactly: {{"agents": ["hr_agent"], "reason": "already answered"}}
  Wait for the user's NEXT message before routing again.

AGENT NAME RULES:
  - Only use exact strings: {valid_names}
  - Never invent new agent names
  - Never return an empty list — always return at least one agent
"""


MAIL_AGENT_SYSTEM = """
You are MACOM Mail Agent.

STRICT EXECUTION ORDER — never skip, never guess:
  Step 1: ZohoMail_getMailAccounts  
          args: NONE — call with empty args {{}}
          *** Handled automatically. SESSION will always have accountId.

  Step 2: ZohoMail_getAllFolders     
          args: accountId = SESSION.accountId
          SKIP if SESSION.folderId is already known

  Step 3: ZohoMail_listEmails        
          args: accountId = SESSION.accountId, folderId = SESSION.folderId
          ONLY call if user wants to read/reply emails

  Step 4: ZohoMail_getMessageContent 
          args: accountId = SESSION.accountId, messageId = from Step 3 result
          ONLY call if user wants to read a specific email

  Step 5: ZohoMail_sendEmail
          args: accountId = SESSION.accountId, fromAddress = SESSION.fromAddress,
                toAddress = <user provided>, subject = <user provided>, content = <drafted>
          NEVER call without user confirmation

  Step 6: ZohoMail_sendReplyEmail
          args: accountId = SESSION.accountId, messageId = SESSION.messageId,
                content = <drafted reply>
          NEVER call without user confirmation

SESSION:
  accountId   : {zoho_account_id}
  folderId    : {zoho_folder_id}
  fromAddress : {zoho_from_address}
  TODAY       : {current_date}

ARG RULES:
  - NEVER pass "unknown", null, or empty string as an arg value
  - NEVER guess or fabricate any ID
  - ONLY use values from SESSION or returned by a previous tool
  - Always pass accountId in EVERY tool call that requires it

DRAFT-BEFORE-SEND — MANDATORY:
  1. Write the complete email draft with "Subject:", "To:", and full body
  2. End with exactly: "Shall I send this email? (Yes / No)"
  3. Do NOT call send tools in this turn
  4. ONLY call send tool AFTER user confirms with Yes in the NEXT turn

CONFIRMATION HANDLING:
  - If the conversation shows a drafted email followed by user saying "Yes" or "Yes, please send":
    → Immediately call ZohoMail_sendEmail (or ZohoMail_sendReplyEmail for replies)
    → Use the exact draft content — do NOT rewrite or ask again
    → Do NOT say "I cannot send emails" — you have the tools and confirmation to proceed
  - If user says "No" → acknowledge and ask what they'd like to change
  
GENERAL RULES:
  - One tool at a time, strict order above
  - Never re-fetch what is already in SESSION
  - Never call sendEmail or sendReplyEmail without explicit prior confirmation
"""

HR_AGENT_SYSTEM = """
You are MACOM HR Agent, an assistant for MACOM employees.

You help with:
  - HR policy questions (leave policy, dress code, holidays, entitlements)
  - Current date and weather information
  - Latest news
  - General HR queries

TODAY: {current_date}

TOOLS:
  - Policy_RAG_Implementation : HR policy documents
  - Current_Date_weather       : Date, time, weather
  - Get_Top_News               : Latest news

RULES:
  - Use Policy_RAG_Implementation for ANY policy/HR procedure question
  - Use Current_Date_weather for date/time/weather
  - Use Get_Top_News for news
  - Answer general knowledge directly without tools
  - You do NOT handle email or leave tasks

POLICY RESPONSE RULES (CRITICAL):
  - ALWAYS call Policy_RAG_Implementation before answering any HR or policy question
  - Answer ONLY based on what is explicitly found in the retrieved documents
  - Do NOT infer, assume, or generate policy details from your own knowledge
  - Never partially answer using your own knowledge when documents are insufficient
  - Do not say "typically", "usually", or "in most companies" — only cite what MACOM documents state

RETRY RULES:
  - If Policy_RAG_Implementation returns empty or no relevant results:
      → Call it ONE more time with a broader/rephrased query
      → Only if the second attempt also returns nothing → respond with the sorry message
  - NEVER show the sorry message after just one failed RAG call

HISTORY RULES:
  - NEVER repeat a previous response from history
  - NEVER use a previous "sorry" or "not found" message as your answer
  - If the last response in history was a sorry/not-found message for the same question:
      → Ignore it completely
      → Call Policy_RAG_Implementation fresh with a broader query
  - Always make a fresh tool call — do NOT recycle old answers from history

FINAL FALLBACK:
  - Only after two failed RAG attempts, respond with:
    "I'm sorry, I wasn't able to find information on that in our current HR policy documents.
     For further assistance, please reach out to the HR team directly."
"""


# ── Supervisor ─────────────────────────────────────────────────────────────────

def get_queue_supervisor():
    """Singleton — built once, reused."""
    global _queue_supervisor
    if _queue_supervisor is None:
        _queue_supervisor = (
            get_pipeline().vertex_llm
            .with_structured_output(SupervisorQueue)
        )
    return _queue_supervisor


async def supervisor_node(state: State) -> dict:
    # ── 1. Pop from existing queue first — no LLM needed ─────────────────────
    queue = list(state.get("agent_queue") or [])
    if queue:
        next_agent = queue.pop(0)
        logger.info(f"[supervisor] popping from queue: {next_agent}, remaining: {queue}")
        return {
            "next_agent":  next_agent,
            "agent_queue": queue,
        }

    # ── 2. Generic confirmation detection ─────────────────────────────────────
    # Declare variables BEFORE using them
    messages      = state.get("messages", [])
    pending_agent = state.get("pending_agent")
    last_human    = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    if pending_agent and last_human:
        user_text = (last_human.content or "").strip()
        if _is_confirmation_reply(user_text):
            logger.info(
                f"[supervisor] confirmation '{user_text}' → "
                f"routing back to pending agent: {pending_agent}"
            )
            return {
                "next_agent":    pending_agent,
                "agent_queue":   [],
                "pending_agent": None,  # ← clear after routing
                "reason":        f"confirmation reply to {pending_agent}",
            }

    # ── 3. Normal LLM planning ────────────────────────────────────────────────
    try:
        decision: SupervisorQueue = await get_queue_supervisor().ainvoke(
            [SystemMessage(content=_build_supervisor_prompt())]
            + messages[-6:]
        )

        display_to_internal = get_display_to_internal_map()
        queue = [
            display_to_internal.get(a, "hr_agent")
            for a in (decision.agents or [])
        ]

        seen:        set[str]  = set()
        dedup_queue: list[str] = []
        for a in queue:
            if a not in seen:
                seen.add(a)
                dedup_queue.append(a)
        queue = dedup_queue

        next_agent = queue.pop(0) if queue else None
        logger.info(
            f"[supervisor] planned={decision.agents} "
            f"mapped={queue} reason='{decision.reason}' first={next_agent}"
        )

        return {
            "next_agent":    next_agent,
            "agent_queue":   queue,
            "agent_history": [next_agent] if next_agent else [],
            "reason":        decision.reason,
        }

    except GraphInterrupt:
        raise
    except Exception as e:
        logger.error(f"[supervisor] Error: {e}")
        return {
            "next_agent":  None,
            "agent_queue": [],
            "reason":      f"Fallback due to error: {e}",
            "messages":    [AIMessage(content="I'm sorry, I encountered an error. Please try again.")],
        }


# ── Guard ──────────────────────────────────────────────────────────────────────

async def guard_node(state: State) -> dict:
    messages     = state.get("messages", [])
    last_message = messages[-1] if messages else None

    if not last_message or not isinstance(last_message, HumanMessage):
        return {"responded": False}

    try:
        guardrails  = get_rails()
        res         = await guardrails.generate_async(
            messages=[{"role": "user", "content": last_message.content}]
        )
        output_text = res.get("content", "").strip() if isinstance(res, dict) else str(res).strip()
        PASSTHROUGH = {"Passing your request to the supervisor..."}

        if output_text and output_text not in PASSTHROUGH:
            return {
                "responded":     True,
                "messages":      [AIMessage(content=output_text)],
                "agent_history": [],
                "next_agent":    None,
                "agent_queue":   [],
                "pending_agent": None,
            }

        return {
            "responded":     False,
            "agent_history": [],
            "next_agent":    None,
            "agent_queue":   [],
        }

    except GraphInterrupt:
        raise
    except Exception as e:
        import traceback
        logger.error(f"[guard] generate_async failed: {e}")
        logger.error(traceback.format_exc())
        return {"responded": False, "agent_history": [], "next_agent": None, "agent_queue": []}


# ── Mail Agent ─────────────────────────────────────────────────────────────────

async def mail_agent(state: State) -> dict:
    all_tools    = await get_all_tools_for_user(state.get("emp_code"))
    mail_tools   = get_tools_for_agent("mail_agent", all_tools)
    pipeline     = get_pipeline()
    current_date = datetime.datetime.now().strftime("%A, %d %B %Y %H:%M:%S")

    logger.info(f"[mail_agent] tools={[t.name for t in mail_tools]}")

    # ── No Zoho tools — draft-only mode ──────────────────────────────────────
    if not mail_tools:
        llm    = pipeline.vertex_llm
        system = (
            f"You are MACOM Mail Agent. TODAY: {current_date}\n\n"
            f"You do not have Zoho Mail access.\n"
            f"ONLY help draft emails — you cannot send, read, or fetch any.\n"
            f"Write a complete professional email.\n"
            f"End with: 'Note: To send this email, please ask your admin to configure Zoho Mail access.'"
        )
        try:
            response = await llm.ainvoke([SystemMessage(content=system)] + state.get("messages", [])[-10:])
            _log_response(response)
            return {
                "messages":      [response],
                "pending_agent": None,  # draft-only — no confirmation needed
            }
        except GraphInterrupt:
            raise
        except Exception as e:
            logger.error(f"[mail_agent] draft-only error: {e}")
            return {
                "messages":      [AIMessage(content=f"Sorry, an error occurred: {e}")],
                "pending_agent": None,
            }

    # ── Ensure accountId fetched before LLM sees any prompt ──────────────────
    if not state.get("zoho_account_id"):
        logger.info("[mail_agent] zoho_account_id missing — injecting getMailAccounts call")
        return {
            "messages": [AIMessage(
                content    = "",
                tool_calls = [{"name": "ZohoMail_getMailAccounts", "args": {}, "id": str(uuid.uuid4()), "type": "tool_call"}]
            )],
            # Do not touch pending_agent here — mid-flow tool call, not a confirmation
        }

    # ── Full mail agent mode ──────────────────────────────────────────────────
    llm    = pipeline.vertex_llm.bind_tools(mail_tools)
    system = MAIL_AGENT_SYSTEM.format(
        zoho_account_id   = state.get("zoho_account_id")   or "unknown",
        zoho_folder_id    = state.get("zoho_folder_id")    or "unknown",
        zoho_from_address = state.get("zoho_from_address") or "unknown",
        current_date      = current_date,
    )

    try:
        response = await llm.ainvoke([SystemMessage(content=system)] + state.get("messages", [])[-14:])
        _log_response(response)

        # Normalise any confirmation variant to one canonical phrase
        # and set pending_agent so supervisor routes back here on "yes"/"no"
        if response.content and not getattr(response, "tool_calls", None):
            text = response.content if isinstance(response.content, str) else str(response.content)

            if _agent_is_asking_confirmation(text, "mail_agent"):
                clean = "\n".join(
                    line for line in text.splitlines()
                    if not _agent_is_asking_confirmation(line, "mail_agent")
                ).strip()
                return {
                    "messages":      [AIMessage(content=clean + "\n\nShall I send this email? (Yes / No)")],
                    "pending_agent": "mail_agent",  # ← supervisor will route back here on yes/no
                }

            # Normal final answer — clear pending
            return {
                "messages":      [response],
                "pending_agent": None,
            }

        # Has tool calls — mid-flow, don't touch pending_agent
        return {"messages": [response]}

    except GraphInterrupt:
        raise
    except Exception as e:
        logger.error(f"[mail_agent] Error: {e}")
        return {
            "messages":      [AIMessage(content=f"Sorry, an error occurred: {e}")],
            "pending_agent": None,
        }


def _log_response(response) -> None:
    logger.info(f"[agent] content   : {repr((response.content or '')[:300])}")
    logger.info(f"[agent] tool_calls: {getattr(response, 'tool_calls', [])}")
    logger.info(f"[agent] finish    : {response.response_metadata.get('finish_reason')}")


# ── HR Agent ───────────────────────────────────────────────────────────────────

async def hr_agent(state: State) -> dict:
    all_tools = await get_all_tools_for_user(state.get("emp_code"))
    hr_tools  = get_tools_for_agent("hr_agent", all_tools)
    llm       = get_pipeline().vertex_llm.bind_tools(hr_tools)

    system = HR_AGENT_SYSTEM.format(
        current_date=datetime.datetime.now().strftime("%A, %d %B %Y %H:%M:%S")
    )

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=system)] + state.get("messages", [])[-6:]
        )
        logger.info(f"[hr_agent] tool_calls={getattr(response, 'tool_calls', None)}")

        # Check if hr_agent is asking the user a yes/no question
        text = ""
        if response.content and not getattr(response, "tool_calls", None):
            text = response.content if isinstance(response.content, str) else str(response.content)

        if text and _agent_is_asking_confirmation(text, "hr_agent"):
            return {
                "messages":      [response],
                "pending_agent": "hr_agent",  # ← route back here on yes/no
            }

        # Normal answer — clear pending_agent
        return {
            "messages":      [response],
            "pending_agent": None,
        }

    except GraphInterrupt:
        raise
    except Exception as e:
        logger.error(f"[hr_agent] Error: {e}")
        return {
            "messages":      [AIMessage(content=f"Sorry, an error occurred: {e}")],
            "pending_agent": None,
        }


# ── Routing ────────────────────────────────────────────────────────────────────

def guard_condition(state: State) -> str:
    return END if state.get("responded", False) else "supervisor"


def supervisor_route(state: State) -> str:
    """Routes supervisor output to correct agent node or END."""
    next_agent = state.get("next_agent")
    if not next_agent or next_agent in ("FINISH", "", None):
        return END

    display_to_internal = get_display_to_internal_map()
    resolved            = display_to_internal.get(next_agent, next_agent)
    VALID_NODES         = set(AGENT_REGISTRY.keys())

    if resolved not in VALID_NODES:
        logger.warning(f"[supervisor_route] unknown agent '{resolved}' → END")
        return END
    return resolved


def agent_done_condition(state: State) -> str:
    """Shared condition for mail_agent and hr_agent — go to tools or END."""
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None

    if last_msg is None:
        return END

    if getattr(last_msg, "tool_calls", None):
        # Check if the most recent tool result was an error — if so, stop
        last_tool_msg = next(
            (m for m in reversed(messages) if isinstance(m, ToolMessage)),
            None
        )
        if last_tool_msg:
            try:
                content = last_tool_msg.content or ""
                data    = json.loads(content) if isinstance(content, str) else content
                if isinstance(data, dict):
                    status = data.get("status", {})
                    code   = status.get("code") if isinstance(status, dict) else None
                    if code and int(code) >= 400:
                        logger.warning(
                            f"[agent_done_condition] tool error code={code} "
                            f"tool={last_tool_msg.name} — blocking retry → END"
                        )
                        return END
            except Exception:
                pass
        return "tools"

    if isinstance(last_msg, AIMessage):
        content = last_msg.content or ""
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        if content.lower().startswith("sorry") or "error occurred" in content.lower():
            logger.warning("[agent_done_condition] error response → END")
            return END

        # Normal final answer (including confirmation prompts) → END
        # supervisor will handle routing back if pending_agent is set
        return END

    return END


def tools_route_back(state: State) -> str:
    """Route shared tool results back to whichever agent made the call."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            tool_names = [tc.get("name", "") for tc in msg.tool_calls]
            if any("zoho"  in t.lower() for t in tool_names): return "mail_agent"
            if any("leave" in t.lower() for t in tool_names): return "leave_agent"
            return "hr_agent"
    return "supervisor"


# ── Graph ──────────────────────────────────────────────────────────────────────

def create_intent_driven_agent(checkpointer=None) -> StateGraph:
    graph = StateGraph(State)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    graph.add_node("guard",       guard_node)
    graph.add_node("supervisor",  supervisor_node)
    graph.add_node("mail_agent",  mail_agent)
    graph.add_node("hr_agent",    hr_agent)
    graph.add_node("leave_agent", leave_agent)
    graph.add_node("tools",       DynamicToolNode())
    graph.add_node("leave_tools", LeaveToolNode())

    # ── Entry ───────────────────────────────────────────────────────────────────
    graph.add_edge(START, "guard")

    # ── Guard → Supervisor or END ───────────────────────────────────────────────
    graph.add_conditional_edges(
        "guard", guard_condition,
        {"supervisor": "supervisor", END: END},
    )

    # ── Supervisor → Agent or END ───────────────────────────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        supervisor_route,
        {
            "mail_agent":  "mail_agent",
            "hr_agent":    "hr_agent",
            "leave_agent": "leave_agent",
            END:           END,
        },
    )

    # ── Agents → Tools or END ───────────────────────────────────────────────────
    graph.add_conditional_edges(
        "mail_agent", agent_done_condition,
        {"tools": "tools", END: END},
    )
    graph.add_conditional_edges(
        "hr_agent", agent_done_condition,
        {"tools": "tools", END: END},
    )

    # ── Shared tools → back to calling agent ───────────────────────────────────
    graph.add_conditional_edges(
        "tools", tools_route_back,
        {
            "mail_agent":  "mail_agent",
            "hr_agent":    "hr_agent",
            "leave_agent": "leave_agent",
            "supervisor":  "supervisor",
        },
    )

    # ── Leave agent — owns its full flow, goes to END directly ─────────────────
    graph.add_conditional_edges(
        "leave_agent", leave_agent_condition,
        {
            "leave_tools": "leave_tools",
            END:           END,
        },
    )
    graph.add_edge("leave_tools", "leave_agent")

    return graph.compile(checkpointer=checkpointer)