# import datetime
# import json
# import os
# import uuid

# from src.pipeline.leave_agent_node import leave_subgraph  
# os.environ["FASTEMBED_CACHE_PATH"] = "C:/fastembed_models"
# from dotenv import load_dotenv
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from langgraph.graph import END, StateGraph, START
# from langchain_core.language_models import BaseChatModel
# from nemoguardrails.rails.llm.options import GenerationOptions
# # from src.pipeline.leave_agent import run_leave_agent
# from src.ColdStart.singleton import get_pipeline
# from src.server.mcp_loader import get_mcp_tools
# from src.server.zoho_session import get_zoho_tools_for_user
# from src.pipeline.tools import tools as localTool
# from src.logging import logger
# from guardrails_check import get_rails
# from langgraph.prebuilt import ToolNode, tools_condition
# from statenode import State
# load_dotenv()

# guardrails = get_rails()

# async def get_all_tools_for_user(emp_code: int = None) -> list:
#     """
#     Build the full tool list for one user:
#       1. Shared MCP tools (internal server — same for everyone)
#       2. Local tools (RAG, weather, news — same for everyone)
#       3. Per-user Zoho tools (fetched via their saved hash — unique per user)

#     Zoho tools override any shared tool with the same name.
#     """
#     # ── Shared tools (same for all users) ─────────────────────────────
#     shared_tools = get_mcp_tools() + localTool

#     if not emp_code:
#         return shared_tools

#     # ── Per-user Zoho tools ────────────────────────────────────────────
#     try:
#         user_zoho_tools = await get_zoho_tools_for_user(emp_code)
#     except Exception as e:
#         logger.error(f"[emp:{emp_code}] Failed to load Zoho tools: {e}")
#         user_zoho_tools = []

#     if not user_zoho_tools:
#         return shared_tools

#     # Zoho tools override any shared tool with the same name
#     user_tool_names = {t.name for t in user_zoho_tools}
#     filtered_shared = [t for t in shared_tools if t.name not in user_tool_names]

#     all_tools = filtered_shared + user_zoho_tools
#     logger.info(
#         f"[emp:{emp_code}] Tools: {len(filtered_shared)} shared + "
#         f"{len(user_zoho_tools)} Zoho = {len(all_tools)} total"
#     )
#     return all_tools


# class DynamicToolNode:
#     async def __call__(self, state: State) -> dict:
#         emp_code = state.get("emp_code")
#         tools    = await get_all_tools_for_user(emp_code)

#         # ── Patch missing accountId in any Zoho mail tool calls ──
#         zoho_account_id   = state.get("zoho_account_id")
#         zoho_from_address = state.get("zoho_from_address")
#         print(f"[TOOL_NODE] zoho_account_id={zoho_account_id} zoho_from_address={zoho_from_address}")
#         if zoho_account_id:
#             last_msg = state["messages"][-1] if state.get("messages") else None
#             if last_msg and hasattr(last_msg, "tool_calls"):
#                 patched = []
#                 for tc in (last_msg.tool_calls or []):
#                     name = tc.get("name", "")
#                     if "Zoho" in name or "zoho" in name:
#                         args = dict(tc.get("args", {}))
#                         # Unwrap kwargs wrapper if LLM used it (dict or JSON string)
#                         if list(args.keys()) == ["kwargs"]:
#                             inner = args["kwargs"]
#                             if isinstance(inner, str):
#                                 try:
#                                     inner = json.loads(inner)
#                                 except Exception:
#                                     pass
#                             if isinstance(inner, dict):
#                                 args = inner
#                         print(f"[TOOL_NODE] raw tool_call name={name} args={json.dumps(args, default=str)[:300]}")

#                         # Ensure path_variables has accountId
#                         pv = dict(args.get("path_variables") or {})
#                         pv["accountId"] = zoho_account_id
#                         args["path_variables"] = pv
#                         args.pop("accountId", None)

#                         # Collect all body fields — from nested body OR flat top-level
#                         body = dict(args.get("body") or {})
#                         BODY_FIELDS = {"fromAddress","toAddress","subject","content",
#                                        "mailFormat","ccAddress","bccAddress","encoding",
#                                        "action","mode"}
#                         # Promote any flat body fields into body dict
#                         for f in BODY_FIELDS:
#                             if f in args and f not in body:
#                                 body[f] = args.pop(f)
#                         # Ensure fromAddress is set
#                         if not body.get("fromAddress") and zoho_from_address:
#                             body["fromAddress"] = zoho_from_address
#                         # Normalise toAddress
#                         to = body.get("toAddress")
#                         if isinstance(to, list):
#                             body["toAddress"] = ",".join(str(x) for x in to)
#                         elif isinstance(to, dict):
#                             body["toAddress"] = to.get("address") or to.get("email") or str(to)
#                         if not body.get("mailFormat") and name == "ZohoMail_sendEmail":
#                             body["mailFormat"] = "html"
#                         args["body"] = body
#                         tc = {**tc, "args": args}
#                     patched.append(tc)
#                 if patched != list(last_msg.tool_calls):
#                     from langchain_core.messages import AIMessage as _AI
#                     fixed_msg = _AI(
#                         content=last_msg.content,
#                         tool_calls=patched,
#                     )
#                     state = {**state, "messages": state["messages"][:-1] + [fixed_msg]}
#                     logger.info("[DynamicToolNode] patched tool_calls: %s", patched)

#         node = ToolNode(tools)
#         return await node.ainvoke(state)


# async def Orchestrator(state: State) -> State:

#     if not (state.get("messages") and isinstance(state["messages"][-1], HumanMessage)):
#         return {
#             "intent":    state.get("intent", "general"),
#             "responded": False,
#         }

#     latest_user_msg = state["messages"][-1].content.strip()
#     leave_step      = state.get("leave_step")
#     active_intent   = state.get("active_intent", "general")

#     active_leave_steps = {
#         "awaiting_leave_type", "awaiting_dates",
#         "awaiting_to_date", "awaiting_remarks", "awaiting_submission"
#     }

#     # ── Explicit leave trigger phrases ──
#     EXPLICIT_LEAVE_TRIGGERS = [
#         "apply leave", "apply for leave", "apply the leave",
#         "i want to apply", "submit leave", "request leave",
#         "book leave", "i need a day off", "need leave",
#         "apply casual", "apply sick", "apply earned",
#         "raise a leave", "put in leave", "request time off"
#     ]

#     EXPLICIT_MAIL_TRIGGERS = [
#         "draft", "compose", "send mail", "send email",
#         "write mail", "write email", "write a mail", "write an email",
#         "send a mail", "send an email", "mail to", "email to",
#         "shoot a mail", "shoot an email",
#         "check mail", "check email", "show mail", "show email",
#         "read mail", "read email", "recent mail", "recent email",
#         "my inbox", "fetch mail", "fetch email",
#         "reply to", "unread mail", "unread email",
#     ]

#     def is_explicit_leave(msg: str) -> bool:
#         msg_lower = msg.lower().strip()
#         return any(phrase in msg_lower for phrase in EXPLICIT_LEAVE_TRIGGERS)

#     def is_explicit_mail(msg: str) -> bool:
#         msg_lower = msg.lower().strip()
#         return any(phrase in msg_lower for phrase in EXPLICIT_MAIL_TRIGGERS)

#     # Case 1: Active interrupt — skip guardrail
#     if leave_step in active_leave_steps:
#         print(f">>> interrupt pending — skipping guardrail")
#         return {"intent": "apply_leave", "responded": False}

#     # Case 2: Explicit leave request — skip guardrail entirely
#     if is_explicit_leave(latest_user_msg):
#         print(f">>> explicit leave request detected — routing directly to leave_node")
#         return {
#             "intent":        "apply_leave",
#             "active_intent": "apply_leave",
#             "responded":     False,
#         }

#     # Case 2b: Explicit mail request — skip guardrail entirely
#     if is_explicit_mail(latest_user_msg):
#         print(f">>> explicit mail request detected — routing directly to assistant")
#         return {
#             "intent":        "Assistant",
#             "active_intent": "Assistant",
#             "responded":     False,
#         }

#     # Case 2c: Already in an assistant conversation — skip guardrail for follow-ups
#     if active_intent == "Assistant":
#         print(f">>> active_intent=Assistant — skipping guardrail for follow-up")
#         return {
#             "intent":        "Assistant",
#             "active_intent": "Assistant",
#             "responded":     False,
#         }

#     # Case 3: Run guardrail (embeddings only — fast)
#     input_messages = [{"role": "user", "content": latest_user_msg}]
#     print(f"Input to Guardrail: {input_messages}")

#     res = await guardrails.generate_async(
#         messages=input_messages,
#         options=GenerationOptions(
#             output_vars=True,
#             log={
#                 "activated_rails": False,
#                 "llm_calls":       False,
#                 "internal_events": True,
#                 "colang_history":  False,
#             },
#             rails=["input", "dialog"],
#         ),
#     )

#     intents = [
#         e for e in (res.log.internal_events or [])
#         if e.get("type") == "UserIntent"
#     ]
#     detected_intent = intents[-1].get("intent") if intents else None
#     print(f"Detected intent: {detected_intent}")

#     output_text = res.response[-1].get("content", "") if res.response else ""

#     PASSTHROUGH_MESSAGES = {
#         "Passing your request to the assistant...",
#         "Passing your request to the leave system..."
#     }

#     # Case 4: Guardrail blocked
#     # Bypass if: mid-assistant-conversation OR guardrail detected Assistant intent
#     if output_text and output_text not in PASSTHROUGH_MESSAGES:
#         print(f">>> Case 4 fired: output_text='{output_text}' detected_intent={detected_intent} active_intent={active_intent}")
#         if active_intent == "Assistant" or detected_intent == "Assistant":
#             print(">>> guardrail blocked but intent=Assistant — continuing to assistant")
#             return {
#                 "intent":        "Assistant",
#                 "active_intent": "Assistant",
#                 "responded":     False,
#             }
#         print(f">>> BLOCKING with: {output_text}")
#         return {
#             "messages":  [AIMessage(content=output_text)],
#             "intent":    detected_intent or active_intent,
#             "responded": True,
#         }

#     # Case 5: apply_leave detected but mid-assistant-conversation
#     # Only override if NOT an explicit leave request (already handled in Case 2)
#     if detected_intent == "apply_leave" and active_intent == "Assistant":
#         print(">>> non-explicit apply_leave during assistant conversation — staying in assistant")
#         return {
#             "intent":        "Assistant",
#             "active_intent": "Assistant",
#             "responded":     False,
#         }

#     # Case 6: Resolve final intent
#     VAGUE_INTENTS = {"follow_up", "ask off topic", None}

#     if leave_step in {"cancelled", "completed", "failed"}:
#         previous_active_intent = "general"
#     else:
#         previous_active_intent = active_intent

#     if detected_intent not in VAGUE_INTENTS:
#         final_intent      = detected_intent
#         new_active_intent = detected_intent
#     else:
#         final_intent      = previous_active_intent
#         new_active_intent = previous_active_intent

#     print(f"Final intent: {final_intent} | active_intent: {new_active_intent}")

#     return {
#         "intent":        final_intent,
#         "active_intent": new_active_intent,
#         "responded":     False,
#     }


# def route_after_classification(state: State) -> str:
#     print(f">>> routing: responded={state.get('responded')}, intent={state.get('intent')}")

#     if state.get("responded", False):
#         return END

#     leave_step = state.get("leave_step")

#     active_leave_steps = {
#         "awaiting_leave_type",
#         "awaiting_dates",
#         "awaiting_to_date",
#         "awaiting_remarks",
#         "awaiting_submission"
#     }


#     if leave_step in active_leave_steps:
#         return "Leave_Application"

#     if leave_step in {"cancelled", "completed", "failed", None}:
#         intent = state.get("intent", "general")
#         if intent == "apply_leave":
#             return "Leave_Application"   # fresh start — enters leave_balance_node
#         if intent == "Assistant":
#             return "assistant"
#         return "assistant"

#     return "assistant"

# def _smart_history(messages: list) -> list:
#     """
#     Return a window of messages that always includes:
#     - The last HumanMessage and everything after it (tool calls + tool results)
#     - Up to 4 previous messages for context
#     Never cuts off mid tool-call/result cycle.
#     """
#     # Find the index of the last HumanMessage
#     last_human_idx = None
#     for i in range(len(messages) - 1, -1, -1):
#         if isinstance(messages[i], HumanMessage):
#             last_human_idx = i
#             break

#     if last_human_idx is None:
#         return messages[-6:]

#     # Take 4 messages before the last human turn for context, plus everything from it onward
#     start = max(0, last_human_idx - 4)
#     return messages[start:]


# def _name_from_email(email: str) -> str:
#     try:
#         local = email.split("@")[0]
#         return " ".join(p.capitalize() for p in local.split("."))
#     except Exception:
#         return email


# def _extract_draft_from_messages(messages: list) -> dict | None:
#     """Find the most recent AI draft message and extract to/subject/content."""
#     import re
#     for msg in reversed(messages):
#         if not isinstance(msg, AIMessage):
#             continue
#         content = msg.content if isinstance(msg.content, str) else ""
#         if "shall i send" not in content.lower():
#             continue
#         to      = re.search(r"To:\s*([\w.@+\-]+)", content)
#         subject = re.search(r"Subject:\s*(.+)", content)
#         body_match = re.search(r"(?:Dear|Hi|Hello).+?(?=Shall I send)", content, re.DOTALL | re.IGNORECASE)
#         return {
#             "toAddress": to.group(1).strip() if to else "",
#             "subject":   subject.group(1).strip() if subject else "(no subject)",
#             "content":   body_match.group(0).strip() if body_match else content,
#         }
#     return None


# def _build_system_prompt(
#     current_date: str,
#     zoho_account_id: str = None,
#     zoho_from_address: str = None,
#     zoho_sender_name: str = None,
# ) -> str:
#     zoho_session = (
#         f"ZOHO SESSION (already authenticated):\n"
#         f"  accountId   = '{zoho_account_id}'\n"
#         f"  fromAddress = '{zoho_from_address}'\n"
#         f"  senderName  = '{zoho_sender_name}'\n"
#         f"Use these directly. Do NOT call ZohoMail_getMailAccounts.\n"
#         f"Always sign emails as '{zoho_sender_name}'.\n"
#     ) if zoho_account_id else (
#         "ZOHO MAIL: Call ZohoMail_getMailAccounts FIRST to get accountId and fromAddress "
#         "before any mail operation. Never guess or hardcode these values.\n"
#     )
#     return (
#         f"You are MACOM AI, an HR Assistant for MACOM employees.\n"
#         f"TODAY'S DATE IS {current_date}. Use this directly — never ask the user for the date.\n\n"
#         f"{zoho_session}\n"
#         "CAPABILITIES — use the right tool for each task:\n"
#         "- HR/Policy questions → call Policy_RAG_Implementation\n"
#         "- Weather → call Current_Date_weather with city name\n"
#         "- News → call Get_Top_News with category\n"
#         "- Email (read/send/reply) → use Zoho tools per rules below\n\n"
#         "EMAIL RULES:\n"
#         "- NEVER ask the user for accountId, folderId, messageId or any technical ID — fetch them via tools\n"
#         "- NEVER ask the user for subject or content if they already described the email — infer and draft it\n"
#         "- folderId: always call ZohoMail_getAllFolders to get it, never use folder name as ID\n"
#         "- folderId goes in query_params, never in path_variables\n"
#         "- sortorder: boolean false, not string\n"
#         "- SENDING: always show a full draft preview first and ask 'Shall I send this? (Yes/No)' — "
#         "NEVER call ZohoMail_sendEmail before user confirms Yes\n"
#         "- REPLYING: same — show draft, wait for Yes, then send\n"
#     )

# def _extract_zoho_account(messages: list) -> tuple:
#     """Scan ToolMessage history for a prior ZohoMail_getMailAccounts result."""
#     from langchain_core.messages import ToolMessage
#     for msg in reversed(messages):
#         if not isinstance(msg, ToolMessage) or msg.name != "ZohoMail_getMailAccounts":
#             continue
#         try:
#             data     = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
#             accounts = (data.get("data", []) if isinstance(data, dict) else [])
#             if accounts:
#                 acct         = accounts[0]
#                 account_id   = str(acct.get("accountId", ""))
#                 # fromAddress can live at top-level OR inside sendMailDetails
#                 from_address = (
#                     acct.get("fromAddress")
#                     or acct.get("mailboxAddress")
#                     or (acct.get("sendMailDetails") or [{}])[0].get("fromAddress", "")
#                 )
#                 sender_name  = acct.get("displayName", "") or _name_from_email(from_address)
#                 return account_id, from_address, sender_name
#         except Exception:
#             pass
#     return None, None, None


# _FOLDER_ASK_PHRASES = [
#     "folder id", "folderid", "which folder", "folder name",
#     "inbox, sent", "inbox or sent", "e.g., inbox", "specify which folder",
#     "please specify", "could you please specify",
# ]

# _FROM_ADDRESS_ERROR_PHRASES = [
#     "given fromaddress not exists",
#     "fromaddress not exists",
#     "fromaddress is not recognized",
#     "from address not exists",
# ]

# _FROM_ADDRESS_ERROR_MSG = (
#     "Unable to send the email. Your Zoho sender address ({}) is not validated. "
#     "Please go to Zoho Mail Settings → Send Mail → validate your email address, then try again."
# )

# def _is_asking_for_folder(response) -> bool:
#     if getattr(response, "tool_calls", None):
#         return False
#     content = response.content if isinstance(response.content, str) else ""
#     lower   = content.lower()
#     return any(phrase in lower for phrase in _FOLDER_ASK_PHRASES)

# def _has_from_address_error(response) -> bool:
#     """Detect when the LLM is reporting a fromAddress error from a prior tool result."""
#     if getattr(response, "tool_calls", None):
#         return False
#     content = response.content if isinstance(response.content, str) else ""
#     lower   = content.lower()
#     return any(phrase in lower for phrase in _FROM_ADDRESS_ERROR_PHRASES)


# async def assistant_node(state: State) -> dict:
#     """LLM node that binds tools and generates a response."""
#     pipeline     = get_pipeline()
#     emp_code     = state.get("emp_code")
#     all_tools    = await get_all_tools_for_user(emp_code)
#     llm          = pipeline.vertex_llm.bind_tools(all_tools)
#     current_date = datetime.datetime.now().strftime("%A, %d %B %Y")
#     zoho_names   = [t.name for t in all_tools if "Zoho" in t.name or "zoho" in t.name]
#     print(f"[ASSISTANT] emp={emp_code} zoho_tools={zoho_names}")

#     zoho_account_id   = state.get("zoho_account_id")
#     zoho_from_address = state.get("zoho_from_address")
#     zoho_sender_name  = None

#     if not zoho_account_id:
#         zoho_account_id, zoho_from_address, zoho_sender_name = \
#             _extract_zoho_account(state.get("messages", []))

#     # Always fetch fresh from Zoho to get correct fromAddress
#     zoho_tool = next((t for t in all_tools if t.name == "ZohoMail_getMailAccounts"), None)
#     print(f"[ASSISTANT] zoho_tool_found={zoho_tool is not None} cached_account_id={zoho_account_id}")
#     if zoho_tool:
#         try:
#             result = await zoho_tool.ainvoke({"args": {}, "id": "prefetch"})
#             raw    = result.content if hasattr(result, "content") else result
#             print(f"[ZOHO_FETCH] FULL raw={str(raw)[:2000]}")
#             data   = json.loads(raw) if isinstance(raw, str) else raw
#             accts  = data.get("data", []) if isinstance(data, dict) else []
#             if accts:
#                 acct              = accts[0]
#                 zoho_account_id   = str(acct.get("accountId", ""))
#                 zoho_from_address = (
#                     acct.get("fromAddress")
#                     or acct.get("mailboxAddress")
#                     or (acct.get("sendMailDetails") or [{}])[0].get("fromAddress", "")
#                 )
#                 zoho_sender_name  = acct.get("displayName", "") or _name_from_email(zoho_from_address)
#                 print(f"[ZOHO_FETCH] accountId={zoho_account_id} fromAddress={zoho_from_address}")
#             else:
#                 print(f"[ZOHO_FETCH] no accounts: {data}")
#         except Exception as e:
#             print(f"[ZOHO_FETCH] FAILED: {e}")

#     if zoho_account_id and not zoho_sender_name:
#         zoho_sender_name = _name_from_email(zoho_from_address or "")

#     # ── No Zoho connection — tell user clearly instead of letting LLM hallucinate ──
#     if not zoho_account_id:
#         zoho_tool_exists = any(t.name == "ZohoMail_getMailAccounts" for t in all_tools)
#         if not zoho_tool_exists:
#             logger.warning("[assistant_node] No Zoho tools found for emp_code=%s", emp_code)
#             # Only block if this is clearly a mail request
#             last_human = next(
#                 (m.content.lower() for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
#                 ""
#             )
#             MAIL_WORDS = ["mail", "email", "draft", "send", "inbox", "reply", "compose"]
#             if any(w in last_human for w in MAIL_WORDS):
#                 return {"messages": [AIMessage(
#                     content="Your Zoho Mail account is not connected. Please ask your admin to configure the Zoho MCP key."
#                 )]}

#     history   = _smart_history(state["messages"])
#     has_human = any(isinstance(m, HumanMessage) for m in history)
#     if not has_human:
#         return {"messages": [AIMessage(content="I'm here to help! What would you like to know?")]}

#     # ── Intercept send confirmation — build tool call directly, no LLM ──
#     last_human_msg = next(
#         (m.content.strip().lower() for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
#         ""
#     )
#     if last_human_msg in ("yes", "y", "send it", "send", "confirm") and zoho_account_id:
#         draft = state.get("pending_email") or _extract_draft_from_messages(state["messages"])
#         if draft and draft.get("toAddress"):
#             print(f"[SEND_INTERCEPT] Building send call: to={draft['toAddress']} subject={draft['subject']}")
#             return {
#                 "messages": [AIMessage(
#                     content="",
#                     tool_calls=[{
#                         "id":   str(uuid.uuid4()),
#                         "name": "ZohoMail_sendEmail",
#                         "args": {
#                             "path_variables": {"accountId": zoho_account_id},
#                             "fromAddress":    zoho_from_address,
#                             "toAddress":      draft["toAddress"],
#                             "subject":        draft["subject"],
#                             "content":        draft["content"],
#                             "mailFormat":     "html",
#                         },
#                     }],
#                 )],
#                 "pending_email": None,
#             }

#     system_content = _build_system_prompt(
#         current_date,
#         zoho_account_id=zoho_account_id,
#         zoho_from_address=zoho_from_address,
#         zoho_sender_name=zoho_sender_name,
#     )

#     logger.info("[assistant_node] system_prompt=\n%s", system_content)
#     logger.info("[assistant_node] history_msgs=%s", [type(m).__name__ + ':' + str(m.content)[:60] for m in history])

#     messages = [SystemMessage(content=system_content)] + history
#     response = await llm.ainvoke(messages)

#     # ── Guard: LLM asked user for folder — force ZohoMail_getAllFolders (read only) ──
#     if _is_asking_for_folder(response) and zoho_account_id:
#         last_human = next(
#             (m.content.lower() for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
#             ""
#         )
#         SEND_WORDS = ["send", "draft", "compose", "write", "mail to", "email to"]
#         is_send_request = any(w in last_human for w in SEND_WORDS)
#         if not is_send_request:
#             logger.warning("[assistant_node] LLM asked user for folder — forcing ZohoMail_getAllFolders call")
#             return {"messages": [AIMessage(
#                 content="",
#                 tool_calls=[{
#                     "id":   str(uuid.uuid4()),
#                     "name": "ZohoMail_getAllFolders",
#                     "args": {
#                         "path_variables": {"accountId": zoho_account_id},
#                         "query_params":   {"fields": "folderId,folderName"},
#                     },
#                 }],
#             )]}

#     # ── Guard: fromAddress error — show clear actionable message, stop retrying ──
#     if _has_from_address_error(response):
#         return {"messages": [AIMessage(
#             content=_FROM_ADDRESS_ERROR_MSG.format(zoho_from_address or "unknown")
#         )]}

#     logger.info(
#         "[assistant_node] tool_calls=%s content=%s",
#         getattr(response, "tool_calls", None),
#         str(response.content)[:300],
#     )

#     # ── Persist zoho credentials + store draft if LLM just showed a preview ──
#     out: dict = {"messages": [response]}
#     if zoho_account_id:
#         out["zoho_account_id"]   = zoho_account_id
#         out["zoho_from_address"] = zoho_from_address
#     # If LLM just showed a draft preview, extract and store it
#     resp_content = response.content if isinstance(response.content, str) else ""
#     if "shall i send" in resp_content.lower() and not getattr(response, "tool_calls", None):
#         draft = _extract_draft_from_messages(state["messages"] + [response])
#         if draft:
#             out["pending_email"] = draft
#             print(f"[DRAFT_STORED] to={draft.get('toAddress')} subject={draft.get('subject')}")
#     return out


# def create_intent_driven_agent(checkpointer=None) -> StateGraph:
#     """Create a LangGraph agent with NeMo Guardrails integration.

#     Graph structure:
#         START -> orchestrator -> route_after_classification
#         -> assistant -> tools_condition -> tools -> assistant -> ...
#     """

#     _leave_subgraph_compiled = leave_subgraph(checkpointer=checkpointer)

#     graph = StateGraph(State)

#     # Add nodes
#     graph.add_node("orchestrator", Orchestrator)
#     graph.add_node("assistant", assistant_node)
#     graph.add_node("leave_node", _leave_subgraph_compiled)
#     graph.add_node("tools", DynamicToolNode())  # Use dynamic tool node

#     # Entry point
#     graph.add_edge(START, "orchestrator")


#     # After orchestrator, route based on intent
#     graph.add_conditional_edges(
#         "orchestrator",
#         route_after_classification,
#         {
#             "assistant": "assistant",
#             "Leave_Application": "leave_node",
#             END: END,
#         },
#     )

#     # Tool call loop: assistant -> tools -> assistant
#     graph.add_conditional_edges("assistant", tools_condition)
#     graph.add_edge("tools", "assistant")

#     return graph.compile(checkpointer=checkpointer)



import datetime
import json
import os
import uuid

from src.pipeline.leave_agent_node import leave_subgraph
os.environ["FASTEMBED_CACHE_PATH"] = "C:/fastembed_models"
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph, START
from nemoguardrails.rails.llm.options import GenerationOptions
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
    shared_tools = get_mcp_tools() + localTool

    if not emp_code:
        return shared_tools

    try:
        user_zoho_tools = await get_zoho_tools_for_user(emp_code)
    except Exception as e:
        logger.error(f"[emp:{emp_code}] Failed to load Zoho tools: {e}")
        user_zoho_tools = []

    if not user_zoho_tools:
        return shared_tools

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
        tools = await get_all_tools_for_user(emp_code)

        zoho_account_id = state.get("zoho_account_id")
        zoho_from_address = state.get("zoho_from_address")
        print(f"[TOOL_NODE] zoho_account_id={zoho_account_id} zoho_from_address={zoho_from_address}")

        if zoho_account_id:
            last_msg = state["messages"][-1] if state.get("messages") else None
            if last_msg and hasattr(last_msg, "tool_calls"):
                patched = []
                for tc in (last_msg.tool_calls or []):
                    name = tc.get("name", "")
                    if "Zoho" in name or "zoho" in name:
                        args = dict(tc.get("args", {}))

                        # FIX 1: Unwrap kwargs wrapper — handle both dict and JSON string
                        if list(args.keys()) == ["kwargs"]:
                            inner = args["kwargs"]
                            if isinstance(inner, str):
                                try:
                                    inner = json.loads(inner)
                                except Exception:
                                    pass
                            if isinstance(inner, dict):
                                args = inner

                        print(f"[TOOL_NODE] raw tool_call name={name} args={json.dumps(args, default=str)[:300]}")

                        # Ensure path_variables has accountId
                        pv = dict(args.get("path_variables") or {})
                        pv["accountId"] = zoho_account_id
                        args["path_variables"] = pv
                        args.pop("accountId", None)

                        qp = dict(args.get("query_params") or {})

                        # Only ZohoMail_getMessageContent requires folderId as a path variable.
                        # ZohoMail_listEmails keeps folderId in query_params.
                        if name.strip().lower() == "zohomail_getmessagecontent" and "folderId" in qp:
                            pv["folderId"] = qp.pop("folderId")
                        elif "folderId" in qp and name.strip().lower() != "zohomail_getmessagecontent":
                            logger.debug(
                                "[DynamicToolNode] leaving folderId in query_params for %s",
                                name
                            )

                        args["path_variables"] = pv

                        # Remove accountId from query_params if present
                        qp.pop("accountId", None)
                        if qp:
                            args["query_params"] = qp
                        else:
                            args.pop("query_params", None)

                        # FIX 2: Collect all body fields correctly
                        body = dict(args.get("body") or {})
                        BODY_FIELDS = {
                            "fromAddress", "toAddress", "subject", "content",
                            "mailFormat", "ccAddress", "bccAddress", "encoding",
                            "action", "mode"
                        }
                        # Promote flat body fields into body dict and remove from top-level
                        for f in list(BODY_FIELDS):
                            if f in args and f not in body:
                                body[f] = args.pop(f)

                        # Ensure fromAddress is set
                        if not body.get("fromAddress") and zoho_from_address:
                            body["fromAddress"] = zoho_from_address

                        # FIX 3: Normalise toAddress — handle list, dict, and plain string
                        to = body.get("toAddress")
                        if isinstance(to, list):
                            body["toAddress"] = ",".join(str(x) for x in to)
                        elif isinstance(to, dict):
                            body["toAddress"] = (
                                to.get("address") or to.get("email") or str(to)
                            )
                        # else: already a string, leave as-is

                        if not body.get("mailFormat") and name == "ZohoMail_sendEmail":
                            body["mailFormat"] = "html"

                        if name == "ZohoMail_getAllFolders":
                            qp = dict(args.get("query_params") or {})
                            qp["fields"] = "folderId,folderName"
                            args["query_params"] = qp

                        if name == "ZohoMail_listEmails":
                            qp = dict(args.get("query_params") or {})
                            qp["fields"] = "messageId,subject,sender,receivedTime"
                            # Normalize sortorder to a boolean for ZohoMail_listEmails.
                            if "sortOrder" in qp:
                                qp["sortorder"] = qp.pop("sortOrder")
                            if "sortorder" in qp:
                                value = qp["sortorder"]
                                if isinstance(value, str):
                                    normalized = value.strip().lower()
                                    if normalized in {"false", "0", "no", "off"}:
                                        qp["sortorder"] = False
                                    elif normalized in {"true", "1", "yes", "on"}:
                                        qp["sortorder"] = True
                                elif isinstance(value, int):
                                    qp["sortorder"] = bool(value)
                            args["query_params"] = qp

                        args["body"] = body
                        tc = {**tc, "args": args}
                    patched.append(tc)

                # FIX 4: Compare correctly — tool_calls may be a list of dicts, not raw list
                if patched != list(last_msg.tool_calls or []):
                    fixed_msg = AIMessage(
                        content=last_msg.content,
                        tool_calls=patched,
                    )
                    state = {**state, "messages": state["messages"][:-1] + [fixed_msg]}
                    logger.info("[DynamicToolNode] patched tool_calls: %s", patched)

        node = ToolNode(tools)
        return await node.ainvoke(state)


async def Orchestrator(state: State) -> State:
    if not (state.get("messages") and isinstance(state["messages"][-1], HumanMessage)):
        return {
            "intent": state.get("intent", "general"),
            "responded": False,
        }

    latest_user_msg = state["messages"][-1].content.strip()
    leave_step = state.get("leave_step")
    active_intent = state.get("active_intent", "general")

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

    EXPLICIT_MAIL_TRIGGERS = [
        "draft", "compose", "send mail", "send email",
        "write mail", "write email", "write a mail", "write an email",
        "send a mail", "send an email", "mail to", "email to",
        "shoot a mail", "shoot an email",
        "check mail", "check email", "show mail", "show email",
        "read mail", "read email", "recent mail", "recent email",
        "my inbox", "fetch mail", "fetch email",
        "reply to", "unread mail", "unread email",
    ]

    def is_explicit_leave(msg: str) -> bool:
        msg_lower = msg.lower().strip()
        return any(phrase in msg_lower for phrase in EXPLICIT_LEAVE_TRIGGERS)

    def is_explicit_mail(msg: str) -> bool:
        msg_lower = msg.lower().strip()
        return any(phrase in msg_lower for phrase in EXPLICIT_MAIL_TRIGGERS)

    # Case 1: Active interrupt — skip guardrail
    if leave_step in active_leave_steps:
        print(f">>> interrupt pending — skipping guardrail")
        return {"intent": "apply_leave", "responded": False}

    # Case 2: Explicit leave request
    if is_explicit_leave(latest_user_msg):
        print(f">>> explicit leave request detected — routing directly to leave_node")
        return {
            "intent": "apply_leave",
            "active_intent": "apply_leave",
            "responded": False,
        }

    # Case 2b: Explicit mail request
    if is_explicit_mail(latest_user_msg):
        print(f">>> explicit mail request detected — routing directly to assistant")
        return {
            "intent": "Assistant",
            "active_intent": "Assistant",
            "responded": False,
        }

    # Case 2c: Already in an assistant conversation — skip guardrail for follow-ups
    if active_intent == "Assistant":
        print(f">>> active_intent=Assistant — skipping guardrail for follow-up")
        return {
            "intent": "Assistant",
            "active_intent": "Assistant",
            "responded": False,
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
                "llm_calls": False,
                "internal_events": True,
                "colang_history": False,
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
    if output_text and output_text not in PASSTHROUGH_MESSAGES:
        print(f">>> Case 4 fired: output_text='{output_text}' detected_intent={detected_intent} active_intent={active_intent}")
        if active_intent == "Assistant" or detected_intent == "Assistant":
            print(">>> guardrail blocked but intent=Assistant — continuing to assistant")
            return {
                "intent": "Assistant",
                "active_intent": "Assistant",
                "responded": False,
            }
        print(f">>> BLOCKING with: {output_text}")
        return {
            "messages": [AIMessage(content=output_text)],
            "intent": detected_intent or active_intent,
            "responded": True,
        }

    # Case 5: apply_leave detected mid-assistant-conversation
    if detected_intent == "apply_leave" and active_intent == "Assistant":
        print(">>> non-explicit apply_leave during assistant conversation — staying in assistant")
        return {
            "intent": "Assistant",
            "active_intent": "Assistant",
            "responded": False,
        }

    # Case 6: Resolve final intent
    VAGUE_INTENTS = {"follow_up", "ask off topic", None}

    # FIX 5: Reset active intent properly when leave is done
    if leave_step in {"cancelled", "completed", "failed"}:
        previous_active_intent = "general"
    else:
        previous_active_intent = active_intent

    if detected_intent not in VAGUE_INTENTS:
        final_intent = detected_intent
        new_active_intent = detected_intent
    else:
        final_intent = previous_active_intent
        new_active_intent = previous_active_intent

    print(f"Final intent: {final_intent} | active_intent: {new_active_intent}")

    return {
        "intent": final_intent,
        "active_intent": new_active_intent,
        "responded": False,
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

    # FIX 6: Removed redundant duplicate `if leave_step in {...}` check;
    # consolidated into a single clean routing block
    intent = state.get("intent", "general")
    if intent == "apply_leave":
        return "Leave_Application"
    return "assistant"


def _smart_history(messages: list) -> list:
    """
    Return a window of messages that always includes:
    - The last HumanMessage and everything after it (tool calls + tool results)
    - Up to 4 previous messages for context
    Never cuts off mid tool-call/result cycle.
    """
    last_human_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break

    if last_human_idx is None:
        return messages[-6:]

    start = max(0, last_human_idx - 4)
    return messages[start:]


def _name_from_email(email: str) -> str:
    try:
        local = email.split("@")[0]
        return " ".join(p.capitalize() for p in local.split("."))
    except Exception:
        return email


def _extract_draft_from_messages(messages: list) -> dict | None:
    """Find the most recent AI draft message and extract to/subject/content."""
    import re
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content if isinstance(msg.content, str) else ""
        if "shall i send" not in content.lower():
            continue
        to = re.search(r"To:\s*([\w.@+\-]+)", content)
        subject = re.search(r"Subject:\s*(.+)", content)
        body_match = re.search(
            r"(?:Dear|Hi|Hello).+?(?=Shall I send)",
            content,
            re.DOTALL | re.IGNORECASE
        )
        return {
            "toAddress": to.group(1).strip() if to else "",
            "subject": subject.group(1).strip() if subject else "(no subject)",
            "content": body_match.group(0).strip() if body_match else content,
        }
    return None


def _build_system_prompt(
    current_date: str,
    zoho_account_id: str = None,
    zoho_from_address: str = None,
    zoho_sender_name: str = None,
) -> str:
    zoho_session = (
        f"ZOHO SESSION (already authenticated):\n"
        f"  accountId   = '{zoho_account_id}'\n"
        f"  fromAddress = '{zoho_from_address}'\n"
        f"  senderName  = '{zoho_sender_name}'\n"
        f"Use these directly. Do NOT call ZohoMail_getMailAccounts.\n"
        f"Always sign emails as '{zoho_sender_name}'.\n"
    ) if zoho_account_id else (
        "ZOHO MAIL: Call ZohoMail_getMailAccounts FIRST to get accountId and fromAddress "
        "before any mail operation. Never guess or hardcode these values.\n"
    )
    return (
        f"You are MACOM AI, an HR Assistant for MACOM employees.\n"
        f"TODAY'S DATE IS {current_date}. Use this directly — never ask the user for the date.\n\n"
        f"{zoho_session}\n"
        "CAPABILITIES — use the right tool for each task:\n"
        "- HR/Policy questions → call Policy_RAG_Implementation\n"
        "- Weather → call Current_Date_weather with city name\n"
        "- News → call Get_Top_News with category\n"
        "- Email (read/send/reply) → use Zoho tools per rules below\n\n"
        "EMAIL RULES:\n"
        "- NEVER ask the user for accountId, folderId, messageId or any technical ID — fetch them via tools\n"
        "- NEVER ask the user for subject or content if they already described the email — infer and draft it\n"
        "- folderId: always call ZohoMail_getAllFolders to get it, never use folder name as ID\n"
        "- ZohoMail_getAllFolders: include 'fields': 'folderId,folderName' in query_params\n"
        "- ZohoMail_listEmails: include 'fields': 'messageId,subject,sender,receivedTime' in query_params\n"
        "- folderId goes in query_params for other mail tools, never in path_variables\n"
        "- sortorder: boolean false, not string\n"
        "- SENDING: always show a full draft preview first and ask 'Shall I send this? (Yes/No)' — "
        "NEVER call ZohoMail_sendEmail before user confirms Yes\n"
        "- REPLYING: same — show draft, wait for Yes, then send\n"
    )


def _extract_zoho_account(messages: list) -> tuple:
    """Scan ToolMessage history for a prior ZohoMail_getMailAccounts result."""
    for msg in reversed(messages):
        # FIX 7: ToolMessage is now imported at the top — removed redundant local import
        if not isinstance(msg, ToolMessage) or msg.name != "ZohoMail_getMailAccounts":
            continue
        try:
            data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            accounts = data.get("data", []) if isinstance(data, dict) else []
            if accounts:
                acct = accounts[0]
                account_id = str(acct.get("accountId", ""))
                from_address = (
                    acct.get("fromAddress")
                    or acct.get("mailboxAddress")
                    or (acct.get("sendMailDetails") or [{}])[0].get("fromAddress", "")
                )
                sender_name = acct.get("displayName", "") or _name_from_email(from_address)
                return account_id, from_address, sender_name
        except Exception:
            pass
    return None, None, None


_FOLDER_ASK_PHRASES = [
    "folder id", "folderid", "which folder", "folder name",
    "inbox, sent", "inbox or sent", "e.g., inbox", "specify which folder",
    "please specify", "could you please specify",
]

_FROM_ADDRESS_ERROR_PHRASES = [
    "given fromaddress not exists",
    "fromaddress not exists",
    "fromaddress is not recognized",
    "from address not exists",
]

_FROM_ADDRESS_ERROR_MSG = (
    "Unable to send the email. Your Zoho sender address ({}) is not validated. "
    "Please go to Zoho Mail Settings → Send Mail → validate your email address, then try again."
)


def _is_asking_for_folder(response) -> bool:
    if getattr(response, "tool_calls", None):
        return False
    content = response.content if isinstance(response.content, str) else ""
    lower = content.lower()
    return any(phrase in lower for phrase in _FOLDER_ASK_PHRASES)


def _has_from_address_error(response) -> bool:
    """Detect when the LLM is reporting a fromAddress error from a prior tool result."""
    if getattr(response, "tool_calls", None):
        return False
    content = response.content if isinstance(response.content, str) else ""
    lower = content.lower()
    return any(phrase in lower for phrase in _FROM_ADDRESS_ERROR_PHRASES)


async def assistant_node(state: State) -> dict:
    """LLM node that binds tools and generates a response."""
    pipeline = get_pipeline()
    emp_code = state.get("emp_code")
    all_tools = await get_all_tools_for_user(emp_code)
    llm = pipeline.vertex_llm.bind_tools(all_tools)
    current_date = datetime.datetime.now().strftime("%A, %d %B %Y")
    zoho_names = [t.name for t in all_tools if "Zoho" in t.name or "zoho" in t.name]
    print(f"[ASSISTANT] emp={emp_code} zoho_tools={zoho_names}")

    zoho_account_id = state.get("zoho_account_id")
    zoho_from_address = state.get("zoho_from_address")
    zoho_sender_name = None

    if not zoho_account_id:
        zoho_account_id, zoho_from_address, zoho_sender_name = \
            _extract_zoho_account(state.get("messages", []))

    # FIX 8: Only prefetch Zoho account if not already cached in state,
    # avoiding a redundant API call every single turn
    if not zoho_account_id:
        zoho_tool = next((t for t in all_tools if t.name == "ZohoMail_getMailAccounts"), None)
        print(f"[ASSISTANT] zoho_tool_found={zoho_tool is not None}")
        if zoho_tool:
            try:
                result = await zoho_tool.ainvoke({"args": {}, "id": "prefetch"})
                raw = result.content if hasattr(result, "content") else result
                print(f"[ZOHO_FETCH] FULL raw={str(raw)[:2000]}")
                data = json.loads(raw) if isinstance(raw, str) else raw
                accts = data.get("data", []) if isinstance(data, dict) else []
                if accts:
                    acct = accts[0]
                    zoho_account_id = str(acct.get("accountId", ""))
                    zoho_from_address = (
                        acct.get("fromAddress")
                        or acct.get("mailboxAddress")
                        or (acct.get("sendMailDetails") or [{}])[0].get("fromAddress", "")
                    )
                    zoho_sender_name = acct.get("displayName", "") or _name_from_email(zoho_from_address)
                    print(f"[ZOHO_FETCH] accountId={zoho_account_id} fromAddress={zoho_from_address}")
                else:
                    print(f"[ZOHO_FETCH] no accounts: {data}")
            except Exception as e:
                print(f"[ZOHO_FETCH] FAILED: {e}")
    else:
        print(f"[ASSISTANT] using cached zoho_account_id={zoho_account_id}")

    if zoho_account_id and not zoho_sender_name:
        zoho_sender_name = _name_from_email(zoho_from_address or "")

    # FIX 9: Mail-block check — only block if Zoho tools are completely absent
    # (previously this could block even when tools existed but fetch temporarily failed)
    if not zoho_account_id:
        zoho_tool_exists = any(t.name == "ZohoMail_getMailAccounts" for t in all_tools)
        if not zoho_tool_exists:
            logger.warning("[assistant_node] No Zoho tools found for emp_code=%s", emp_code)
            last_human = next(
                (m.content.lower() for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
                ""
            )
            MAIL_WORDS = ["mail", "email", "draft", "send", "inbox", "reply", "compose"]
            if any(w in last_human for w in MAIL_WORDS):
                return {"messages": [AIMessage(
                    content="Your Zoho Mail account is not connected. Please ask your admin to configure the Zoho MCP key."
                )]}

    history = _smart_history(state["messages"])
    has_human = any(isinstance(m, HumanMessage) for m in history)
    if not has_human:
        return {"messages": [AIMessage(content="I'm here to help! What would you like to know?")]}

    # Intercept send confirmation — build tool call directly, no LLM
    last_human_msg = next(
        (m.content.strip().lower() for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    last_human_index = next(
        (i for i in range(len(state["messages"]) - 1, -1, -1)
         if isinstance(state["messages"][i], HumanMessage)),
        None
    )
    has_messages_after_last_human = (
        last_human_index is not None
        and len(state["messages"]) - last_human_index - 1 > 0
    )
    if last_human_msg in ("yes", "y", "send it", "send", "confirm") and zoho_account_id and not has_messages_after_last_human:
        draft = state.get("pending_email") or _extract_draft_from_messages(state["messages"])
        if draft and draft.get("toAddress"):
            print(f"[SEND_INTERCEPT] Building send call: to={draft['toAddress']} subject={draft['subject']}")
            return {
                "messages": [AIMessage(
                    content="",
                    tool_calls=[{
                        "id": str(uuid.uuid4()),
                        "name": "ZohoMail_sendEmail",
                        "args": {
                            "path_variables": {"accountId": zoho_account_id},
                            "body": {
                                # FIX 10: All send fields go inside body dict, not at top-level,
                                # consistent with how DynamicToolNode patches tool calls
                                "fromAddress": zoho_from_address,
                                "toAddress": draft["toAddress"],
                                "subject": draft["subject"],
                                "content": draft["content"],
                                "mailFormat": "html",
                            },
                        },
                    }],
                )],
                "pending_email": None,
            }

    system_content = _build_system_prompt(
        current_date,
        zoho_account_id=zoho_account_id,
        zoho_from_address=zoho_from_address,
        zoho_sender_name=zoho_sender_name,
    )

    logger.info("[assistant_node] system_prompt=\n%s", system_content)
    logger.info(
        "[assistant_node] history_msgs=%s",
        [type(m).__name__ + ':' + str(m.content)[:60] for m in history]
    )

    messages = [SystemMessage(content=system_content)] + history
    response = await llm.ainvoke(messages)

    # Guard: LLM asked user for folder — force ZohoMail_getAllFolders
    if _is_asking_for_folder(response) and zoho_account_id:
        last_human = next(
            (m.content.lower() for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
            ""
        )
        SEND_WORDS = ["send", "draft", "compose", "write", "mail to", "email to"]
        is_send_request = any(w in last_human for w in SEND_WORDS)
        if not is_send_request:
            logger.warning("[assistant_node] LLM asked user for folder — forcing ZohoMail_getAllFolders call")
            return {"messages": [AIMessage(
                content="",
                tool_calls=[{
                    "id": str(uuid.uuid4()),
                    "name": "ZohoMail_getAllFolders",
                    "args": {
                        "path_variables": {"accountId": zoho_account_id},
                        "body": {"fields": "folderId,folderName"},
                    },
                }],
            )]}

    # Guard: fromAddress error — show clear actionable message
    if _has_from_address_error(response):
        return {"messages": [AIMessage(
            content=_FROM_ADDRESS_ERROR_MSG.format(zoho_from_address or "unknown")
        )]}

    logger.info(
        "[assistant_node] tool_calls=%s content=%s",
        getattr(response, "tool_calls", None),
        str(response.content)[:300],
    )

    # Persist zoho credentials + store draft if LLM just showed a preview
    out: dict = {"messages": [response]}
    if zoho_account_id:
        out["zoho_account_id"] = zoho_account_id
        out["zoho_from_address"] = zoho_from_address
    resp_content = response.content if isinstance(response.content, str) else ""
    if "shall i send" in resp_content.lower() and not getattr(response, "tool_calls", None):
        draft = _extract_draft_from_messages(state["messages"] + [response])
        if draft:
            out["pending_email"] = draft
            print(f"[DRAFT_STORED] to={draft.get('toAddress')} subject={draft.get('subject')}")
    return out


def create_intent_driven_agent(checkpointer=None) -> StateGraph:
    """Create a LangGraph agent with NeMo Guardrails integration."""

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
            "assistant": "assistant",
            "Leave_Application": "leave_node",
            END: END,
        },
    )

    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile(checkpointer=checkpointer)