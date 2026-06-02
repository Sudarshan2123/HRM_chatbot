"""
comp_leave_node.py
──────────────────
Compensatory leave subgraph.

Flow:
  1. GET  /api/CompLeave/{emp_code}   — fetch available compensatory entries
  2. If empty  → cancel.
     If one    → show details + yes/no confirmation.
     If many   → show numbered list, user picks ONE (selecting = confirmation).
  3. Ask for leave date  (single day, validated against expiry).
  4. Ask for reason      (free text).
  5. Show summary + confirm.
  6. POST /api/CompLeaveApply.

KEY DESIGN RULE:
  Each node fires AT MOST ONE interrupt().
  comp_fetch_node  → one interrupt (yes/no for single, selection for multiple).
  comp_select_node → one interrupt (only re-asks if selection was invalid).
  comp_date_node   → one interrupt.
  comp_reason_node → one interrupt.
  comp_submit_node → one interrupt.
"""

import datetime
import logging
import os
import requests

from langgraph.types import interrupt
from langgraph.graph import END, START, StateGraph
from statenode import State as CompState
from langchain_core.messages import AIMessage, HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LEAVE_BASE_URL = os.getenv("LEAVE_BASE_URL", "https://macserv.mactech.net.in/chatbotleaveapi")


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

def call_api(method: str, url: str, **kwargs) -> object:
    kwargs.setdefault("verify", True)  # SECURITY: Always verify SSL certificates
    headers = {"Content-Type": "application/json"}
    response = requests.request(method, url, headers=headers, timeout=30, **kwargs)
    if not response.ok:
        try:
            err_body = response.json()
        except Exception:
            err_body = response.text
        logger.error(
            "[comp_api] %s %s → HTTP %s | request=%s | response=%s",
            method, url, response.status_code,
            kwargs.get("json", "N/A"), err_body,
        )
        response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _parse_api_date(date_str: str) -> datetime.date | None:
    try:
        return datetime.datetime.fromisoformat(date_str.rstrip("Z")).date()
    except Exception:
        return None


def _fmt_display_date(date_str: str) -> str:
    """'2023-06-28T00:00:00' → '28/MAY/2026'"""
    d = _parse_api_date(date_str)
    return d.strftime("%d/%b/%Y").upper() if d else date_str


def _fmt_api_date(date_obj: datetime.date) -> str:
    """datetime.date → '11-MAY-2026'"""
    return date_obj.strftime("%d-%b-%Y").upper()


def _parse_user_date(date_str: str) -> datetime.date | None:
    try:
        return datetime.datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except Exception:
        return None


def _entry_display(e: dict, idx: int | None = None) -> str:
    prefix    = f"{idx}. " if idx is not None else ""
    comp_date = _fmt_display_date(e.get("compDate", ""))
    exp_date  = _fmt_display_date(e.get("expDate",  ""))
    return (
        f"{prefix}──────────────────────────────\n"
        f"  Compensatory Date : {comp_date}\n"
        f"  Compensatory Name : {e.get('compName', '')}\n"
        f"  State             : {e.get('stateName', '')}\n"
        f"  Expiry Date       : {exp_date}"
    )


def _build_option_label(e: dict) -> str:
    """Short label used for the clickable card buttons in the frontend."""
    return (
        f"{e.get('compName', 'Entry')} "
        f"(Comp: {_fmt_display_date(e.get('compDate', ''))}, "
        f"Exp: {_fmt_display_date(e.get('expDate', ''))})"
    )


# ---------------------------------------------------------------------------
# Node 1 — Fetch + first interrupt
# ---------------------------------------------------------------------------

def comp_fetch_node(state: CompState) -> dict:
    """
    Fetch entries and fire ONE interrupt:
      • Single entry  → yes/no confirmation interrupt.
                        On 'yes' → directly set comp_id/name/exp + step=awaiting_date.
                        On 'no'  → cancel.
      • Multi entries → numbered selection interrupt (picking = confirming).
                        Store entries in state for comp_select_node.
    """
    emp_code = state["emp_code"]

    try:
        result = call_api("GET", f"{LEAVE_BASE_URL}/api/CompLeave/{emp_code}")
        logger.info("[CompLeave] response: %s", result)
    except Exception as e:
        logger.error("[CompLeave] fetch error: %s", e)
        return {
            "comp_step": "failed",
            "messages": [AIMessage(
                content="Unable to fetch compensatory leave details. Please try again later."
            )],
        }

    entries: list[dict] = result if isinstance(result, list) else result.get("data", [])

    # ── No entries ────────────────────────────────────────────────────────────
    if not entries:
        return {
            "comp_step": "cancelled",
            "messages": [AIMessage(
                content="You have no compensatory leave available to apply at this time."
            )],
        }

    # ── Single entry — one yes/no interrupt ───────────────────────────────────
    if len(entries) == 1:
        e   = entries[0]
        msg = (
            f"Here is your available compensatory leave:\n\n"
            f"{_entry_display(e)}\n\n"
            f"Would you like to apply using this compensatory? (yes / no)"
        )
        decision = interrupt({"message": msg, "action": "comp_proceed_confirmation"})

        if str(decision).strip().lower() not in ("yes", "y"):
            return {
                "comp_step": "cancelled",
                "messages": [
                    AIMessage(content=msg),
                    HumanMessage(content=str(decision)),
                    AIMessage(content="Okay! Let me know if you need anything else."),
                ],
            }

        # ✅ Confirmed — go directly to date, no extra interrupt
        return {
            "comp_id":       e["compId"],
            "comp_name":     e.get("compName", ""),
            "comp_exp_date": e.get("expDate", ""),
            "comp_step":     "awaiting_date",
            "comp_entries":  [],            # clear — not needed
            "messages": [
                AIMessage(content=msg),
                HumanMessage(content=str(decision)),
                AIMessage(content=f"Compensatory selected: {e.get('compName', '')}. Now let's set the leave date."),
            ],
        }

    # ── Multiple entries — ONE selection interrupt ────────────────────────────
    # User picks a number → that IS the confirmation. No second yes/no needed.
    display_lines = "\n\n".join(
        _entry_display(e, idx=i + 1) for i, e in enumerate(entries)
    )
    option_labels = [_build_option_label(e) for e in entries]
    option_values = [str(i + 1) for i in range(len(entries))]

    msg = (
        f"You have {len(entries)} compensatory leaves available.\n"
        f"Please select one to apply (you can only apply one at a time):\n\n"
        f"{display_lines}"
    )

    selection = interrupt({
        "message": msg,
        "action":  "comp_selection",
        "options": option_labels,
        "values":  option_values,
    })

    # Store entries so comp_select_node can resolve without another API call
    return {
        "comp_step":              "awaiting_comp_selection",
        "comp_entries":           entries,
        "comp_pending_selection": str(selection).strip(),
        "messages": [
            AIMessage(content=msg),
            HumanMessage(content=str(selection)),
        ],
    }


# ---------------------------------------------------------------------------
# Node 1b — Resolve multi-entry selection (NO confirmation interrupt here)
# ---------------------------------------------------------------------------

def comp_select_node(state: CompState) -> dict:
    """
    Resolve the pending numeric selection from comp_fetch_node.
    On valid pick → store comp_id/name/exp, move to awaiting_date immediately.
    On invalid    → re-ask with ONE interrupt (same comp_selection action).

    IMPORTANT: Never fires a yes/no here. Selecting a number = confirmation.
    """
    entries: list[dict] = state.get("comp_entries") or []
    sel_str: str        = str(state.get("comp_pending_selection", "")).strip()

    logger.info("[comp_select] sel='%s' entries=%d", sel_str, len(entries))

    if not entries:
        # Should not happen — defensive guard
        return {
            "comp_step": "failed",
            "messages": [AIMessage(
                content="Session data was lost. Please start the leave application again."
            )],
        }

    option_labels = [_build_option_label(e) for e in entries]
    option_values = [str(i + 1) for i in range(len(entries))]
    chosen: dict | None = None

    # Try numeric index (1-based)
    if sel_str.isdigit():
        idx = int(sel_str) - 1
        if 0 <= idx < len(entries):
            chosen = entries[idx]

    # Try full label match (sent by JS card click)
    if chosen is None:
        for i, label in enumerate(option_labels):
            if sel_str.lower() == label.lower():
                chosen = entries[i]
                break

    # ── Invalid — ONE re-ask interrupt ───────────────────────────────────────
    if chosen is None:
        display_lines = "\n\n".join(
            _entry_display(e, idx=i + 1) for i, e in enumerate(entries)
        )
        error_msg = (
            f"Invalid selection. Please enter a number between 1 and {len(entries)}.\n\n"
            f"{display_lines}"
        )
        new_sel = interrupt({
            "message": error_msg,
            "action":  "comp_selection",
            "options": option_labels,
            "values":  option_values,
        })
        return {
            "comp_step":              "awaiting_comp_selection",
            "comp_entries":           entries,   # keep entries for next round
            "comp_pending_selection": str(new_sel).strip(),
            "messages": [
                AIMessage(content=error_msg),
                HumanMessage(content=str(new_sel)),
            ],
        }

    # ── Valid selection → proceed directly to date ────────────────────────────
    chosen_name = chosen.get("compName", "")
    chosen_exp  = _fmt_display_date(chosen.get("expDate", ""))
    logger.info("[comp_select] resolved: compId=%s name='%s'", chosen["compId"], chosen_name)

    return {
        "comp_id":                chosen["compId"],
        "comp_name":              chosen_name,
        "comp_exp_date":          chosen.get("expDate", ""),
        "comp_step":              "awaiting_date",
        "comp_entries":           [],    # clear — no longer needed
        "comp_pending_selection": "",
        "messages": [
            AIMessage(
                content=f"Selected: {chosen_name} (Expiry: {chosen_exp}). Now let's set the leave date."
            ),
        ],
    }


# ---------------------------------------------------------------------------
# Node 2 — Date input  (single day, validated)
# ---------------------------------------------------------------------------

def comp_date_node(state: CompState) -> dict:
    last_messages = state.get("messages", [])
    error_prefix  = ""
    if last_messages:
        last = last_messages[-1]
        if isinstance(last, AIMessage) and any(
            kw in str(last.content)
            for kw in ["Sunday", "Invalid", "past", "expired", "before"]
        ):
            error_prefix = f"{last.content}\n\n"

    comp_name = state.get("comp_name", "compensatory")
    exp_str   = state.get("comp_exp_date", "")
    exp_date  = _parse_api_date(exp_str)
    exp_disp  = _fmt_display_date(exp_str) if exp_str else "N/A"

    msg = (
        f"{error_prefix}"
        f"Enter the date you want to take compensatory leave for *{comp_name}*\n"
        f"(Expiry: {exp_disp})  —  format: YYYY-MM-DD:"
    )

    decision = interrupt({"message": msg, "action": "comp_date_input"})
    date_str  = str(decision).strip()
    chosen    = _parse_user_date(date_str)

    if chosen is None:
        return {
            "comp_step": "awaiting_date",
            "messages": [
                AIMessage(content=msg),
                HumanMessage(content=date_str),
                AIMessage(content="Invalid format. Please use YYYY-MM-DD (e.g. 2026-05-12)."),
            ],
        }

    today = datetime.date.today()

    if chosen < today:
        return {
            "comp_step": "awaiting_date",
            "messages": [
                AIMessage(content=msg),
                HumanMessage(content=date_str),
                AIMessage(content=f"{chosen} is in the past. Please enter a future date."),
            ],
        }

    if chosen.weekday() == 6:
        return {
            "comp_step": "awaiting_date",
            "messages": [
                AIMessage(content=msg),
                HumanMessage(content=date_str),
                AIMessage(
                    content=f"{chosen.strftime('%A, %d %B %Y')} is a Sunday and not a working day. "
                            "Please enter a valid working day (Monday to Saturday)."
                ),
            ],
        }

    if exp_date and chosen > exp_date:
        return {
            "comp_step": "awaiting_date",
            "messages": [
                AIMessage(content=msg),
                HumanMessage(content=date_str),
                AIMessage(
                    content=f"The selected date {chosen} is after the compensatory expiry "
                            f"({exp_disp}). Please choose a date on or before the expiry."
                ),
            ],
        }

    api_date = _fmt_api_date(chosen)
    return {
        "comp_leave_date":     date_str,
        "comp_leave_date_api": api_date,
        "comp_step":           "awaiting_reason",
        "messages": [
            AIMessage(content=msg),
            HumanMessage(content=date_str),
            AIMessage(content=f"Leave date set to: {api_date}"),
        ],
    }


# ---------------------------------------------------------------------------
# Node 3 — Reason input
# ---------------------------------------------------------------------------

def comp_reason_node(state: CompState) -> dict:
    comp_name  = state.get("comp_name", "compensatory")
    prompt_msg = (
        f"Compensatory: *{comp_name}*\n\n"
        "Please type the reason for taking this compensatory leave:"
    )
    raw_reason = interrupt({"message": prompt_msg, "action": "comp_reason_input"})
    raw_reason = str(raw_reason).strip()

    if not raw_reason:
        return {
            "comp_step": "awaiting_reason",
            "messages": [
                AIMessage(content=prompt_msg),
                AIMessage(content="Reason cannot be empty. Please provide a brief description."),
            ],
        }

    return {
        "comp_reason": raw_reason,
        "comp_step":   "awaiting_submission",
        "messages": [
            AIMessage(content=prompt_msg),
            HumanMessage(content=raw_reason),
            AIMessage(content=f"Reason noted: {raw_reason}"),
        ],
    }


# ---------------------------------------------------------------------------
# Node 4 — Summary + Submit
# ---------------------------------------------------------------------------

def comp_submit_node(state: CompState) -> dict:
    comp_name  = state.get("comp_name", "N/A")
    leave_date = state.get("comp_leave_date_api", "N/A")
    reason     = state.get("comp_reason", "N/A")
    exp_disp   = _fmt_display_date(state.get("comp_exp_date", ""))

    summary = (
        f"─── Compensatory Leave Summary ───\n"
        f"  Compensatory : {comp_name}\n"
        f"  Leave Date   : {leave_date}\n"
        f"  Expiry Date  : {exp_disp}\n"
        f"  Reason       : {reason}\n"
        f"───────────────────────────────────\n\n"
        f"Confirm submission? (yes / no)"
    )

    conf          = interrupt({"message": summary, "action": "comp_final_confirmation"})
    user_decision = str(conf).strip().lower()

    if user_decision not in ("yes", "y"):
        return {
            "comp_step": "cancelled",
            "messages": [
                AIMessage(content=summary),
                HumanMessage(content=str(conf)),
                AIMessage(content="Compensatory leave application cancelled. Let me know if you need anything else."),
            ],
        }

    payload = {
        "empCode":   int(state["emp_code"]),
        "leaveDate": leave_date,
        "compId":    int(state.get("comp_id", 0)),
        "reason":    reason,
        "email":     "",
    }
    logger.info("[comp_submit] payload=%s", payload)

    try:
        result    = call_api("POST", f"{LEAVE_BASE_URL}/api/CompLeaveApply", json=payload)
        logger.info("[comp_submit] API response=%s", result)

        api_msg   = str(result.get("message", ""))
        succeeded = result.get("succeeded")
        success   = succeeded is True or any(
            kw in api_msg.upper() for kw in ["APPLIED", "SUCCESS", "APPROVED"]
        )
        failure   = succeeded is False or any(
            kw in api_msg.upper() for kw in ["FAILED", "ERROR", "INVALID", "REJECTED"]
        )

        if success and not failure:
            status   = "completed"
            res_text = f"✅ {api_msg}" if api_msg else "✅ Compensatory leave applied successfully!"
        elif failure:
            status   = "failed"
            res_text = f"❌ {api_msg}" if api_msg else "❌ Submission failed."
        else:
            status   = "completed"
            res_text = f"✅ {api_msg}" if api_msg else "✅ Compensatory leave applied successfully!"

        return {
            "comp_step": status,
            "response":  res_text,
            "messages": [
                AIMessage(content=summary),
                HumanMessage(content=str(conf)),
                AIMessage(content=res_text),
            ],
        }

    except Exception as e:
        logger.error("[comp_submit] Exception: %s", e, exc_info=True)
        return {
            "comp_step": "failed",
            "messages": [AIMessage(content=f"A system error occurred: {str(e)}")],
        }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_comp(state: CompState) -> str:
    step = state.get("comp_step")
    if step in ("completed", "cancelled", "failed"):
        return END
    return {
        "awaiting_comp_selection": "comp_select",
        "awaiting_date":           "comp_date",
        "awaiting_reason":         "comp_reason",
        "awaiting_submission":     "comp_submit",
    }.get(step, "comp_fetch")


# ---------------------------------------------------------------------------
# Subgraph factory
# ---------------------------------------------------------------------------

def comp_leave_subgraph(checkpointer=None) -> StateGraph:
    sg = StateGraph(CompState)

    sg.add_node("comp_fetch",  comp_fetch_node)
    sg.add_node("comp_select", comp_select_node)
    sg.add_node("comp_date",   comp_date_node)
    sg.add_node("comp_reason", comp_reason_node)
    sg.add_node("comp_submit", comp_submit_node)

    sg.add_edge(START, "comp_fetch")

    for node in ("comp_fetch", "comp_select", "comp_date", "comp_reason", "comp_submit"):
        sg.add_conditional_edges(node, route_comp)

    return sg.compile(checkpointer=checkpointer)


# """
# comp_leave_node.py
# ──────────────────
# Compensatory leave subgraph.

# Flow:
#   1. GET  /api/CompLeave/{emp_code}   — fetch available compensatory entries
#   2. Show ALL entries to user; user picks one (or sees "none available")
#   3. Ask for leave date  (single day only — validated against comp expiry)
#   4. Ask for reason      (free text)
#   5. Confirm summary
#   6. POST /api/CompLeaveApply

# Kept in a separate file intentionally — compensatory leave has a completely
# different API, state shape, and validation from regular leave.
# """

# import datetime
# import logging
# import os
# import requests
# import urllib3

# from langgraph.types import interrupt
# from langgraph.graph import END, START, StateGraph
# from statenode import State as CompState
# from langchain_core.messages import AIMessage, HumanMessage

# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# LEAVE_BASE_URL = os.getenv("LEAVE_BASE_URL", "https://macserv.mactech.net.in/chatbotleaveapi")


# # ---------------------------------------------------------------------------
# # API helper
# # ---------------------------------------------------------------------------

# def call_api(method: str, url: str, **kwargs) -> object:
#     kwargs.setdefault("verify", False)
#     headers = {"Content-Type": "application/json"}
#     response = requests.request(method, url, headers=headers, timeout=30, **kwargs)
#     if not response.ok:
#         try:
#             err_body = response.json()
#         except Exception:
#             err_body = response.text
#         logger.error(
#             "[comp_api] %s %s → HTTP %s\n  request : %s\n  response: %s",
#             method, url, response.status_code,
#             kwargs.get("json", "N/A"),
#             err_body,
#         )
#         response.raise_for_status()
#     return response.json()


# # ---------------------------------------------------------------------------
# # Date helpers
# # ---------------------------------------------------------------------------

# def _parse_api_date(date_str: str) -> datetime.date | None:
#     """Parse ISO date string from API  e.g. '2023-06-28T00:00:00'."""
#     try:
#         return datetime.datetime.fromisoformat(date_str.rstrip("Z")).date()
#     except Exception:
#         return None


# def _fmt_display_date(date_str: str) -> str:
#     """'2023-06-28T00:00:00'  →  '28/JUN/2023'"""
#     d = _parse_api_date(date_str)
#     return d.strftime("%d/%b/%Y").upper() if d else date_str


# def _fmt_api_date(date_obj: datetime.date) -> str:
#     """datetime.date  →  '11-MAY-2026'  (format required by CompLeaveApply)"""
#     return date_obj.strftime("%d-%b-%Y").upper()


# def _parse_user_date(date_str: str) -> datetime.date | None:
#     """Accept YYYY-MM-DD from the date picker widget."""
#     try:
#         return datetime.datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
#     except Exception:
#         return None


# def _entry_display(e: dict, idx: int | None = None) -> str:
#     """Format a single compensatory entry for display."""
#     prefix    = f"{idx}. " if idx is not None else ""
#     comp_date = _fmt_display_date(e.get("compDate", ""))
#     exp_date  = _fmt_display_date(e.get("expDate",  ""))
#     return (
#         f"{prefix}──────────────────────────────\n"
#         f"  Compensatory Date : {comp_date}\n"
#         f"  Compensatory Name : {e.get('compName', '')}\n"
#         f"  State             : {e.get('stateName', '')}\n"
#         f"  Expiry Date       : {exp_date}"
#     )


# # ---------------------------------------------------------------------------
# # Node 1 — Fetch available compensatory entries
# # ---------------------------------------------------------------------------

# def comp_fetch_node(state: CompState) -> dict:
#     """
#     GET /api/CompLeave/{emp_code}

#     • Empty  → inform user and cancel.
#     • One    → show detail card, ask to proceed (yes / no).
#     • Many   → show ALL numbered cards, ask user to pick one.

#     The fetched entries list is stored in state["comp_entries"] so that
#     comp_select_node can resolve the user's choice without a second API call.
#     """
#     emp_code = state["emp_code"]

#     try:
#         result = call_api("GET", f"{LEAVE_BASE_URL}/api/CompLeave/{emp_code}")
#         logger.info("[CompLeave] response: %s", result)
#     except Exception as e:
#         logger.error("[CompLeave] fetch error: %s", e)
#         return {
#             "comp_step": "failed",
#             "messages": [AIMessage(
#                 content="Unable to fetch compensatory leave details. Please try again later."
#             )],
#         }

#     entries: list[dict] = result if isinstance(result, list) else result.get("data", [])

#     # ── No compensatory available ─────────────────────────────────────────────
#     if not entries:
#         return {
#             "comp_step": "cancelled",
#             "messages": [AIMessage(
#                 content="You have no compensatory leave available to apply at this time."
#             )],
#         }

#     # ── Single entry — show card and ask to proceed ───────────────────────────
#     if len(entries) == 1:
#         e   = entries[0]
#         msg = (
#             f"Here is your available compensatory leave:\n\n"
#             f"{_entry_display(e)}\n\n"
#             f"Would you like to apply using this compensatory? (yes / no)"
#         )
#         decision = interrupt({
#             "message": msg,
#             "action":  "comp_proceed_confirmation",
#         })
#         user_input = str(decision).strip().lower()

#         if user_input not in ("yes", "y"):
#             return {
#                 "comp_step": "cancelled",
#                 "messages": [
#                     AIMessage(content=msg),
#                     HumanMessage(content=str(decision)),
#                     AIMessage(content="Okay! Let me know if you need anything else."),
#                 ],
#             }

#         return {
#             "comp_id":        e["compId"],
#             "comp_name":      e.get("compName", ""),
#             "comp_exp_date":  e.get("expDate", ""),
#             "comp_step":      "awaiting_date",
#             "messages": [
#                 AIMessage(content=msg),
#                 HumanMessage(content=str(decision)),
#                 AIMessage(content=f"Compensatory selected: {e.get('compName', '')}"),
#             ],
#         }

#     # ── Multiple entries — show ALL with numbers, ask user to pick one ─────────
#     display_lines = "\n\n".join(
#         _entry_display(e, idx=i + 1) for i, e in enumerate(entries)
#     )
#     option_labels = [
#         f"{e.get('compName', 'Entry')} "
#         f"(Comp: {_fmt_display_date(e.get('compDate', ''))}, "
#         f"Exp: {_fmt_display_date(e.get('expDate', ''))})"
#         for e in entries
#     ]
#     option_values = [str(i + 1) for i in range(len(entries))]

#     msg = (
#         f"You have {len(entries)} compensatory leaves available.\n"
#         f"Please select one to apply (only one at a time):\n\n"
#         f"{display_lines}\n\n"
#         f"Enter the number (1–{len(entries)}) of the compensatory you want to apply:"
#     )

#     selection = interrupt({
#         "message": msg,
#         "action":  "comp_selection",
#         "options": option_labels,
#         "values":  option_values,
#     })

#     # Store fetched entries in state so comp_select_node can resolve without
#     # another API call.  We serialise them as a simple list of dicts.
#     return {
#         "comp_step":    "awaiting_comp_selection",
#         "comp_entries": entries,          # <-- persisted for comp_select_node
#         "comp_pending_selection": str(selection).strip(),
#         "messages": [
#             AIMessage(content=msg),
#             HumanMessage(content=str(selection)),
#         ],
#     }


# # ---------------------------------------------------------------------------
# # Node 1b — Resolve the user's selection from the stored entries list
# # ---------------------------------------------------------------------------

# def comp_select_node(state: CompState) -> dict:
#     """
#     Resolve the pending selection made in comp_fetch_node against the
#     already-fetched comp_entries list.  No second API call needed.

#     Accepts:
#       • A digit string "1" … "N"
#       • A label string matching one of the option_labels
#     """
#     entries: list[dict] = state.get("comp_entries", [])
#     sel_str: str        = state.get("comp_pending_selection", "").strip()

#     option_labels = [
#         f"{e.get('compName', 'Entry')} "
#         f"(Comp: {_fmt_display_date(e.get('compDate', ''))}, "
#         f"Exp: {_fmt_display_date(e.get('expDate', ''))})"
#         for e in entries
#     ]

#     chosen: dict | None = None

#     # Numeric selection
#     if sel_str.isdigit():
#         idx = int(sel_str) - 1
#         if 0 <= idx < len(entries):
#             chosen = entries[idx]

#     # Label match (case-insensitive)
#     if chosen is None:
#         for i, label in enumerate(option_labels):
#             if sel_str.lower() == label.lower():
#                 chosen = entries[i]
#                 break

#     # ── Invalid — re-ask (re-display all entries and prompt again) ────────────
#     if chosen is None:
#         display_lines = "\n\n".join(
#             _entry_display(e, idx=i + 1) for i, e in enumerate(entries)
#         )
#         option_values = [str(i + 1) for i in range(len(entries))]
#         error_msg = (
#             f"Invalid selection '{sel_str}'. "
#             f"Please enter a number between 1 and {len(entries)}.\n\n"
#             f"{display_lines}\n\n"
#             f"Enter the number (1–{len(entries)}):"
#         )
#         new_selection = interrupt({
#             "message": error_msg,
#             "action":  "comp_selection",
#             "options": option_labels,
#             "values":  option_values,
#         })
#         return {
#             "comp_step":              "awaiting_comp_selection",
#             "comp_pending_selection": str(new_selection).strip(),
#             "messages": [
#                 AIMessage(content=error_msg),
#                 HumanMessage(content=str(new_selection)),
#             ],
#         }

#     # ── Valid selection ───────────────────────────────────────────────────────
#     return {
#         "comp_id":       chosen["compId"],
#         "comp_name":     chosen.get("compName", ""),
#         "comp_exp_date": chosen.get("expDate", ""),
#         "comp_step":     "awaiting_date",
#         # Clear transient selection state
#         "comp_entries":           [],
#         "comp_pending_selection": "",
#         "messages": [
#             AIMessage(content=f"✅ Selected: {chosen.get('compName', '')} "
#                               f"(Expiry: {_fmt_display_date(chosen.get('expDate', ''))})"),
#         ],
#     }


# # ---------------------------------------------------------------------------
# # Node 2 — Date input  (single day only, before expiry)
# # ---------------------------------------------------------------------------

# def comp_date_node(state: CompState) -> dict:
#     """
#     Ask for the leave date.
#     Validations:
#       - Must be a valid YYYY-MM-DD date
#       - Cannot be a Sunday
#       - Cannot be in the past
#       - Cannot be after the compensatory expiry date
#       - Only a single day is allowed per compensatory
#     """
#     last_messages = state.get("messages", [])
#     error_prefix  = ""
#     if last_messages:
#         last = last_messages[-1]
#         if isinstance(last, AIMessage) and any(
#             kw in str(last.content)
#             for kw in ["Sunday", "Invalid", "past", "expired", "before"]
#         ):
#             error_prefix = f"{last.content}\n\n"

#     comp_name = state.get("comp_name", "compensatory")
#     exp_str   = state.get("comp_exp_date", "")
#     exp_date  = _parse_api_date(exp_str)
#     exp_disp  = _fmt_display_date(exp_str) if exp_str else "N/A"

#     msg = (
#         f"{error_prefix}"
#         f"Enter the date you want to take compensatory leave for *{comp_name}*\n"
#         f"(Expiry: {exp_disp})  —  format: YYYY-MM-DD:"
#     )

#     decision = interrupt({"message": msg, "action": "comp_date_input"})
#     date_str = str(decision).strip()
#     chosen   = _parse_user_date(date_str)

#     if chosen is None:
#         return {
#             "comp_step": "awaiting_date",
#             "messages": [
#                 AIMessage(content=msg),
#                 HumanMessage(content=date_str),
#                 AIMessage(content="Invalid format. Please use YYYY-MM-DD (e.g. 2026-05-12)."),
#             ],
#         }

#     today = datetime.date.today()

#     if chosen < today:
#         return {
#             "comp_step": "awaiting_date",
#             "messages": [
#                 AIMessage(content=msg),
#                 HumanMessage(content=date_str),
#                 AIMessage(content=f"{chosen} is in the past. Please enter a future date."),
#             ],
#         }

#     if chosen.weekday() == 6:  # Sunday
#         return {
#             "comp_step": "awaiting_date",
#             "messages": [
#                 AIMessage(content=msg),
#                 HumanMessage(content=date_str),
#                 AIMessage(
#                     content=f"{chosen.strftime('%A, %d %B %Y')} is a Sunday and not a working day. "
#                             "Please enter a valid working day (Monday to Saturday)."
#                 ),
#             ],
#         }

#     if exp_date and chosen > exp_date:
#         return {
#             "comp_step": "awaiting_date",
#             "messages": [
#                 AIMessage(content=msg),
#                 HumanMessage(content=date_str),
#                 AIMessage(
#                     content=f"The selected date {chosen} is after the compensatory expiry "
#                             f"date ({exp_disp}). Please choose a date on or before the expiry."
#                 ),
#             ],
#         }

#     api_date = _fmt_api_date(chosen)   # e.g. "12-MAY-2026"

#     return {
#         "comp_leave_date":     date_str,   # YYYY-MM-DD — for display
#         "comp_leave_date_api": api_date,   # DD-MON-YYYY — for API
#         "comp_step":           "awaiting_reason",
#         "messages": [
#             AIMessage(content=msg),
#             HumanMessage(content=date_str),
#             AIMessage(content=f"Leave date set to: {api_date}"),
#         ],
#     }


# # ---------------------------------------------------------------------------
# # Node 3 — Reason input  (free text)
# # ---------------------------------------------------------------------------

# def comp_reason_node(state: CompState) -> dict:
#     comp_name = state.get("comp_name", "compensatory")

#     prompt_msg = (
#         f"Compensatory: *{comp_name}*\n\n"
#         "Please type the reason for taking this compensatory leave:"
#     )
#     raw_reason = interrupt({"message": prompt_msg, "action": "comp_reason_input"})
#     raw_reason = str(raw_reason).strip()

#     if not raw_reason:
#         return {
#             "comp_step": "awaiting_reason",
#             "messages": [
#                 AIMessage(content=prompt_msg),
#                 AIMessage(content="Reason cannot be empty. Please provide a brief description."),
#             ],
#         }

#     return {
#         "comp_reason": raw_reason,
#         "comp_step":   "awaiting_submission",
#         "messages": [
#             AIMessage(content=prompt_msg),
#             HumanMessage(content=raw_reason),
#             AIMessage(content=f"Reason noted: {raw_reason}"),
#         ],
#     }


# # ---------------------------------------------------------------------------
# # Node 4 — Summary + Submit
# # ---------------------------------------------------------------------------

# def comp_submit_node(state: CompState) -> dict:
#     """Show summary, confirm, then POST to /api/CompLeaveApply."""

#     comp_name  = state.get("comp_name", "N/A")
#     leave_date = state.get("comp_leave_date_api", "N/A")   # DD-MON-YYYY
#     reason     = state.get("comp_reason", "N/A")
#     exp_disp   = _fmt_display_date(state.get("comp_exp_date", ""))

#     summary = (
#         f"─── Compensatory Leave Summary ───\n"
#         f"  Compensatory : {comp_name}\n"
#         f"  Leave Date   : {leave_date}\n"
#         f"  Expiry Date  : {exp_disp}\n"
#         f"  Reason       : {reason}\n"
#         f"───────────────────────────────────\n\n"
#         f"Confirm submission? (yes / no)"
#     )

#     conf          = interrupt({"message": summary, "action": "comp_final_confirmation"})
#     user_decision = str(conf).strip().lower()

#     if user_decision not in ("yes", "y"):
#         return {
#             "comp_step": "cancelled",
#             "messages": [
#                 AIMessage(content=summary),
#                 HumanMessage(content=str(conf)),
#                 AIMessage(content="Compensatory leave application cancelled. Let me know if you need anything else."),
#             ],
#         }

#     # ── Build payload ─────────────────────────────────────────────────────────
#     payload = {
#         "empCode":   int(state["emp_code"]),
#         "leaveDate": leave_date,              # "11-MAY-2026"
#         "compId":    int(state["comp_id"]),
#         "reason":    reason,
#         "email":     "",                      # not required
#     }

#     logger.info("[comp_submit] payload=%s", payload)

#     try:
#         result = call_api("POST", f"{LEAVE_BASE_URL}/api/CompLeaveApply", json=payload)
#         logger.info("[comp_submit] API response=%s", result)

#         api_message   = str(result.get("message", ""))
#         api_succeeded = result.get("succeeded")

#         success_kw = ["APPLIED", "SUCCESSFULLY", "SUCCESS", "APPROVED"]
#         failure_kw = ["FAILED", "ERROR", "INVALID", "INSUFFICIENT",
#                       "REJECTED", "ALREADY", "DUPLICATE"]

#         api_upper   = api_message.upper()
#         has_success = any(kw in api_upper for kw in success_kw)
#         has_failure = any(kw in api_upper for kw in failure_kw)

#         if api_succeeded is True or (has_success and not has_failure):
#             status   = "completed"
#             res_text = f"✅ {api_message}" if api_message else "✅ Compensatory leave applied successfully!"
#         elif api_succeeded is False or has_failure:
#             status   = "failed"
#             res_text = f"❌ {api_message}" if api_message else "❌ Submission failed."
#         else:
#             status   = "completed"
#             res_text = f"✅ {api_message}" if api_message else "✅ Compensatory leave applied successfully!"

#         return {
#             "comp_step": status,
#             "response":  res_text,
#             "messages": [
#                 AIMessage(content=summary),
#                 HumanMessage(content=str(conf)),
#                 AIMessage(content=res_text),
#             ],
#         }

#     except Exception as e:
#         logger.error("[comp_submit] Exception: %s", e, exc_info=True)
#         return {
#             "comp_step": "failed",
#             "messages": [
#                 AIMessage(
#                     content=f"A system error occurred while submitting compensatory leave: {str(e)}"
#                 )
#             ],
#         }


# # ---------------------------------------------------------------------------
# # Routing
# # ---------------------------------------------------------------------------

# def route_comp(state: CompState) -> str:
#     step = state.get("comp_step")

#     if step in ("completed", "cancelled", "failed"):
#         return END

#     return {
#         # comp_fetch_node stored entries + pending selection → resolve it
#         "awaiting_comp_selection": "comp_select",
#         # normal forward flow
#         "awaiting_date":       "comp_date",
#         "awaiting_reason":     "comp_reason",
#         "awaiting_submission": "comp_submit",
#     }.get(step, "comp_fetch")


# # ---------------------------------------------------------------------------
# # Subgraph factory
# # ---------------------------------------------------------------------------

# def comp_leave_subgraph(checkpointer=None) -> StateGraph:
#     sg = StateGraph(CompState)

#     sg.add_node("comp_fetch",  comp_fetch_node)
#     sg.add_node("comp_select", comp_select_node)   # NEW — resolves multi-entry pick
#     sg.add_node("comp_date",   comp_date_node)
#     sg.add_node("comp_reason", comp_reason_node)
#     sg.add_node("comp_submit", comp_submit_node)

#     sg.add_edge(START, "comp_fetch")

#     for node in ("comp_fetch", "comp_select", "comp_date", "comp_reason", "comp_submit"):
#         sg.add_conditional_edges(node, route_comp)

#     return sg.compile(checkpointer=checkpointer)