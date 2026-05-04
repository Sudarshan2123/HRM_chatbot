import datetime
import random
import threading
import logging
import os
import requests
import urllib3
from typing import Optional

from langgraph.types import interrupt
from langgraph.graph import END, START, StateGraph
from statenode import State as LeaveState  
from langchain_core.messages import AIMessage, HumanMessage

# --- Configuration & Logging ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOGIN_URL = os.getenv("LOGIN_URL")
LEAVE_BASE_URL = os.getenv("LEAVE_BASE_URL")

_auth_token: str = ""
_token_lock = threading.Lock()

LOGIN_CREDENTIALS = {
    "username": "100606",
    "password": "123",
    "firmId": 3,
    "ipAddress": "string",
    "userAgent": "string",
    "forceLogin": True
}

# --- Helper Functions ---

def login_and_get_token() -> str:
    global _auth_token
    try:
        response = requests.post(LOGIN_URL, json=LOGIN_CREDENTIALS, timeout=30, verify=False)
        data = response.json()
        if not data.get("succeeded"):
            raise Exception(f"Login failed: {data.get('message')}")
        with _token_lock:
            _auth_token = data["data"]["token"]
        return _auth_token
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise

def get_headers() -> dict:
    global _auth_token
    if not _auth_token: login_and_get_token()
    return {"Content-Type": "application/json", "Authorization": f"Bearer {_auth_token}"}

def call_with_retry(method: str, url: str, **kwargs) -> dict:
    kwargs.setdefault("verify", False)
    response = requests.request(method, url, headers=get_headers(), **kwargs)
    if response.status_code == 401:
        login_and_get_token()
        response = requests.request(method, url, headers=get_headers(), **kwargs)
    return response.json()

def calculate_working_days(from_date: str, to_date: str) -> float:
    try:
        start = datetime.datetime.strptime(from_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(to_date, "%Y-%m-%d")
        days = 0
        curr = start
        while curr <= end:
            # 0-5 are Mon-Sat. Since Saturday is working, we count < 6.
            if curr.weekday() < 6: 
                days += 1
            curr += datetime.timedelta(days=1)
        return float(days)
    except Exception: 
        return 0.0

# --- Graph Nodes ---

def leave_balance_node(state: LeaveState) -> dict:
    try:
        # 1. Fetch data from API
        result = call_with_retry(
            "POST", 
            f"{LEAVE_BASE_URL}/EmployeeLeaveApplication/GetRemainingLeave", 
            json={"firmId": state["firm_id"], "empCode": state["emp_code"]}
        )
        print(state["emp_code"])
    except Exception as e:
        logger.error(f"HRMS Connection Error: {e}")
        # Must return a dict, not just the message object
        return {
            "leave_step": "failed", 
            "messages": [AIMessage(content="Unable to connect to HRMS. Please try again later.")]
        }
    
    balances = result.get("data", [])
    if not balances:
        return {
            "leave_step": "failed",
            "messages": [AIMessage(content="No leave balance found for your account. Please contact HR.")]
        }

    # 2. Prepare the prompt
    msg = f"Hi {state.get('emp_name', 'Employee')}, your balances:\n" + \
          "\n".join([f"- {b['leaveName']}: {b['eligibleDays']} days" for b in balances]) + \
          "\n\nWould you like to proceed? (yes/no)"

    # 3. Trigger Interrupt
    # When the user replies, the function resumes here, but 'decision' is now the input string
    decision = interrupt({"message": msg, "action": "proceed_confirmation"})
    
    user_input = str(decision).strip().lower()

    # 4. Handle Decision and update history
    if user_input not in ["yes", "y"]:
        return {
            "leave_step": "cancelled", 
            "messages": [
                AIMessage(content=msg), 
                HumanMessage(content=str(decision)),
                AIMessage(content="Request cancelled.")
            ]
        }
        
    return {
        "leave_step": "awaiting_leave_type", 
        "messages": [
            AIMessage(content=msg), 
            HumanMessage(content=str(decision))
        ]
    }

def leave_types_node(state: LeaveState) -> dict:
    try:
        # 1. Fetch leave types from API
        result = call_with_retry(
            "POST", 
            f"{LEAVE_BASE_URL}/EmployeeLeaveApplication/GetLeaveTypes", 
            json={"firmId": state["firm_id"], "empCode": state["emp_code"]}
        )
    except Exception as e:
        logger.error(f"Error fetching leave types: {e}")
        return {
            "leave_step": "failed",
            "messages": [AIMessage(content="Unable to fetch leave types. Please try again later.")]
        }
    
    types = result.get("data", [])
    if not types:
        return {
            "leave_step": "failed",
            "messages": [AIMessage(content="No leave types found for your account. Please contact HR.")]
        }
    
    # 2. Format the list for the user
    msg = "Please select a leave type (enter the number):\n" + \
          "\n".join([f"{i+1}. {t['leaveType']}" for i, t in enumerate(types)])
    
    # 3. Trigger Interrupt
    selection = interrupt({"message": msg, "action": "leave_type_selection"})
    
    # 4. Process the selection
    if str(selection).isdigit() and 1 <= int(selection) <= len(types):
        sel = types[int(selection) - 1]
        
        return {
            "leave_step": "awaiting_dates",
            "leave_type_id": sel["id"],
            "leave_type_name": sel["leaveType"],
            "messages": [
                AIMessage(content=msg), 
                HumanMessage(content=str(selection)), # Show the number they typed
                AIMessage(content=f"You selected: {sel['leaveType']}") # Confirmation
            ]
        }
    
    # Handle invalid input
    return {
        "leave_step": "cancelled", 
        "messages": [
            AIMessage(content=msg),
            HumanMessage(content=str(selection)),
            AIMessage(content="Invalid selection. Request cancelled.")
        ]
    }

def get_from_date(state: LeaveState) -> dict:
    
    last_messages = state.get("messages", [])
    error_prefix = ""
    if last_messages:
        last = last_messages[-1]
        if isinstance(last, AIMessage) and "Sunday" in str(last.content):
            error_prefix = f"{last.content}\n\n"
        elif isinstance(last, AIMessage) and "Invalid format" in str(last.content):
            error_prefix = f"{last.content}\n\n"

    msg = f"{error_prefix}Enter the Start Date (YYYY-MM-DD):"
    
    decision = interrupt({"message": msg, "action": "from_date_input"})
    
    try:
        date_str = str(decision).strip()
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        
        if dt.weekday() == 6:
            day_name = dt.strftime("%A, %d %B %Y")
            error_msg = (
                f"{day_name} is a Sunday and not a working day.\n"
                f"Please enter a valid working day (Monday to Saturday)."
            )
            return {
                "leave_step": "awaiting_dates",
                "messages": [
                    AIMessage(content=msg),
                    HumanMessage(content=date_str),
                    AIMessage(content=error_msg)   # ← stored for next interrupt
                ]
            }
        
        return {
            "from_date": date_str,
            "leave_step": "awaiting_to_date",
            "messages": [
                AIMessage(content=msg),
                HumanMessage(content=date_str),
                AIMessage(content=f"From date set to: {date_str}")
            ]
        }

    except ValueError:
        return {
            "leave_step": "awaiting_dates",
            "messages": [
                AIMessage(content=msg),
                HumanMessage(content=str(decision)),
                AIMessage(content="Invalid format. Please use YYYY-MM-DD (e.g. 2026-04-27).")
            ]
        }

def get_to_date(state: LeaveState) -> dict:

    last_messages = state.get("messages", [])
    error_prefix = ""
    if last_messages:
        last = last_messages[-1]
        if isinstance(last, AIMessage) and any(
            word in str(last.content)
            for word in ["Error", "Rejected", "Invalid", "Sunday"]
        ):
            error_prefix = f"{last.content}\n\n"

    msg = f"{error_prefix}Enter the End Date (YYYY-MM-DD):"
    decision = interrupt({"message": msg, "action": "to_date_input"})

    try:
        to_str  = str(decision).strip()
        to_dt   = datetime.datetime.strptime(to_str, "%Y-%m-%d")
        from_dt = datetime.datetime.strptime(state["from_date"], "%Y-%m-%d")

        if to_dt < from_dt:
            return {
                "leave_step": "awaiting_to_date",
                "messages": [
                    AIMessage(content=msg),
                    HumanMessage(content=to_str),
                    AIMessage(content=f"End date ({to_str}) cannot be before Start date ({state['from_date']}). Please re-enter.")
                ]
            }

        if to_dt.weekday() == 6:
            day_name = to_dt.strftime("%A, %d %B %Y")
            return {
                "leave_step": "awaiting_to_date",
                "messages": [
                    AIMessage(content=msg),
                    HumanMessage(content=to_str),
                    AIMessage(content=f"{day_name} is a Sunday and not a working day. Please enter a valid working day (Monday to Saturday).")
                ]
            }

        work_days = calculate_working_days(state["from_date"], to_str)

        if work_days <= 0:
            day_type = "a Sunday" if from_dt == to_dt else "only Sundays"
            return {
                "leave_step": "awaiting_to_date",
                "messages": [
                    AIMessage(content=msg),
                    HumanMessage(content=to_str),
                    AIMessage(content=f"Rejected: The selected range contains {day_type}. Please include at least one working day (Mon-Sat).")
                ]
            }

        return {
            "to_date":    to_str,
            "leave_days": work_days,
            "leave_step": "awaiting_remarks",
            "messages": [
                AIMessage(content=msg),
                HumanMessage(content=to_str),
                AIMessage(content=f"End date set to: {to_str}. Total working days: {work_days}")
            ]
        }

    except ValueError:
        return {
            "leave_step": "awaiting_to_date",
            "messages": [
                AIMessage(content=msg),
                HumanMessage(content=str(decision)),
                AIMessage(content="Invalid format. Please use YYYY-MM-DD (e.g. 2026-04-28).")
            ]
        }

def remarks_reason(state: LeaveState) -> dict:
    msg = "Please enter the reason for your leave:"
    reason = interrupt({"message": msg, "action": "remarks_input"})
    
    reason_str = str(reason).strip()
    
    if not reason_str:
        return {
            "messages": [
                AIMessage(content=msg),
                AIMessage(content="Reason cannot be empty. Please provide a brief explanation.")
            ]
        }
    
    return {
        "remarks": reason_str, 
        "leave_step": "awaiting_submission",
        "messages": [
            AIMessage(content=msg), 
            HumanMessage(content=reason_str),
            AIMessage(content="Reason noted.")
        ]
    }

def submit_leave_application(state: LeaveState) -> dict:
    # 1. Provide a clear summary for final confirmation
    summary = (
        f"---Final Summary---\n"
        f"Type: {state['leave_type_name']}\n"
        f"Dates: {state['from_date']} to {state['to_date']}\n"
        f"Duration: {state['leave_days']} working day(s)\n"
        f"Reason: {state['remarks']}\n\n"
        f"Confirm submission? (yes/no)"
    )
    
    conf = interrupt({"message": summary, "action": "final_confirmation"})
    user_decision = str(conf).strip().lower()
    
    # 2. Handle cancellation
    if user_decision not in ["yes", "y"]:
        return {
            "leave_step": "cancelled", 
            "messages": [
                AIMessage(content=summary),
                HumanMessage(content=str(conf)),
                AIMessage(content="Leave application discarded.")
            ]
        }

    # 3. Build API payload
    payload = {
        "EmpCode": state["emp_code"],
        "roleId": state["role_id"],
        "FirmId": state["firm_id"],
        "LeaveType": state["leave_type_id"],
        "CategoryId": 1,
        "LeaveFrdate": state["from_date"],
        "LeaveTodate": state["to_date"],
        "LeaveDays": state["leave_days"],
        "EmpRemarks": state["remarks"],
        "File1": "" # Ensuring keys required by the API are present
    }
    
    # 4. Final Submission
    try:
        result = call_with_retry("POST", f"{LEAVE_BASE_URL}/EmployeeLeaveApplication/CreateLeave", json=payload)
        succeeded = result.get("succeeded", False)
        
        status = "completed" if succeeded else "failed"
        res_text = "Your leave application has been submitted successfully!" if succeeded \
                   else f"Submission failed: {result.get('message', 'Unknown error')}"
        
        output = {
            "leave_step": status,
            "response": res_text,
            "messages": [
                AIMessage(content=summary),
                HumanMessage(content=str(conf)),
                AIMessage(content=res_text)
            ]
        }

        if random.random() < 0.20:
            feedback_url = "https://docs.google.com/forms/d/e/1FAIpQLSdbAipY_JI26rgL2pTUgcPlgl0lAVWDdY_z0fboHtMBf9cEKQ/viewform?usp=publish-editor"
            output["feedback_url"] = feedback_url
            output["response"] += "\n\nWe value your feedback!"
            # Update the last message to include the feedback text
            output = {
                    "leave_step": status,
                    "response": res_text,
                    "messages": [
                        AIMessage(content=summary),
                        HumanMessage(content=str(conf)),
                        AIMessage(content=f"{res_text}\n\nWe value your feedback!\n{feedback_url}")
                    ]
                } 
        return output
        
    except Exception as e:
        error_msg = f"System error during submission: {str(e)}"
        return {
            "leave_step": "failed",
            "messages": [AIMessage(content=error_msg)]
        }

# --- Routing ---

def route_leave(state: LeaveState) -> str:
    step = state.get("leave_step")
    if step in ["completed", "cancelled", "failed"]:
        return END
    return {
        "awaiting_leave_type": "leave_type",
        "awaiting_dates":      "from_date",   
        "awaiting_to_date":    "to_date",
        "awaiting_remarks":    "remarks",
        "awaiting_submission": "submit"
    }.get(step, "leave_balance")

def leave_subgraph(checkpointer=None) -> StateGraph:
    sg = StateGraph(LeaveState)
    sg.add_node("leave_balance", leave_balance_node)
    sg.add_node("leave_type", leave_types_node)
    sg.add_node("from_date", get_from_date)
    sg.add_node("to_date", get_to_date)
    sg.add_node("remarks", remarks_reason)
    sg.add_node("submit", submit_leave_application)
    sg.add_edge(START, "leave_balance")
    for node in ["leave_balance", "leave_type", "from_date", "to_date", "remarks", "submit"]:
        sg.add_conditional_edges(node, route_leave)
    return sg.compile(checkpointer=checkpointer)