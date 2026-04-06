
import threading
from typing import Optional

import requests

import logging
import os
from langgraph.types import interrupt
from langgraph.graph import END, START, StateGraph
from statenode import State as LeaveState
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)
LOGIN_URL      = os.getenv("LOGIN_URL")
LEAVE_BASE_URL = os.getenv("LEAVE_BASE_URL")
_auth_token: str = ""
_token_lock      = threading.Lock()
_user_info: dict = {
    "empCode":  None,   
    "empName":  None,   
    "firmId":   None,   
    "firmName": None,   
    "role":     None,   
    "roleName": None,   
}
LOGIN_CREDENTIALS = {
    "username":   "1203",
    "password":   "123",
    "firmId":     3,
    "ipAddress":  "string",
    "userAgent":  "string",
    "forceLogin": True
}


def login_and_get_token() -> str:
    global _auth_token, _user_info
    try:
        logger.info("Logging in to HRMS...")
        print("Sending login request...")
        response = requests.post(
            LOGIN_URL,
            json=LOGIN_CREDENTIALS,
            timeout=30,
            verify=False
        )
        logger.info(f"Login status: {response.status_code}")
        print(f"Login status: {response.status_code}")
        data = response.json()
        logger.info(f"Login response: {data}")
        print(f"Login response: {data}")
        if not data.get("succeeded"):
            raise Exception(f"Login failed: {data.get('message', 'Unknown error')}")

        d = data["data"]

        with _token_lock:
            _auth_token = d["token"]
            _user_info  = {
                "empCode":  d["empCode"],  
                "empName":  d["empName"],  
                "firmId":   d["firmId"],  
                "firmName": d["firmName"], 
                "role":     d["role"],      
                "roleName": d["roleName"],  
            }

        logger.info(
            f"Login OK — "
            f"EmpCode: {_user_info['empCode']} | "
            f"Name: {_user_info['empName']} | "
            f"FirmId: {_user_info['firmId']} | "
            f"Role: {_user_info['role']} ({_user_info['roleName']})"
        )
        return _auth_token

    except requests.exceptions.Timeout:
        logger.error("Login timed out.")
        raise
    except requests.exceptions.SSLError as e:
        logger.error(f"SSL error: {e}")
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise

def get_token() -> str:
    global _auth_token
    with _token_lock:
        if not _auth_token:
            login_and_get_token()
        return _auth_token

def get_headers() -> dict:
    return {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {get_token()}"
    }

def call_with_retry(method: str, url: str, **kwargs) -> dict:
    """Makes API call. Auto re-logins once on 401."""
    kwargs.setdefault("verify",  False)
    kwargs.setdefault("timeout", 30)

    response = requests.request(method, url, headers=get_headers(), **kwargs)
    print(f"API Request: {method} {url} → Status: {response.status_code}")
    # logger.debug(f"{method} {url} → {response.status_code}")

    if response.status_code == 401:
        logger.warning("401 received — re-logging in...")
        login_and_get_token()
        response = requests.request(method, url, headers=get_headers(), **kwargs)

    # logger.debug(f"Response: {response.text[:300]}")
    print(f"API Response Status: {response.status_code}")
    return response.json()

def leave_balance_node(state: LeaveState) -> dict:
    """Gets employee's leave balances and asks for confirmation."""
    print("Inside leave_balance_node")
    user = get_user_info() 
    result   = api_get_remaining_leave(state["emp_code"], state["firm_id"])
    print("got the result")
    print(result.get("data", []))
    balances = result.get("data", [])

    lines = [f"Hi {state['emp_name']}, your leave balances:\n"]
    for item in balances:
        lines.append(f"  - {item['leaveName']}: {item['eligibleDays']} days")
    lines.append("\nWould you like to proceed? (yes / no)")

    decision = interrupt({              # ← pauses correctly here
        "message": "\n".join(lines),
        "action": "proceed_confirmation"
    })

    if decision.strip().lower() not in ["yes", "y"]:
        return {"leave_step": "cancelled",
                "response": "Leave application cancelled."}

    return {"leave_step": "awaiting_leave_type"}

def route_leave(state: LeaveState) -> str:
    return state.get("leave_step", "cancelled")


def api_get_remaining_leave(emp_code: int, firm_id: int) -> dict:
    return call_with_retry(
        "POST",
        f"{LEAVE_BASE_URL}/EmployeeLeaveApplication/GetRemainingLeave",
        json={"firmId": firm_id, "empCode": emp_code}
    )

def api_get_leave_types(emp_code: int, firm_id: int) -> dict:
    return call_with_retry(
        "POST",
        f"{LEAVE_BASE_URL}/EmployeeLeaveApplication/GetLeaveTypes",
        json={"firmId": firm_id, "empCode": emp_code}
    )

def api_check_eligibility(emp_code: int, firm_id: int, leave_type_id: int) -> dict:
    return call_with_retry(
        "POST",
        f"{LEAVE_BASE_URL}/EmployeeLeaveApplication/IsLeaveEligible",
        json={"firmId": firm_id, "empCode": emp_code, "id": leave_type_id}
    )


def get_user_info() -> dict:
    """Returns login user info. Triggers login if not yet done."""
    if not _user_info["empCode"]:
        login_and_get_token()
    return _user_info

leave_sessions: dict = {}
def get_session(emp_code: int) -> dict:
    if emp_code not in leave_sessions:
        user = get_user_info() 
        leave_sessions[emp_code] = {
            "emp_code":        user["empCode"],   
            "emp_name":        user["empName"],   
            "firm_id":         user["firmId"],    
            "firm_name":       user["firmName"],
            "role_id":         user["role"],     
            "role_name":       user["roleName"], 
            "leave_type_id":   None,   
            "leave_type_name": None,   
            "leave_types_map": {},     
            "from_date":       None,  
            "to_date":         None,  
            "leave_days":      None,   
            "emp_remarks":     "",     
        }
    return leave_sessions[emp_code]


def leave_subgraph(checkpointer=None) -> StateGraph:
    sg = StateGraph(LeaveState)
    sg.add_node("leave_balance", leave_balance_node)
    sg.add_node("leave_type", api_get_leave_types)
    sg.add_node("leave_Eligibility", api_check_eligibility)
    sg.add_edge(START, "leave_balance")
    sg.add_conditional_edges("leave_balance", route_leave, {
        "awaiting_leave_type": "leave_type",
        "cancelled":            END
    })
    return sg.compile(checkpointer=checkpointer)
