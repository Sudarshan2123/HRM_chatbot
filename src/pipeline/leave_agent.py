# leave_agent.py

import requests
from datetime import datetime, timedelta
import logging
import threading
import urllib3
from langgraph.types import interrupt
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

LOGIN_URL      = "https://macserv.mactech.net.in/HrmsGeneralApi/api/Login/Login"
LEAVE_BASE_URL = "https://macserv.mactech.net.in/HrmsLeaveApi/api"

LOGIN_CREDENTIALS = {
    "username":   "1203",
    "password":   "123",
    "firmId":     3,
    "ipAddress":  "string",
    "userAgent":  "string",
    "forceLogin": True
}

# ─────────────────────────────────────────────
# TOKEN + USER INFO STORE
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# LOGIN & TOKEN MANAGEMENT
# ─────────────────────────────────────────────
def login_and_get_token() -> str:
    global _auth_token, _user_info
    try:
        logger.info("Logging in to HRMS...")
        response = requests.post(
            LOGIN_URL,
            json=LOGIN_CREDENTIALS,
            timeout=30,
            verify=False
        )
        logger.info(f"Login status: {response.status_code}")
        data = response.json()
        logger.info(f"Login response: {data}")

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


def get_user_info() -> dict:
    """Returns login user info. Triggers login if not yet done."""
    if not _user_info["empCode"]:
        login_and_get_token()
    return _user_info


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
    logger.debug(f"{method} {url} → {response.status_code}")

    if response.status_code == 401:
        logger.warning("401 received — re-logging in...")
        login_and_get_token()
        response = requests.request(method, url, headers=get_headers(), **kwargs)

    logger.debug(f"Response: {response.text[:300]}")
    return response.json()


# ─────────────────────────────────────────────
# API CALLERS
# All fields come from session (which is built from login response)
# ─────────────────────────────────────────────

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


def api_submit_leave(
    emp_code:    int,
    role_id:     int,
    firm_id:     int,
    leave_type:  int,
    from_date:   str,
    to_date:     str,
    leave_days:  float,
    emp_remarks: str
) -> dict:

    payload = {
        "EmpCode":           emp_code,
        "roleId":            role_id,
        "FirmId":            firm_id,
        "LeaveType":         leave_type,
        "CategoryId":        1,         
        "LeaveFrdate":       from_date,
        "LeaveTodate":       to_date,
        "LeaveDays":         leave_days,
        "HpayCategory":      None,      
        "EmpRemarks":        emp_remarks,
        "RecAuthorizedEmp":  None,
        "ApprAuthorizedEmp": None,
        "File1":             ""
    }
    return call_with_retry(
        "POST",
        f"{LEAVE_BASE_URL}/EmployeeLeaveApplication/CreateLeave",
        json=payload
    )


# ─────────────────────────────────────────────
# WORKING DAYS CALCULATOR
# ─────────────────────────────────────────────
def calculate_working_days(from_date: str, to_date: str) -> float:
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end   = datetime.strptime(to_date,   "%Y-%m-%d")
    return float(sum(
        1 for i in range((end - start).days + 1)
        if (start + timedelta(days=i)).weekday() < 5  
    ))


# ─────────────────────────────────────────────
# SESSION STATE (in-memory per emp_code)
# ─────────────────────────────────────────────
leave_sessions: dict = {}

def get_session(emp_code: int) -> dict:
    if emp_code not in leave_sessions:
        user = get_user_info() 
        leave_sessions[emp_code] = {
            "step":            "start",
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


def clear_session(emp_code: int):
    if emp_code in leave_sessions:
        del leave_sessions[emp_code]



def run_leave_agent(emp_code: int, user_message: str) -> str:
    session = get_session(emp_code)
    step    = session["step"]
    msg     = user_message.strip().lower()

    s_emp  = session["emp_code"]
    s_firm = session["firm_id"]

    # ── STEP 1: Show leave balance ────────────────────────────
    if step == "start":
        try:
            result = api_get_remaining_leave(s_emp, s_firm)
        except Exception as e:
            clear_session(emp_code)
            return f"Unable to connect to HRMS. Please try again later.\nError: {str(e)}"

        if not result.get("succeeded"):
            clear_session(emp_code)
            return f"Unable to fetch leave balance: {result.get('message', 'Please try again.')}"

        balances = result.get("data", [])
        if not balances:
            clear_session(emp_code)
            return "No leave balance found for your account. Please contact HR."

        lines = [f"Hi {session['emp_name']}, here are your available leave balances:\n"]
        for item in balances:
            lines.append(f"  - {item['leaveName']}: {item['eligibleDays']} days")
        lines.append("\nWould you like to proceed with applying for leave? (yes / no)")
        session["step"] = "awaiting_proceed"
        decision = interrupt({
            "message": "\n".join(lines),   # ← this is what your UI shows
            "action": "api_get_remaining_leave",
            "emp_code": emp_code
            })
            # ── STEP 2: Confirm proceed → fetch leave types ─────────
        if decision.strip().lower() not in ["yes", "y"]:
            clear_session(emp_code)
            raise PermissionError("Leave application cancelled. Let me know if you need anything else.")

        try:
            result = api_get_leave_types(s_emp, s_firm)
        except Exception as e:
            clear_session(emp_code)
            raise PermissionError(f"Unable to fetch leave types.\nError: {str(e)}")

        if not result.get("succeeded"):
            clear_session(emp_code)
            raise(f"Unable to fetch leave types: {result.get('message', 'Please try again.')}")

        types = result.get("data", [])
        if not types:
            clear_session(emp_code)
            raise ValueError("No leave types found. Please contact HR.")

        # Store id → name map for validation in next step
        session["leave_types_map"] = {
            str(item["id"]): item["leaveType"] for item in types
        }

        selected_id = interrupt({
            "message": "\n".join(lines),
            "action": "api_get_leave_type",
            "emp_code": emp_code
        })

        # validate what they selected
        if selected_id.strip() not in session["leave_types_map"]:
            valid = ", ".join(session["leave_types_map"].keys())
            raise ValueError(f"Invalid choice. Please enter a valid number ({valid}):")

        session["leave_type_id"]   = int(selected_id.strip())
        session["leave_type_name"] = session["leave_types_map"][selected_id.strip()]

        try:
            result = api_check_eligibility(s_emp, s_firm, session["leave_type_id"])
        except Exception as e:
            clear_session(emp_code)
            return f"Unable to check leave eligibility.\nError: {str(e)}"

        if not result.get("succeeded"):
            clear_session(emp_code)
            return f"Eligibility check failed: {result.get('message', 'Please try again.')}"

        data = result.get("data", {})
        if data.get("balanceLeaves", 0) == 0:
            clear_session(emp_code)
            return (
                f"Sorry, you have no balance remaining for "
                f"{session['leave_type_name']}.\n"
                "Please choose a different leave type or contact HR."
            )

        from_date = interrupt({
            "message": (
                f"You have {data['balanceLeaves']} days of "
                f"{data['leaveTypeName']} available.\n\n"
                "Please enter your leave start date (YYYY-MM-DD):"
            ),
            "action": "collect_from_date",  
            "emp_code": emp_code
        })

        # ── STEP 4a: Collect from_date ──────────────────────────
        try:
            datetime.strptime(user_message.strip(), "%Y-%m-%d")
        except ValueError:
            return "Invalid date format. Please enter as YYYY-MM-DD (e.g. 2025-04-10):"

        session["from_date"] = user_message.strip()
        session["step"]      = "awaiting_to_date"
        to_date = interrupt({
            "message": (
                "Please enter your leave start date (YYYY-MM-DD):"
            ),
            "action": "collect_to_date",  
            "emp_code": emp_code
        })
         # ── STEP 4b: Collect to_date ──────────────────────────────
        try:
            datetime.strptime(user_message.strip(), "%Y-%m-%d")
        except ValueError:
            return "Invalid date format. Please enter as YYYY-MM-DD (e.g. 2025-04-15):"

        to_date   = user_message.strip()
        from_date = session["from_date"]

        if datetime.strptime(to_date, "%Y-%m-%d") < datetime.strptime(from_date, "%Y-%m-%d"):
            return "End date cannot be before start date. Please enter a valid end date:"

        leave_days = calculate_working_days(from_date, to_date)
        if leave_days == 0:
            return "Selected dates fall on weekends only. Please choose valid working days:"

        session["to_date"]    = to_date
        session["leave_days"] = leave_days
        session["step"]       = "awaiting_remarks"
        remarks = interrupt({
            "message": (
                "Any remarks for this leave? (type 'none' to skip):"
            ),
            "action": "awaiting_remarks",  
            "emp_code": emp_code
        })
        # ── STEP 4c: Collect remarks ──────────────────────────────

        remarks = user_message.strip()
        session["emp_remarks"] = (
            "" if remarks.lower() in ["none", "no", "skip", ""] else remarks
        )

        summary = (
            f"Here is your leave application summary:\n\n"
            f"  Employee    : {session['emp_name']} ({session['emp_code']})\n"
            f"  Firm        : {session['firm_name']}\n"
            f"  Leave Type  : {session['leave_type_name']}\n"
            f"  From Date   : {session['from_date']}\n"
            f"  To Date     : {session['to_date']}\n"
            f"  Total Days  : {session['leave_days']} working day(s)\n"
            f"  Remarks     : "
            f"{session['emp_remarks'] if session['emp_remarks'] else 'None'}\n\n"
            f"Do you confirm to submit? (yes / no)"
        )

        session["step"] = "awaiting_confirmation"
        confirmation = interrupt({
            "message": (summary),
            "action": "awaiting_remarks",  
            "emp_code": emp_code
        })
        # ── STEP 5: Confirm → submit ──────────────────────────────
        if confirmation.strip().lower() not in ["yes", "y"]:
            clear_session(emp_code)
            PermissionError ("Leave application cancelled. Let me know if you need anything else.")

        try:
            result = api_submit_leave(
                emp_code    = session["emp_code"],    
                role_id     = session["role_id"],     
                firm_id     = session["firm_id"],   
                leave_type  = session["leave_type_id"],
                from_date   = session["from_date"],
                to_date     = session["to_date"],
                leave_days  = session["leave_days"],
                emp_remarks = session["emp_remarks"]
            )
        except Exception as e:
            clear_session(emp_code)
            return f"Submission failed due to a connection error.\nError: {str(e)}"

        clear_session(emp_code)

        if result.get("succeeded"):
            app_id = result["data"]["leaveId"]
            return (
                f"Your leave has been submitted successfully!\n\n"
                f"  Application ID : {app_id}\n"
                f"  Status         : Pending Approval\n\n"
                "Is there anything else I can help you with?"
            )
        else:
            return (
                f"Leave submission failed: "
                f"{result.get('message', 'Unknown error. Please try again.')}"
            )

    # ── Fallback ──────────────────────────────────────────────
    else:
        clear_session(emp_code)
        return "Something went wrong. Please start again by saying 'apply leave'."