# src/pipeline/leave_tools.py
import datetime
import os
import requests
from langchain_core.tools import tool

LEAVE_BASE_URL = os.getenv("LEAVE_BASE_URL")

LEAVE_TYPES = {1: "Casual Leave", 2: "Sick Leave", 3: "Earned Leave"}


def _call_leave_api(method: str, path: str, **kwargs) -> dict:
    kwargs.setdefault("verify", False)
    response = requests.request(
        method,
        f"{LEAVE_BASE_URL}{path}",
        headers={"Content-Type": "application/json"},
        timeout=30,
        **kwargs,
    )
    response.raise_for_status()
    return response.json()


def _parse_api_date(date_str: str) -> str:
    if not date_str:
        return "N/A"
    normalized = str(date_str).strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1]
    try:
        return datetime.datetime.fromisoformat(normalized).date().isoformat()
    except ValueError:
        return normalized


def _approximate_leave_days(app: dict, from_date: str, to_date: str) -> str:
    raw_days = app.get("noOfDays") or app.get("days") or app.get("duration")
    if raw_days is not None:
        if isinstance(raw_days, (int, float)):
            return str(int(raw_days))
        if isinstance(raw_days, str) and raw_days.strip().isdigit():
            return raw_days.strip()

    try:
        start = datetime.datetime.fromisoformat(str(from_date).rstrip("Z")).date()
        end = datetime.datetime.fromisoformat(str(to_date).rstrip("Z")).date()
        delta = (end - start).days + 1
        return str(delta if delta > 0 else 1)
    except Exception:
        return "?"


def _format_leave_status(records) -> str:
    if isinstance(records, dict) and records.get("status") == "error":
        return (
            "I'm sorry, I was unable to fetch your leave status at this time. "
            f"Error: {records.get('message', 'unknown error')}"
        )

    if isinstance(records, dict):
        records = records.get("data", [])

    if not records:
        return (
            "You have no leave applications on record.\n"
            "Is there anything else I can help you with?"
        )

    LEAVE_TYPE_NAMES = {1: "Casual Leave", 2: "Sick Leave", 3: "Earned Leave"}

    lines = ["Your Leave Applications:"]
    for app in records:
        # ── Leave type ────────────────────────────────────────────────────────
        leave_type = (
            app.get("leaveTypeName")
            or app.get("leave_type")
            or app.get("leaveType")
            or LEAVE_TYPE_NAMES.get(app.get("type"))
            or app.get("categoryName")
            or "Leave"
        )

        # ── Dates ─────────────────────────────────────────────────────────────
        from_raw  = app.get("fromDate")  or app.get("from_date")  or app.get("startDate") or "N/A"
        to_raw    = app.get("toDate")    or app.get("to_date")    or app.get("endDate")   or "N/A"
        from_date = _parse_api_date(from_raw)
        to_date   = _parse_api_date(to_raw)
        days      = _approximate_leave_days(app, from_raw, to_raw)

        # ── Status ────────────────────────────────────────────────────────────
        status = (
            app.get("status")
            or app.get("approvalStatus")
            or app.get("leaveStatus")
            or "Unknown"
        )

        # ── Recommender ───────────────────────────────────────────────────────
        recommender_name = app.get("recommenderName") or app.get("recommender_name") or ""
        recommender_code = app.get("recommenderCode") or app.get("recommender_code") or ""
        recommender_date = _parse_api_date(
            app.get("recommenderDate") or app.get("recommender_date") or ""
        )

        # ── Approver ──────────────────────────────────────────────────────────
        approver_name = app.get("approverName") or app.get("approver_name") or ""
        approver_code = app.get("approverCode") or app.get("approver_code") or ""
        approver_date = _parse_api_date(
            app.get("approverDate") or app.get("approver_date") or ""
        )

        # ── Format labels ─────────────────────────────────────────────────────
        date_label   = from_date if from_date == to_date else f"{from_date} → {to_date}"
        day_label    = f"{days} day" if str(days).strip() == "1" else f"{days} days"
        status_label = str(status).strip().title()

        # ── Build entry ───────────────────────────────────────────────────────
        entry_lines = [
            f"• {leave_type}: {date_label} | {day_label} | {status_label}"
        ]

        if recommender_name or recommender_code:
            rec_parts = filter(None, [recommender_name, f"({recommender_code})" if recommender_code else ""])
            rec_str   = " ".join(rec_parts)
            rec_line  = f"  Recommended  : {rec_str}"
            if recommender_date and recommender_date != "N/A":
                rec_line += f" on {recommender_date}"
            entry_lines.append(rec_line)

        if approver_name or approver_code:
            apv_parts = filter(None, [approver_name, f"({approver_code})" if approver_code else ""])
            apv_str   = " ".join(apv_parts)
            apv_line  = f"  Approved by  : {apv_str}"
            if approver_date and approver_date != "N/A":
                apv_line += f" on {approver_date}"
            entry_lines.append(apv_line)

        lines.append("\n".join(entry_lines) + "\n")

    lines.append("\nIs there anything else I can help you with?")
    return "\n".join(lines)


def _fetch_leave_status_data(emp_id: str):
    try:
        result = _call_leave_api("GET", f"/api/LeaveStatus/{emp_id}")
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("data", result)
        return [result]
    except Exception as e:
        return {"status": "error", "message": f"Error fetching leave status: {e}"}


# ── Safe Tools ────────────────────────────────────────────────────────────────

@tool("leave_check_balance")
def leave_check_balance(emp_id: str) -> dict:
    """
    Check remaining leave balance for an employee.
    Returns raw data for Casual, Sick, and Earned leave.
    """
    try:
        result = _call_leave_api("GET", f"/api/LeaveBalance/{emp_id}")
        raw = result if isinstance(result, list) else result.get("data", [])

        # Return RAW data so the LLM can follow the System Prompt rules
        return {
            "emp_id": emp_id,
            "balances": raw,  # This contains the total, used, and available fields
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool("leave_get_status")
def leave_get_status(emp_id: str):
    """
    Get status of all previously applied leaves for an employee.
    Returns a formatted leave summary string for display.
    """
    raw = _fetch_leave_status_data(emp_id)
    if isinstance(raw, dict) and raw.get("status") == "error":
        return raw
    return _format_leave_status(raw)


@tool("leave_get_categories")
def leave_get_categories(emp_id: str) -> str:
    """
    Get the full list of leave categories with their IDs from the API.
    Always call this before applying leave so the user can pick a category.

    Returns a DISPLAY section (sequential 1-based numbers for the user to pick)
    and an INTERNAL LOOKUP section (maps display index → real categoryId).
    IMPORTANT: Never show INTERNAL LOOKUP to the user.
    When user picks number N, use INTERNAL LOOKUP to resolve the real categoryId.
    """
    try:
        result = _call_leave_api("GET", "/api/LeaveCategory")
        raw = result if isinstance(result, list) else result.get("data", [])
        if not raw:
            return "No categories found."

        # Sort by categoryId so they display in a consistent order
        sorted_categories = sorted(raw, key=lambda c: c["categoryId"])

        # DISPLAY: sequential 1-based index so the user always picks 1, 2, 3...
        display_lines = [
            f"{i + 1}. {c['categoryName']}"
            for i, c in enumerate(sorted_categories)
        ]

        # INTERNAL: display_index → real categoryId (never show to user)
        id_map_lines = [
            f"{i + 1}={c['categoryId']}"
            for i, c in enumerate(sorted_categories)
        ]

        return (
            "DISPLAY TO USER:\n"
            + "\n".join(display_lines)
            + "\n\n"
            + "INTERNAL LOOKUP — DO NOT SHOW TO USER — index:category_id mapping:\n"
            + ", ".join(id_map_lines)
            + "\n\nWhen user picks number N, set CATEGORY_ID = the id mapped to N above. "
            "Set CATEGORY_NAME = the name at position N. Never show the id to the user."
        )
    except Exception as e:
        return f"Error fetching categories: {e}"


@tool("leave_calculate_days")
def leave_calculate_days(from_date: str, to_date: str) -> str:
    """
    Calculate working days (Monday–Saturday) between two dates.
    Dates must be in YYYY-MM-DD format.
    Use this to confirm day count before applying leave.
    """
    try:
        start = datetime.datetime.strptime(from_date, "%Y-%m-%d")
        end   = datetime.datetime.strptime(to_date,   "%Y-%m-%d")

        if end < start:
            return f"Error: End date {to_date} cannot be before start date {from_date}."

        days, curr = 0, start
        while curr <= end:
            if curr.weekday() < 6:
                days += 1
            curr += datetime.timedelta(days=1)

        return f"{days} working day(s) between {from_date} and {to_date}."
    except ValueError:
        return "Error: Invalid date format. Use YYYY-MM-DD."


@tool("leave_find_reasons")
def leave_find_reasons(user_text: str, category_id: int) -> str:
    """
    Fetch leave reasons for a category from the API, then return the top matches.
    Returns reason text AND reason ID — the ID is required for leave submission.
    Always call this before leave_apply to get a valid reason_id.
    user_text: what the user typed as their reason (used to filter/rank results).
    category_id: must come from leave_get_categories INTERNAL LOOKUP (real categoryId).
    """
    try:
        result = _call_leave_api("GET", f"/api/LeaveReason/{category_id}")
        raw = result if isinstance(result, list) else result.get("data", [])

        if not raw:
            return f"No reasons found for category ID {category_id}."

        user_lower = user_text.lower()
        matched = [
            r for r in raw
            if user_lower in r.get("reasonName", "").lower()
            or any(word in r.get("reasonName", "").lower() for word in user_lower.split())
        ]

        candidates = matched if matched else raw
        candidates = candidates[:10]

        # ── Clean display list (shown to user) ──────────────────────
        display_lines = [
            f"{i + 1}. {r['reasonName']}"
            for i, r in enumerate(candidates)
        ]

        # ── Hidden ID map (for LLM use only — never show to user) ───
        id_map_lines = [
            f"{i + 1}={r['reasonId']}"
            for i, r in enumerate(candidates)
        ]

        return (
            "DISPLAY TO USER:\n"
            + "\n".join(display_lines)
            + "\n\n"
            + "INTERNAL LOOKUP — DO NOT SHOW TO USER — index:reason_id mapping:\n"
            + ", ".join(id_map_lines)
            + "\n\nWhen user picks a number N, set REASON_ID = the id mapped to N above. "
            "Set REASON_TEXT = the name at position N. Never show the id to the user."
        )
    except Exception as e:
        return f"Error fetching reasons: {e}"


# ── Sensitive Tool ────────────────────────────────────────────────────────────

@tool("leave_apply")
def leave_apply(
    emp_id: str,
    leave_type_id: int,
    from_date: str,
    to_date: str,
    category_id: int,
    reason_id: int,
    reason_text: str,
) -> str:
    """
    SENSITIVE: Submit a leave application. Requires user confirmation before execution.
    leave_type_id: 1=Casual, 2=Sick, 3=Earned.
    reason_id must come from leave_find_reasons INTERNAL LOOKUP — never guess it.
    category_id must come from leave_get_categories INTERNAL LOOKUP — never guess it.
    Dates must be YYYY-MM-DD format.
    """
    def to_iso(d: str) -> str:
        return datetime.datetime.strptime(d, "%Y-%m-%d").isoformat() + "Z"

    payload = {
        "employeeId": str(emp_id),
        "type":       int(leave_type_id),
        "fromDate":   to_iso(from_date),
        "toDate":     to_iso(to_date),
        "categoryId": int(category_id),
        "reasonId":   int(reason_id),
        "reason":     reason_text,
    }
    try:
        result = _call_leave_api("POST", "/api/LeaveApply", json=payload)
        msg    = result.get("message", "")
        return f" {msg}" if result.get("succeeded") else f" {msg}"
    except Exception as e:
        return f"Error applying leave: {e}"


# ── Registry ──────────────────────────────────────────────────────────────────

LEAVE_SAFE_TOOLS = [
    leave_check_balance,
    leave_get_status,
    leave_get_categories,
    leave_calculate_days,
    leave_find_reasons,
]

LEAVE_SENSITIVE_TOOLS = [leave_apply]
LEAVE_SENSITIVE_NAMES = {t.name for t in LEAVE_SENSITIVE_TOOLS}
LEAVE_ALL_TOOLS       = LEAVE_SAFE_TOOLS + LEAVE_SENSITIVE_TOOLS