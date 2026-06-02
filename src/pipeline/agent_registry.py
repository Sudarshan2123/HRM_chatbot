# src/pipeline/agent_registry.py

from dataclasses import dataclass, field
from typing import Callable, Awaitable
from statenode import State


@dataclass
class AgentConfig:
    # Identity
    name:         str          # internal node name e.g. "leave_agent"
    display_name: str          # what supervisor sees e.g. "leave_approval_agent"
    description:  str          # what this agent does — fed directly into supervisor prompt

    # Routing
    keywords:     list[str]    # quick-match keywords before LLM routing
    has_tools:    bool = True  # does it have a separate tool node?
    tool_node:    str | None = None  # e.g. "leave_tools" — None means uses shared "tools"
    streams_tokens: bool = True
    # Runtime (populated at graph build time)
    fn:           Callable | None = None


# ── Registry ──────────────────────────────────────────────────────────────────
# To add a new agent: add one entry here. Nothing else changes.

AGENT_REGISTRY: dict[str, AgentConfig] = {

    "mail_agent": AgentConfig(
        name         = "mail_agent",
        display_name = "mail_agent",
        description  = "ALL Zoho email tasks — read inbox, send email, reply, check unread",
        keywords = ["send email", "send mail", "read email", "read mail",
                "show email", "show mail", "inbox", "reply to", "draft email",
                "draft mail", "unread", "recent mail", "recent email"],
        has_tools    = True,
        tool_node    = None,
        streams_tokens= True   # uses shared DynamicToolNode
    ),

    "hr_agent": AgentConfig(
        name         = "hr_agent",
        display_name = "hr_agent",
        description  = "HR policy questions, weather, news, general queries, greetings",
        keywords = ["policy", "dress code", "weather", "news", "what is today",
                "today's date", "holiday list", "public holiday", "general query"],
        has_tools    = True,
        tool_node    = None,
        streams_tokens= True   # uses shared DynamicToolNode
    ),

    "leave_agent": AgentConfig(
        name         = "leave_agent",
        display_name = "leave_agent",
        description  = "ALL leave tasks — apply for leave, check balance, cancel leave, leave status",
        keywords = ["apply leave", "apply for leave", "leave balance",
                "check leave", "leave status", "casual leave", "sick leave",
                "earned leave", "take leave", "days off", "time off"],
        has_tools    = True,
        tool_node    = "leave_tools",
        streams_tokens= False  # owns its own tool node
    ),

    # ── Add new agents below — nothing else in the codebase needs to change ──

    # "payroll_agent": AgentConfig(
    #     name         = "payroll_agent",
    #     display_name = "payroll_agent",
    #     description  = "Salary slips, payroll queries, tax declarations, PF/ESI details",
    #     keywords     = ["salary", "payroll", "slip", "tax", "pf", "esi", "ctc"],
    #     has_tools    = True,
    #     tool_node    = None,
    # ),

    # "attendance_agent": AgentConfig(
    #     name         = "attendance_agent",
    #     display_name = "attendance_agent",
    #     description  = "Attendance records, punch-in/out corrections, shift details",
    #     keywords     = ["attendance", "punch", "shift", "late", "absent", "regularize"],
    #     has_tools    = True,
    #     tool_node    = None,
    # ),
}


def get_supervisor_agent_block() -> str:
    """
    Builds the AGENTS section of the supervisor prompt dynamically
    from the registry. Adding a new agent auto-updates the prompt.
    """
    lines = []
    for cfg in AGENT_REGISTRY.values():
        lines.append(f'  "{cfg.display_name}" → {cfg.description}')
    return "\n".join(lines)


def get_display_to_internal_map() -> dict[str, str]:
    """
    Maps display_name → internal node name.
    Supervisor LLM uses display_name; graph uses internal name.
    """
    return {cfg.display_name: cfg.name for cfg in AGENT_REGISTRY.values()}


def get_all_display_names() -> list[str]:
    return [cfg.display_name for cfg in AGENT_REGISTRY.values()]


def keyword_route(message: str) -> str | None:
    """
    Fast pre-LLM keyword routing.
    Scores all agents by number of keyword matches.
    Returns internal node name only if one agent clearly wins (score >= 1).
    Returns None if ambiguous — lets LLM decide.
    """
    text = message.lower()
    scores: dict[str, int] = {}

    for name, cfg in AGENT_REGISTRY.items():
        score = sum(1 for kw in cfg.keywords if kw in text)
        if score > 0:
            scores[name] = score

    if not scores:
        return None

    best_name  = max(scores, key=lambda k: scores[k])
    best_score = scores[best_name]

    # If two agents tie, fall back to LLM — don't guess
    tied = [n for n, s in scores.items() if s == best_score]
    if len(tied) > 1:
        return None

    return best_name