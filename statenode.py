from typing import Optional, TypedDict, Annotated, Any
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages:        Annotated[list[Any], add_messages]
    intent:          Annotated[str, "general"]
    active_intent:   Optional[str]
    feedback_url:    Optional[str]
    responded:       bool
    emp_code:        Optional[int]
    firm_id:         Optional[int]
    emp_name:        Optional[str]
    role_id:         Optional[int]
    firm_name:       Optional[str]
    leave_step:      Optional[str]
    leave_type_id:   Optional[int]
    leave_type_name: Optional[str]
    leave_days  :Optional[str]
    from_date:       Optional[str]
    to_date:         Optional[str]
    remarks:         Optional[str]
    response:        Optional[str]
    tools:           Optional[str]