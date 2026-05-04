from typing import Optional, TypedDict, Annotated, Any
from langgraph.graph.message import add_messages
from pydantic import BaseModel

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
    zoho_account_id: Optional[str] 
    zoho_from_address: Optional[str]
    pending_email:   Optional[dict]  # stores draft {to, subject, content} awaiting confirmation



# ── Request / Response Models ──────────────────────────────

class ChatRequest(BaseModel):
    message:   str
    thread_id: Optional[str]
    emp_code:  int  
    firm_id:   int 
    emp_name:  str  
    role_id:   int  
    firm_name: str  


class ChatResponse(BaseModel):
    thread_id: str
    intent:    str
    response:  str
    status:    str           = "completed"
    action:    Optional[str] = None
    feedback_url: Optional[str] = None


class ResumeRequest(BaseModel):
    thread_id: str
    decision:  str
    emp_code:  int = 1203


class ConversationRequest(BaseModel):
    emp_code:  int
    thread_id: str


class UserConversationsRequest(BaseModel):
    emp_code: int


class EncryptedLoginData(BaseModel):
    username: str
    password: str

    class Config:
        extra = 'forbid'

class ZohoKeyRequest(BaseModel):
    emp_code: int
    zoho_mcp_key: str

