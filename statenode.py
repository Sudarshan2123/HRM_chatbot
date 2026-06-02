from dataclasses import dataclass
from typing import List, Literal, Optional,Any
from langgraph.graph.message import MessagesState
from pydantic import BaseModel, Field

class State(MessagesState):
    emp_code:          Optional[int]   = None
    emp_name:          Optional[str]   = None
    firm_id:           Optional[int]   = None
    firm_name:         Optional[str]   = None
    role_id:           Optional[int]   = None
    next_agent:        Optional[str]   = None
    agent_queue:       List[str]
    reason:            Optional[str]   = None
    responded:         bool            = False
    agent_history:     list            = []
    all_tools:         Optional[list]  = None
    pending_agent: str | None  
    # Zoho session state
    zoho_account_id:   Optional[str]   = None
    zoho_folder_id:    Optional[str]   = None
    zoho_from_address: Optional[str]   = None
    # Leave state
    leave_step:        Optional[str]   = None
    leave_type_id:     Optional[int]   = None
    leave_type_name:   Optional[str]   = None
    from_date:         Optional[str]   = None
    to_date:           Optional[str]   = None
    leave_days:        Optional[float] = None
    category_id:     int | None
    category_name:   str | None
    reason_text:       Optional[str]   = None
    reason_id:         Optional[int]   = None
    response:          Optional[str]   = None # stores draft {to, subject, content} awaiting confirmation



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
    options:   Optional[list[str]]  = None   # displayed as buttons in frontend
    values:    Optional[list[Any]]  = None 


class ResumeRequest(BaseModel):
    thread_id: str
    decision:  str
    emp_code:  int 


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



class SupervisorQueue(BaseModel):
    agents:List[str]=Field(description="Ordered list of agents to call. Valid values: 'mail_agent', 'hr_agent', 'leave_approval_agent'")
    reason:str=Field(description="one short explaination of the choice")


@dataclass
class ToolMetadata:
    agents: list[str]  # List of agent names that can access this tool
    category: Optional[str] = None  # Optional category for fallback permissions


class MailPlan(BaseModel):
    intent:Literal["list_mail", "show_mail", "read_mail", "send_mail"]=Field(description="what user wants to do")
    tool_order:list[str]=Field(description="exact ordered list of Zoho tool names to execute")
    reason:str=Field(description="explain why this tool order was chosen")
    to_address: str | None    = Field(None, description="recipient email if mentioned")
    subject: str | None       = Field(None, description="subject if mentioned or inferable")
    content: str | None       = Field(None, description="email body if mentioned or inferable")
    search_query: str | None  = Field(None, description="search term if looking for specific email")
    reason: str               = Field(description="one line explaining the plan")


class FeedbackRequest(BaseModel):
    emp_code:  int
    rating:    int                        = Field(..., ge=1, le=5)
    category:  str = Field(..., max_length=50)
    comments:  Optional[str]             = Field(None, max_length=2000)
    thread_id: Optional[str]             = None 