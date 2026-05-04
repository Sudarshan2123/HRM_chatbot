from typing import Any, Dict, List, Optional, TypedDict
import pandas as pd
from pydantic import BaseModel,Field
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()


class OTP_generator_request(BaseModel):
    Phone_number: str
    class Config:
        extra = 'forbid'

class LoginRequest(BaseModel):
    employee_code: str
    firm_id: str

class ChatRegister(BaseModel):
    employee_code: str
    firm_id: str
    class Config:
        extra = 'forbid'


class ChatRequest(BaseModel):
    input: str
    class Config:
        extra = 'forbid'
class ChatRequest2(BaseModel):
    input: str
    user_id:str
    lang: str
    firm_id: str
    role_id: str
    class Config:
        extra = 'forbid'


class ChatHistoryClear(BaseModel):
    access_token: str
    class Config:
        extra = 'forbid'

class TranslationRequest(BaseModel):
    text: str
    target_language: str
    # access_token:str
    class Config:
        extra = 'forbid'

class TextToSpeechRequest(BaseModel):
    text: str
    language: str
    class Config:
        extra = 'forbid'

class EncryptedLoginData(BaseModel):
    userName: str
    password: str
    class Config:
        extra = 'forbid'

class TextRequest(BaseModel):
    emp_code: int

class Base_Config(BaseModel):
    MONGODB_URI: str

    DB_NAME: str
    HISTORY_COLLECTION_NAME: str
    HISTORY_COLLECTION_Logs:str
    collection_user:str
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    CIPHER_KEY: bytes
    DB_PATH : str
    # CHROMA_PATH: str
    RAG_MODEL: str
    GCP_LOCATION :str
    EMBEDD_MODEL: str
    API_KEY: str
    # CHROMA_COLLECTION: str
    #SQLLITE_CONNECTION_STRING: str
    MAX_POOL_SIZE: int
    REDIS_USERNAME:str
    REDIS_PASSWORD:str
    REDIS_HOST:str
    REDIS_PORT:int
    REDIS_DB: int
    CACHE_TTL: int
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str


class API_Config(BaseModel):
    API_Key: str


class User_Master(Base):
    __tablename__ = 'User_Master'

    employee_code = Column(Integer, primary_key=True)
    name = Column(String)
    token = Column(String)
    session_id = Column(String)

# from langchain_core.pydantic_v1 import BaseModel, Field

class AnalysisQueryModel(BaseModel):
    analysis: str = Field(description="analysis of the query")
    query: str = Field(description="query")

class AgentState(TypedDict):
    """State management for the AI agent."""
    input: str
    session_id: Optional[str]
    available_tables: Optional[Dict[str, Dict[str, Any]]]  # table_name -> metadata
    selected_tables: Optional[List[str]]
    loaded_data: Optional[Dict[str, pd.DataFrame]]
    decision: Optional[str]
    response: Optional[str]
    analysis_context: Optional[str]

class QueryResponse(BaseModel):
    success: bool
    response: str
    selected_tables: Optional[List[str]] = None
    analysis_type: Optional[str] = None
    error: Optional[str] = None