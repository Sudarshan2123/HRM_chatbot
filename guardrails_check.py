
import os
os.environ["FASTEMBED_CACHE_PATH"] = "C:/fastembed_models"
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

from src.logging import logger
from functools import lru_cache

_GUARDRAILS_DIR = os.path.join(os.path.dirname(__file__), "guardrails")

@lru_cache(maxsize=1)
def get_rails() -> LLMRails:
    
    with open(os.path.join(_GUARDRAILS_DIR, "config.yml"),encoding="utf-8") as f:
        yaml_content = f.read()
    with open(os.path.join(_GUARDRAILS_DIR,"rails.co"),encoding="utf-8") as f :
        colang_content = f.read()
        
    config = RailsConfig.from_content(
        yaml_content=yaml_content,
        colang_content=colang_content
    )
    logger.info("Guardrails loaded successfully")
    
    guardrails = LLMRails(config=config,verbose= True)  # ← LLMRails, not RunnableRails
    logger.info("Guardrails loaded successfully")
    return guardrails
