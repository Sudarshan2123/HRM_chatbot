from src.constants._init_ import CONFIG_FILE_PATH
from src.utils.common import read_yaml
from src.entity._init_ import (AgentState, Base_Config)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH):

        self.config = read_yaml(config_filepath)

    def get_base_config(self) -> Base_Config:
        config = self.config.config

        base_config = Base_Config(
            MONGODB_URI=config.MONGODB_URI,
            DB_NAME=config.DB_NAME,
            HISTORY_COLLECTION_NAME=config.HISTORY_COLLECTION_NAME,
            collection_user=config.collection_user,
            HISTORY_COLLECTION_Logs=config.HISTORY_COLLECTION_Logs,
            SECRET_KEY=config.SECRET_KEY,
            ALGORITHM=config.ALGORITHM,
            ACCESS_TOKEN_EXPIRE_MINUTES=config.ACCESS_TOKEN_EXPIRE_MINUTES,
            CIPHER_KEY = config.CIPHER_KEY,
            DB_PATH = config.DB_PATH,
            RAG_MODEL=config.RAG_MODEL,
            GCP_LOCATION=config.GCP_LOCATION,
            EMBEDD_MODEL=config.EMBEDD_MODEL,
            API_KEY=config.API_KEY,
            SQLLITE_CONNECTION_STRING=config.SQLLITE_CONNECTION_STRING,
            MAX_POOL_SIZE=config.maxPoolSize,
            REDIS_USERNAME=config.REDIS_USERNAME,
            REDIS_PASSWORD=config.REDIS_PASSWORD,
            REDIS_HOST= config.REDIS_HOST,
            REDIS_PORT= config.REDIS_PORT,
            REDIS_DB= config.REDIS_DB,
            CACHE_TTL= config.CACHE_TTL,
            POSTGRES_HOST=config.POSTGRES_HOST,
            POSTGRES_PORT=config.POSTGRES_PORT,
            POSTGRES_DB=config.POSTGRES_DB,
            POSTGRES_USER=config.POSTGRES_USER,
            POSTGRES_PASSWORD=config.POSTGRES_PASSWORD
        )

        return base_config

    def AgentState(self, default_connection_data, user_input, user_id=None, firm_id=None, role_id=None):
        conn_data = default_connection_data

        # FIX: Added 'user_id' at the root level so data_analyzer.py can retrieve it via state.get("user_id")
        agent_state: AgentState = {
                "input": user_input,
                "user_id": user_id,
                "available_tables": conn_data['table_metadata'],
                "loaded_data": {},
                "selected_tables": None,
                "decision": None,
                "response": None,
                "analysis_context": {
                    "emp_code": user_id,
                    "firm_id": firm_id,
                    "role_id": role_id
                }
            }

        return agent_state