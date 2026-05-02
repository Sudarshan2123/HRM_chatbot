
import pytz,jwt
from datetime import datetime, timedelta
from typing import Optional
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from src.ColdStart.singleton import Init
import logging

class Token:
    def __init__(self):
        self.config = Init()
    
    def create_access_token(self,data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(pytz.utc) + expires_delta
        else:
            expire = datetime.now(pytz.utc) + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.config.SECRET_KEY, algorithm=self.config.ALGORITHM)
        return encoded_jwt
    
    def create_update_token(self,data:dict):
            logging.info("Entering access token validation method ")
            User_name = data["userName"]
            session_id = data["session_id"]
            access_token_expires = timedelta(minutes=self.config.ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = self.create_access_token(data={"userName": User_name,"session_id":session_id}, expires_delta=access_token_expires)
            return access_token

    def get_user_name_from_access_token(self,access_token):
            decoded_jwt = jwt.decode(access_token, self.config.SECRET_KEY, algorithms=[self.config.ALGORITHM])
            userName = decoded_jwt.get('userName')  # Adjust according to your token payload
            return userName  
    
    
    def validate_access_token(self,token: str) -> Optional[dict]:
        try:
            logging.info("Entering access token validation method ")
         
            decoded_jwt = jwt.decode(token, self.config.SECRET_KEY, algorithms=[self.config.ALGORITHM])
            userName = decoded_jwt.get('userName')  # Adjust according to your token payload        
            if not userName:
                logging.error("Token is invalid does not contain required fields")
                return None
 
            if userName:
                return decoded_jwt
            else:
                logging.error("Token is invalid does not match requeird token")
                return None
        except ExpiredSignatureError:
            logging.error("Token has expired.")
            return None
        except InvalidTokenError:
            logging.error("Invalid token.")
            return None
        except Exception as e:
             logging.error(f"An unexpected error occurred during token validation:{e}")
             return None
            
    
    def data_update(self, users_collection, employee_code, access_token, session_id):
        users_collection.update_one(
            {"employee_code": employee_code},  # Filter criteria
            {"$set": {"token": access_token, "session_id": session_id}}  # Fields to update
        )
    


