import os
from src.ColdStart.singleton import Init
import logging
from src.constants.token import Token
import uuid
from datetime import timedelta
import requests
from requests.exceptions import RequestException

class Login:
    def __init__(self):
        try:
            self.config = Init()
            self.token = Token()
        except Exception as e:
            logging.error(f"Error initializing Login class: Unexpected error: {e}")
            raise

    def login_user(self, userName: int, password: str) -> dict:
        try:
            logging.info("Entering the login process function")
            # Use MongoDB to find the user by userName
            # user = self.user_collection.find_one({'employee_code': userName})  # Assuming 'name' field stores the username

            if userName:
                url = 'https://docker.mactech.net.in:5013/ldap-service/loginthroughEmail'
                data = {"userName": userName, "password": password}
                result,flag = call_api_post(url, data)
                

                if result.status_code == 200:
                    if flag =="active":
                        session_id = str(uuid.uuid4())
                        access_token_expires = timedelta(minutes=self.config.ACCESS_TOKEN_EXPIRE_MINUTES)
                        access_token = self.token.create_access_token(
                            data={"userName": userName, "session_id": session_id},
                            expires_delta=access_token_expires
                        )
                        return {"access_token": access_token, "token_type": "bearer", "status": "success","user_Status":"active"}
                    elif flag=="Inactive":
                          logging.warning(f"User{userName} is Inactive,OTP verification Required")
                          session_id = str(uuid.uuid4())
                          access_token_expires = timedelta(minutes=self.config.ACCESS_TOKEN_EXPIRE_MINUTES)
                          access_token = self.token.create_access_token(
                            data={"userName": userName, "session_id": session_id},
                            expires_delta=access_token_expires
                            )
                          return{"access_token": access_token, "token_type": "bearer", "status": "success","user_Status":"Inactive"}
                    else:
                        logging.warning(f"User {userName} not Found during Login ,User_status is invalid")     
                else:
                    logging.error(f"LDAP service returned an error: {result.status_code}, {result.text}")
                    return {'status': 'error', 'message': 'Authentication failed with LDAP service'}
            else:
                return {"status": "OTP verification failed"}  # Or "User not found", depending on your logic
        except RequestException as e:
            logging.error(f"Login failed for user {userName}: LDAP service request failed: {e}")
            return {'status': 'error', 'message': 'Failed to communicate with authentication service'}
        except Exception as e:
            logging.error(f"Login failed for user {userName}: Unexpected error: {e}")
            return {'status': 'error', 'message': 'An unexpected error occurred during login'}



import requests
import json

def call_api_post(url: str, data: dict) -> requests.Response:
    """
    Sends a POST request to the specified URL with JSON data.
    Args:
        url: The URL to send the request to.
        data: The data to send in the request body.
    Returns:
        The response from the server.
    Raises:
        requests.exceptions.RequestException: If the request fails.
    """
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        response_data=response.json()
        flag=response_data.get("status")
        return response,flag
    except requests.exceptions.RequestException as e:
        logging.error(f"API call to {url} failed: {e}")
        raise  # Re-raise the exception to be handled by the caller
