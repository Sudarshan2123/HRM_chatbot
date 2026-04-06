import os
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
import yaml
import logging
from cryptography.fernet import Fernet
import time
from functools import wraps
import base64
import os,requests
from fastapi import  HTTPException
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at: {path}")



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

def decode_data(encode_data,CIPHER_KEY):
    #CIPHER_KEY = b'PVmmpjmeRmvPHuoKEiHw5FJntTM2sJAe8baw_8L6RHE='
    cipher_suite = Fernet(CIPHER_KEY)
    decoded_data = cipher_suite.decrypt(encode_data.encode()).decode()
    return decoded_data



def time_it(func):
    """Decorator to measure the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the duration
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        logging.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result  # Return the result of the function
    return wrapper




def decrypt_credentials(encrypted_creds):
    """
    Decrypt credentials that were encrypted using the JavaScript encryptCredentials function.
    
    Args:
        encrypted_creds (dict): Dictionary containing encrypted 'username' and 'password'
    
    Returns:
        dict: Dictionary containing decrypted 'username' and 'password'
    """
    shift = 5  # Same shift value used in JavaScript
    
    def decrypt_string(encrypted_str):
        # First reverse the character shift
        shifted_str = ''.join(
            chr(ord(char) - shift) for char in encrypted_str
        )
        
        # Then decode from base64
        try:
            decoded_bytes = base64.b64decode(shifted_str)
            return decoded_bytes.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to decrypt string: {str(e)}")
    
    try:
        decrypted = {
            'username': decrypt_string(encrypted_creds['username']),
            'password': decrypt_string(encrypted_creds['password'])
        }
        return decrypted
    except KeyError as e:
        raise ValueError(f"Missing required credential field: {str(e)}")
    


def Translate_process_chat(text,target_language,api_key):
        url = f"https://translation.googleapis.com/language/translate/v2?key={api_key}"
        payload = {
            "q": text,
            "target": target_language
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            translation = response.json().get('data', {}).get('translations', [])[0].get('translatedText')
            return translation
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())