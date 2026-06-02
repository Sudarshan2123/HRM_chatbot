from src.constants.token import Token   
from fastapi import Depends, HTTPException
from fastapi import security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import logging

logger = logging.getLogger(__name__)
_token_service = Token()
security = HTTPBearer(auto_error=False)

def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Validates the Bearer JWT from the Authorization header.
    Returns the decoded payload (includes userName, session_id, exp).
    SECURITY: Validates issuer and signature.
    Raises 401 if missing, expired, invalid, or issuer mismatch.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing or malformed."
        )

    token = credentials.credentials

    # Strip "Bearer " prefix if the frontend accidentally double-wraps it
    if token.lower().startswith("bearer "):
        token = token[7:]

    payload = _token_service.validate_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Token is expired or invalid."
        )
    
    # SECURITY: Verify issuer claim to ensure token was issued by our service
    if payload.get("iss") != "macom-hrms":
        logger.warning("Token with incorrect issuer attempted to authenticate")
        raise HTTPException(
            status_code=401,
            detail="Token invalid — issuer mismatch."
        )

    return payload

def assert_token_matches_emp(token_payload: dict, emp_code: int):
    """
    Validates token's userName against the emp_code in the request body.
    Strips whitespace and normalizes to string to avoid type/format mismatches.
    SECURITY: Strict comparison — raises 403 immediately on mismatch.
    
    Args:
        token_payload: Decoded JWT token dict with 'userName' claim
        emp_code: Employee code from request body
        
    Raises:
        HTTPException 403 if token does not match emp_code
    """
    if not token_payload:
        raise HTTPException(
            status_code=403,
            detail="Token payload is missing."
        )
    
    token_user   = str(token_payload.get("userName", "")).strip()
    request_user = str(emp_code).strip()

    if not token_user:
        raise HTTPException(
            status_code=403,
            detail="Token does not contain a userName claim."
        )
    
    if not request_user:
        raise HTTPException(
            status_code=403,
            detail="Request emp_code is missing or invalid."
        )

    if token_user != request_user:
        # SECURITY: Don't echo back the values in error message
        logger.warning(f"Authorization mismatch for emp_code {request_user}")
        raise HTTPException(
            status_code=403,
            detail="Access denied — token/identity mismatch."
        )
    
    return True