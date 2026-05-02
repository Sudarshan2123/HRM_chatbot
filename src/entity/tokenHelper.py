from src.constants.token import Token   
from fastapi import Depends, HTTPException
from fastapi import security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
_token_service = Token()
security = HTTPBearer(auto_error=False)

def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Validates the Bearer JWT from the Authorization header.
    Returns the decoded payload (includes userName, session_id, exp).
    Raises 401 if missing, expired, or invalid.
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

    return payload

def assert_token_matches_emp(token_payload: dict, emp_code: int):
    """
    Compares token's userName against the emp_code in the request body.
    Strips whitespace and normalises to string to avoid type/format mismatches.
    Raises 403 immediately if they don't match.
    """
    token_user   = str(token_payload.get("userName", "")).strip()
    request_user = str(emp_code).strip()

    if not token_user:
        raise HTTPException(
            status_code=403,
            detail="Token does not contain a userName claim."
        )

    if token_user != request_user:
        raise HTTPException(
            status_code=403,
            detail=f"Token identity '{token_user}' does not match emp_code '{request_user}'."
        )