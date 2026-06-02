# src/pipeline/Login.py

import uuid
import pytz
import httpx
import os

from datetime import datetime, timedelta
from typing import Optional

from src.ColdStart.singleton import Init
from src.constants.token import Token
from src.entity.DBHelper import execute_query,execute_query_single
from src.server.session_manager import SessionManager
from src.logging import logger


# ── Constants ──────────────────────────────────────────────────────────────────

LDAP_LOGIN_URL = os.getenv(
    "LDAP_LOGIN_URL", 
    "https://docker.mactech.net.in:5013/ldap-service/loginthroughEmail"
)
LDAP_TIMEOUT   = float(os.getenv("LDAP_TIMEOUT", "10.0"))  # seconds

# SECURITY: Validate LDAP URL at startup
if not LDAP_LOGIN_URL.startswith("https://"):
    logger.warning("LDAP_LOGIN_URL does not use HTTPS — this is a security risk")
if not LDAP_LOGIN_URL:
    raise RuntimeError("LDAP_LOGIN_URL environment variable must be set")


class Login:
    def __init__(self):
        try:
            self.config = Init()
            self.token  = Token()
        except Exception as e:
            logger.error(f"[Login] Initialization failed: {e}")
            raise

    async def login_user(
        self,
        userName:   str,
        password:   str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> dict:
        """
        Full login flow:
          1. Validate credentials against LDAP service
          2. Upsert employee into PostgreSQL (satisfies FK)
          3. Create JWT + session
        """

        # ── Step 1: Call LDAP service ─────────────────────────────────────────
        ldap_result = await _call_ldap(userName, password)

        if ldap_result is None:
            # Network/timeout error — already logged in _call_ldap
            await SessionManager._audit(
                event_type = "login_failed",
                ip_address = ip_address,
                detail     = "ldap_service_unreachable",
            )
            return {
                "status":  "error",
                "message": "Authentication service unavailable. Please try again.",
            }

        status_code, user_status = ldap_result

        if status_code != 200 or user_status not in ("active", "Inactive"):
            logger.warning(
                f"[Login] LDAP rejected emp={userName} "
                f"status_code={status_code} user_status={user_status}"
            )
            await SessionManager._audit(
                event_type = "login_failed",
                ip_address = ip_address,
                detail     = f"ldap_status={status_code} user_status={user_status}",
            )
            # Generic message — never reveal why
            return {"status": "error", "message": "Invalid credentials."}

        # ── Step 2: Upsert employee into PostgreSQL ───────────────────────────
        # Must happen before session insert — FK requires emp_code to exist
        upserted = await _upsert_employee(
            emp_code   = userName,
            user_status = user_status,
        )
        if not upserted:
            return {
                "status":  "error",
                "message": "Login failed. Please try again.",
            }

        # ── Step 3: Create token ──────────────────────────────────────────────
        session_id = str(uuid.uuid4())
        expires_at = datetime.now(pytz.utc) + timedelta(
            minutes=self.config.ACCESS_TOKEN_EXPIRE_MINUTES
        )

        # Active vs Inactive get different tokens
        if user_status == "active":
            token_data = {
                "userName":   str(userName),
                "session_id": session_id,
                "status":     "active",
            }
        else:
            # Inactive — OTP required
            # Give shorter expiry — only long enough to complete OTP
            expires_at = datetime.now(pytz.utc) + timedelta(minutes=10)
            token_data = {
                "userName":   str(userName),
                "session_id": session_id,
                "status":     "pending_otp",   # scope-limited token
            }
            logger.warning(f"[Login] emp={userName} is Inactive — OTP token issued.")

        access_token = self.token.create_access_token(
            data          = token_data,
            expires_delta = expires_at - datetime.now(pytz.utc),
        )

        # ── Step 4: Store session in PostgreSQL ───────────────────────────────
        # Only store session for active users
        # Inactive users get a short OTP token but no persistent session
        if user_status == "active":
            created = await SessionManager.create_session(
                emp_code   = int(userName),
                session_id = session_id,
                token      = access_token,
                expires_at = expires_at,
                ip_address = ip_address,
                user_agent = user_agent,
            )
            if not created:
                return {
                    "status":  "error",
                    "message": "Login failed. Please try again.",
                }

        logger.info(
            f"[Login] emp={userName} logged in "
            f"status={user_status} ip={ip_address}"
        )

        return {
            "status":       "success",
            "access_token": access_token,
            "token_type":   "bearer",
            "user_Status":  user_status,
        }


# ── LDAP Caller ────────────────────────────────────────────────────────────────

async def _call_ldap(
    userName: str,
    password: str,
) -> Optional[tuple[int, str]]:
    """
    Async call to LDAP service.
    Returns (status_code, user_status) or None on network error.

    Uses httpx (async) instead of requests (sync).
    verify=True — never disable SSL in production.
    """
    try:
        async with httpx.AsyncClient(
            timeout = LDAP_TIMEOUT,
            verify  = True,            # ← SSL verification ON
        ) as client:
            response = await client.post(
                url     = LDAP_LOGIN_URL,
                json    = {"userName": userName, "password": password},
                headers = {"Content-Type": "application/json"},
            )

        # Don't raise_for_status — we handle status codes ourselves
        user_status = response.json().get("status", "")
        logger.info(
            f"[Login] LDAP response emp={userName} "
            f"http={response.status_code} user_status={user_status}"
        )
        return response.status_code, user_status

    except httpx.TimeoutException:
        logger.error(f"[Login] LDAP service timed out for emp={userName}")
        return None
    except httpx.ConnectError:
        logger.error(f"[Login] LDAP service unreachable for emp={userName}")
        return None
    except Exception as e:
        logger.error(f"[Login] LDAP call failed for emp={userName}: {e}")
        return None


# ── Employee Upsert ────────────────────────────────────────────────────────────

@staticmethod
async def _upsert_employee(
    emp_code:    str,
    user_status: str,
) -> bool:
    try:
        is_active = user_status == "active"

        # Split into two separate simple queries
        # to avoid any psycopg3 parsing issue with ON CONFLICT

        # Step 1 — check if employee already exists
        existing = await execute_query_single(
            "SELECT emp_code FROM employees WHERE emp_code = %s",
            (int(emp_code),)
        )

        if existing:
            # Step 2a — update existing employee
            await execute_query(
                "UPDATE employees SET is_active = %s, updated_at = NOW() WHERE emp_code = %s",
                (is_active, int(emp_code)),
                fetch=False,
            )
        else:
            # Step 2b — insert new employee
            await execute_query(
                "INSERT INTO employees (emp_code, emp_name, email, password_hash, is_active) VALUES (%s, %s, %s, %s, %s)",
                (int(emp_code), str(emp_code), f"{emp_code}@macom.local", "EXTERNAL_AUTH", is_active),
                fetch=False,
            )

        logger.info(f"[Login] Employee upserted emp={emp_code} is_active={is_active}")
        return True

    except Exception as e:
        logger.error(f"[Login] _upsert_employee failed emp={emp_code}: {e}")
        return False