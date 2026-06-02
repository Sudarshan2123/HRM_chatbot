import hashlib
import hmac
import uuid
from fastapi import Request
import pytz
import jwt
import logging

from datetime import datetime, timedelta
from typing import Optional
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from src.ColdStart.singleton import Init
from src.logging import logger

MAX_TOKEN_EXPIRY_MINUTES = 480  # 8-hour hard ceiling


class Token:
    def __init__(self):
        self.config = Init()

    # ── Token Creation ────────────────────────────────────────────────────────

    def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        requested_minutes = (
            expires_delta.total_seconds() / 60
            if expires_delta else 15
        )
        safe_minutes = min(requested_minutes, MAX_TOKEN_EXPIRY_MINUTES)
        now    = datetime.now(pytz.utc)
        expire = now + timedelta(minutes=safe_minutes)

        to_encode = data.copy()
        to_encode.update({
            "exp": expire,
            "iat": now,
            "jti": str(uuid.uuid4()),   # unique per token
            "iss": "macom-hrms",
        })
        return jwt.encode(
            to_encode,
            self.config.SECRET_KEY,
            algorithm=self.config.ALGORITHM,
        )

    def create_update_token(self, data: dict) -> str:
        expires = timedelta(minutes=self.config.ACCESS_TOKEN_EXPIRE_MINUTES)
        return self.create_access_token(
            data={
                "userName":   data["userName"],
                "session_id": data["session_id"],
            },
            expires_delta=expires,
        )

    # ── Token Validation ──────────────────────────────────────────────────────

    def validate_access_token(self, token: str) -> Optional[dict]:
        try:
            decoded = jwt.decode(
                token,
                self.config.SECRET_KEY,
                algorithms=[self.config.ALGORITHM],
                issuer="macom-hrms",
            )

            if not decoded.get("userName"):
                logger.warning("[Token] Missing userName claim.")
                return None

            if not decoded.get("session_id"):
                logger.warning("[Token] Missing session_id claim.")
                return None

            return decoded

        except ExpiredSignatureError:
            logger.warning("[Token] Token expired.")
            return None
        except InvalidTokenError as e:
            logger.warning(f"[Token] Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"[Token] Unexpected validation error: {e}")
            return None

    def get_user_name_from_access_token(self, access_token: str) -> Optional[str]:
        try:
            decoded = jwt.decode(
                access_token,
                self.config.SECRET_KEY,
                algorithms=[self.config.ALGORITHM],
            )
            return decoded.get("userName")
        except (ExpiredSignatureError, InvalidTokenError) as e:
            logger.warning(f"[Token] Decode failed: {e}")
            return None

    # ── Hash Helper ───────────────────────────────────────────────────────────

    @staticmethod
    def hash_token(token: str) -> str:
        """SHA-256 hash of raw token — store this, never the raw token."""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def get_emp_code_from_request(self,  request: Request) -> str:
        """Extract emp_code from Bearer token for rate limiting."""
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            try:
                decoded = self.validate_access_token(auth.split(" ", 1)[1])
                if decoded and decoded.get("emp_code"):
                    return str(decoded["emp_code"])
            except Exception:
                pass
        # fallback to real client IP (handles load balancer correctly)
        forwarded = request.headers.get("X-Forwarded-For")
        return forwarded.split(",")[0].strip() if forwarded else (request.client.host or "unknown")