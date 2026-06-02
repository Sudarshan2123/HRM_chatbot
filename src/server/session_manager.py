# src/server/session_manager.py
# ONLY change: all $1/$2/etc → %s to match psycopg3

import asyncio
import hashlib
from datetime import datetime
from typing import Optional

from src.entity.DBHelper import execute_query, execute_query_single
from src.utils.database import db_transaction, get_db_pool
from src.logging import logger


class SessionManager:

    @staticmethod
    async def create_session(
        emp_code:   int,
        session_id: str,
        token:      str,
        expires_at: datetime,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> bool:
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        try:
            async with db_transaction() as conn:
                # Step 1 — revoke previous sessions
                await conn.execute(
                    """
                    UPDATE employee_sessions
                    SET    is_active      = FALSE,
                           revoked_at     = NOW(),
                           revoked_reason = 'new_login'
                    WHERE  emp_code  = %s
                    AND    is_active = TRUE
                    """,
                    (emp_code,),
                )

                # Step 2 — insert new session
                await conn.execute(
                    """
                    INSERT INTO employee_sessions
                        (emp_code, session_id, token_hash,
                         expires_at, ip_address, user_agent)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (emp_code, session_id, token_hash,
                     expires_at, ip_address, user_agent),
                )

                # Step 3 — audit log inside same transaction
                await conn.execute(
                    """
                    INSERT INTO auth_audit_log
                        (emp_code, event_type, session_id,
                         ip_address, user_agent)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (emp_code, "login_success", session_id,
                     ip_address, user_agent),
                )

            logger.info(f"[SessionManager] Session created emp={emp_code}")
            return True

        except Exception as e:
            logger.error(f"[SessionManager] create_session failed: {e}")
            return False

    @staticmethod
    async def is_session_active(session_id: str) -> bool:
        try:
            row = await execute_query_single(
                """
                SELECT emp_code
                FROM   employee_sessions
                WHERE  session_id = %s
                AND    is_active  = TRUE
                AND    expires_at > NOW()
                """,
                (session_id,)
            )
            return row is not None
        except Exception as e:
            logger.error(f"[SessionManager] is_session_active failed: {e}")
            return False

    @staticmethod
    async def touch_session(session_id: str) -> None:
        async def _touch():
            try:
                await execute_query(
                    """
                    UPDATE employee_sessions
                    SET    last_used_at = NOW()
                    WHERE  session_id  = %s
                    AND    is_active   = TRUE
                    """,
                    (session_id,),
                    fetch=False,
                )
            except Exception as e:
                logger.warning(f"[SessionManager] touch_session failed: {e}")

        asyncio.create_task(_touch())

    @staticmethod
    async def revoke_session(
        session_id: str,
        ip_address: Optional[str] = None,
        reason:     str = "logout",
    ) -> bool:
        try:
            async with db_transaction() as conn:
                # Atomic UPDATE + get emp_code back
                cur = await conn.execute(
                    """
                    UPDATE employee_sessions
                    SET    is_active      = FALSE,
                           revoked_at     = NOW(),
                           revoked_reason = %s
                    WHERE  session_id = %s
                    AND    is_active  = TRUE
                    RETURNING emp_code
                    """,
                    (reason, session_id),
                )
                row = await cur.fetchone()

                if row:
                    await conn.execute(
                        """
                        INSERT INTO auth_audit_log
                            (emp_code, event_type, session_id, ip_address)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (row["emp_code"], "logout", session_id, ip_address),
                    )

            logger.info(f"[SessionManager] Revoked: {str(session_id)[:8]}...")
            return True

        except Exception as e:
            logger.error(f"[SessionManager] revoke_session failed: {e}")
            return False

    @staticmethod
    async def revoke_all_sessions(
        emp_code:   int,
        reason:     str = "admin_revoke",
        ip_address: Optional[str] = None,
    ) -> bool:
        try:
            async with db_transaction() as conn:
                await conn.execute(
                    """
                    UPDATE employee_sessions
                    SET    is_active      = FALSE,
                           revoked_at     = NOW(),
                           revoked_reason = %s
                    WHERE  emp_code  = %s
                    AND    is_active = TRUE
                    """,
                    (reason, emp_code),
                )
                await conn.execute(
                    """
                    INSERT INTO auth_audit_log
                        (emp_code, event_type, ip_address, detail)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (emp_code, "all_sessions_revoked",
                     ip_address, f"reason={reason}"),
                )

            logger.info(f"[SessionManager] All sessions revoked emp={emp_code}")
            return True

        except Exception as e:
            logger.error(f"[SessionManager] revoke_all_sessions failed: {e}")
            return False

    @staticmethod
    async def get_active_sessions(emp_code: int) -> list:
        try:
            rows = await execute_query(
                """
                SELECT session_id, issued_at, last_used_at,
                       ip_address, user_agent, expires_at
                FROM   employee_sessions
                WHERE  emp_code  = %s
                AND    is_active = TRUE
                AND    expires_at > NOW()
                ORDER  BY issued_at DESC
                """,
                (emp_code,)
            )
            if not rows:
                return []

            return [
                {
                    "session_preview": str(r["session_id"])[:8] + "***",
                    "issued_at":       r["issued_at"].isoformat(),
                    "last_used_at":    r["last_used_at"].isoformat() if r["last_used_at"] else None,
                    "ip_address":      r["ip_address"],
                    "user_agent":      r["user_agent"],
                    "expires_at":      r["expires_at"].isoformat(),
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"[SessionManager] get_active_sessions failed: {e}")
            return []

    @staticmethod
    async def cleanup_expired_sessions() -> bool:
        BATCH_SIZE = 1000
        total      = 0

        try:
            while True:
                rows = await execute_query(
                    """
                    DELETE FROM employee_sessions
                    WHERE id IN (
                        SELECT id FROM employee_sessions
                        WHERE  expires_at < NOW() - INTERVAL '7 days'
                        LIMIT  %s
                    )
                    RETURNING id
                    """,
                    (BATCH_SIZE,)
                )
                batch = len(rows) if rows else 0
                total += batch

                if batch < BATCH_SIZE:
                    break

                await asyncio.sleep(0.1)

            logger.info(f"[SessionManager] Cleanup done — {total} deleted.")
            return True

        except Exception as e:
            logger.error(f"[SessionManager] cleanup failed: {e}")
            return False

    @staticmethod
    async def _audit(
        event_type: str,
        emp_code:   Optional[int] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        detail:     Optional[str] = None,
        _retry:     int = 2,
    ) -> None:
        for attempt in range(_retry + 1):
            try:
                await execute_query(
                    """
                    INSERT INTO auth_audit_log
                        (emp_code, event_type, session_id,
                         ip_address, user_agent, detail)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (emp_code, event_type, session_id,
                     ip_address, user_agent, detail),
                    fetch=False,
                )
                return
            except Exception as e:
                if attempt < _retry:
                    await asyncio.sleep(0.2 * (attempt + 1))
                else:
                    logger.error(
                        f"[SessionManager] AUDIT LOST after {_retry} retries — "
                        f"event={event_type} emp={emp_code} error={e}"
                    )