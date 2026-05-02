# ── DB Helpers ─────────────────────────────────────────────

from src.utils.database import *


async def save_user_thread(emp_code: int, thread_id: str):
    await execute_query(
        """
        INSERT INTO user_conversations (emp_code, thread_id)
        VALUES (%s, %s)
        ON CONFLICT (thread_id) DO UPDATE
            SET last_active = NOW()
        """,
        (emp_code, thread_id),
        fetch=False
    )


async def update_thread_active(thread_id: str):
    await execute_query(
        """
        UPDATE user_conversations
        SET last_active = NOW()
        WHERE thread_id = %s
        """,
        (thread_id,),
        fetch=False
    )


async def verify_thread_ownership(emp_code: int, thread_id: str) -> bool:
    result = await execute_query_single(
        """
        SELECT thread_id FROM user_conversations
        WHERE emp_code = %s AND thread_id = %s
        """,
        (emp_code, thread_id)
    )
    return result is not None
