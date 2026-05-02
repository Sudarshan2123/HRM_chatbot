import re
import logging
from typing import Optional
from src.utils.database import execute_query_single, execute_query

logger = logging.getLogger(__name__)

_HASH_RE   = re.compile(r"^[a-f0-9]{32}$", re.IGNORECASE)
_ZOHO_BASE = "https://mail-sending-replies-60069513271.zohomcp.in/mcp"


def _normalize_url(raw: str) -> str:
    raw = raw.strip().rstrip("/")

    # Case 1: bare 32-char hash
    if _HASH_RE.match(raw):
        url = f"{_ZOHO_BASE}/{raw}/message"
        logger.info("Normalized bare hash → %s", url)
        return url

    # Case 2: missing protocol
    if not raw.startswith("http://") and not raw.startswith("https://"):
        raw = "https://" + raw

    # Case 3: ensure it ends with /message
    if not raw.endswith("/message"):
        raw = raw + "/message"

    logger.info("Normalized URL → %s", raw)
    return raw


async def save_zoho_key(emp_code: int, raw_url: str) -> None:
    clean_url = _normalize_url(raw_url)
    logger.info("[zoho:%s] Saving URL to DB: %s", emp_code, clean_url)
    await execute_query(
        """
        INSERT INTO zoho_user_keys (emp_code, zoho_key, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (emp_code) DO UPDATE
            SET zoho_key   = EXCLUDED.zoho_key,
                updated_at = NOW()
        """,
        (emp_code, clean_url),
        fetch=False,
    )
    logger.info("[zoho:%s] URL saved successfully", emp_code)


async def get_zoho_key(emp_code: int) -> Optional[str]:
    row = await execute_query_single(
        "SELECT zoho_key FROM zoho_user_keys WHERE emp_code = %s",
        (emp_code,)
    )
    url = row["zoho_key"] if row else None
    logger.info("[zoho:%s] Loaded URL from DB: %s", emp_code, url)
    return url


async def has_zoho_key(emp_code: int) -> bool:
    return (await get_zoho_key(emp_code)) is not None