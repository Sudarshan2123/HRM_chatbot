import re
import logging
from typing import Optional
from src.utils.database import execute_query_single, execute_query
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_HASH_RE   = re.compile(r"^[a-f0-9]{32}$", re.IGNORECASE)
_ZOHO_BASE = "https://mail-sending-replies-60069513271.zohomcp.in/mcp"

# SECURITY: Whitelist allowed Zoho domains to prevent SSRF
_ZOHO_ALLOWED_DOMAINS = {
    "mail-sending-replies-60069513271.zohomcp.in",
    "zohomcp.in",
    "zoho.com",
}


def _validate_url(url: str) -> bool:
    """
    Validates that URL is a Zoho MCP endpoint.
    Returns True if valid, False otherwise.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check against whitelist
        if not any(domain.endswith(allowed) for allowed in _ZOHO_ALLOWED_DOMAINS):
            logger.warning(f"URL domain {domain} not in whitelist")
            return False
        
        # Must use HTTPS
        if parsed.scheme != "https":
            logger.warning(f"URL scheme {parsed.scheme} not HTTPS")
            return False
        
        return True
    except Exception as e:
        logger.error(f"URL validation error: {e}")
        return False


def _normalize_url(raw: str) -> str:
    raw = raw.strip().rstrip("/")

    # Case 1: bare 32-char hash
    if _HASH_RE.match(raw):
        url = f"{_ZOHO_BASE}/{raw}/message"
        logger.info("Normalized bare hash → %s (hash redacted)", "***")
        return url

    # Case 2: missing protocol
    if not raw.startswith("http://") and not raw.startswith("https://"):
        raw = "https://" + raw

    # Case 3: ensure it ends with /message
    if not raw.endswith("/message"):
        raw = raw + "/message"

    # SECURITY: Validate URL before returning
    if not _validate_url(raw):
        raise ValueError(f"URL failed security validation — not a whitelisted Zoho domain")

    logger.info("Normalized URL (domain redacted for security)")
    return raw


async def save_zoho_key(emp_code: int, raw_url: str) -> None:
    clean_url = _normalize_url(raw_url)
    logger.info("[zoho:%s] Saving MCP URL to DB", emp_code)  # SECURITY: Don't log the actual URL
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
    logger.info("[zoho:%s] MCP URL saved successfully", emp_code)


async def get_zoho_key(emp_code: int) -> Optional[str]:
    row = await execute_query_single(
        "SELECT zoho_key FROM zoho_user_keys WHERE emp_code = %s",
        (emp_code,)
    )
    url = row["zoho_key"] if row else None
    # SECURITY: Don't log the actual URL in production
    logger.debug("[zoho:%s] MCP URL loaded (domain redacted for security)", emp_code)
    return url


async def has_zoho_key(emp_code: int) -> bool:
    return (await get_zoho_key(emp_code)) is not None