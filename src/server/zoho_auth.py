import httpx
import os
import logging
from langchain_core.tools import tool

ZOHO_CLIENT_ID     = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")
ZOHO_API_BASE      = "https://mail.zoho.in/api"
ZOHO_TOKEN_URL     = "https://accounts.zoho.in/oauth/v2/token"


async def _get_access_token() -> str:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            ZOHO_TOKEN_URL,
            data={
                "grant_type":    "refresh_token",
                "client_id":     ZOHO_CLIENT_ID,
                "client_secret": ZOHO_CLIENT_SECRET,
                "refresh_token": ZOHO_REFRESH_TOKEN,
            }
        )
        r.raise_for_status()
        data = r.json()
        if "access_token" not in data:
            raise RuntimeError(f"Token refresh failed: {data}")
        return data["access_token"]


async def _get_account_and_token() -> tuple[str, str]:
    token = await _get_access_token()
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{ZOHO_API_BASE}/accounts",
            headers={"Authorization": f"Zoho-oauthtoken {token}"}
        )
        r.raise_for_status()
        account_id = r.json()["data"][0]["accountId"]
    return account_id, token


@tool
async def list_emails(count: int = 10) -> str:
    """List recent emails from Zoho Mail inbox."""
    try:
        account_id, token = await _get_account_and_token()

        # Step 1: get inbox folder ID
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{ZOHO_API_BASE}/accounts/{account_id}/folders",
                headers={"Authorization": f"Zoho-oauthtoken {token}"}
            )
            r.raise_for_status()
            folders = r.json().get("data", [])

        # Find inbox folder ID
        folder_id = None
        for f in folders:
            if f.get("folderName", "").lower() == "inbox":
                folder_id = f["folderId"]
                break

        if not folder_id:
            return "Could not find inbox folder."

        # Step 2: list emails using folder ID
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{ZOHO_API_BASE}/accounts/{account_id}/messages/view",
                headers={"Authorization": f"Zoho-oauthtoken {token}"},
                params={
                    "folderId": folder_id,
                    "limit":    count
                }
            )
            r.raise_for_status()
            messages = r.json().get("data", [])

        if not messages:
            return "No messages found in inbox."

        lines = [
            f"ID: {m.get('messageId')} | "
            f"FolderID: {m.get('folderId')} | "
            f"From: {m.get('fromAddress')} | "
            f"Subject: {m.get('subject')} | "
            f"Date: {m.get('receivedTime')}"
            for m in messages
        ]
        return "\n".join(lines)

    except Exception as e:
        logging.error(f"list_emails failed: {e}")
        return f"Failed to list emails: {str(e)}"

@tool
async def read_email(message_id: str, folder_id: str) -> str:
    """
    Read a specific email content by message ID and folder ID.
    Use list_emails first to get both the message ID and folder ID.
    """
    try:
        account_id, token = await _get_account_and_token()
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{ZOHO_API_BASE}/accounts/{account_id}/folders/{folder_id}/messages/{message_id}/content",
                headers={"Authorization": f"Zoho-oauthtoken {token}"}
            )
            r.raise_for_status()
            data = r.json().get("data", {})

        return (
            f"From: {data.get('fromAddress')}\n"
            f"To: {data.get('toAddress')}\n"
            f"Subject: {data.get('subject')}\n"
            f"Date: {data.get('receivedTime')}\n"
            f"Body:\n{data.get('content', 'No content')}"
        )
    except Exception as e:
        logging.error(f"read_email failed: {e}")
        return f"Failed to read email: {str(e)}"


def get_zoho_read_tools() -> list:
    # Only read tools — send/reply come from Zoho MCP
    return [list_emails, read_email]