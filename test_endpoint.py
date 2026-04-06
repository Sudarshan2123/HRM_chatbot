import os
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_HTML_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stream_test.html")


@router.get("/stream-test", response_class=HTMLResponse)
async def stream_test_ui():
    with open(_HTML_FILE, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html, media_type="text/html; charset=utf-8")