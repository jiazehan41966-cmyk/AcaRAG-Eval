from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(prefix="/ui", tags=["ui"])


@router.get("/evaluation")
def evaluation_page():
    root = Path(__file__).resolve().parents[3]
    page = root / "frontend" / "app" / "evaluation" / "index.html"
    if page.exists():
        return FileResponse(page)
    return {
        "status": "not_found",
        "message": f"evaluation page not found: {page}",
    }
