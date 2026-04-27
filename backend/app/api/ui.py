from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse

router = APIRouter(prefix="/ui", tags=["ui"])


def _page_response(page_name: str):
    root = Path(__file__).resolve().parents[3]
    page = root / "frontend" / "app" / page_name / "index.html"
    if page.exists():
        return FileResponse(page)
    return {
        "status": "not_found",
        "message": f"page not found: {page}",
    }


@router.get("")
def ui_root():
    return RedirectResponse(url="/ui/evaluation")


@router.get("/evaluation")
def evaluation_page():
    return _page_response("evaluation")


@router.get("/documents")
def documents_page():
    return _page_response("documents")


@router.get("/chat")
def chat_page():
    return _page_response("chat")


@router.get("/traces")
def traces_page():
    return _page_response("traces")
