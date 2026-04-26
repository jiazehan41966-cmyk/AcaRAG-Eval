from fastapi import APIRouter

from app.api import chat, documents, eval, search, traces

api_router = APIRouter()
api_router.include_router(documents.router)
api_router.include_router(search.router)
api_router.include_router(chat.router)
api_router.include_router(eval.router)
api_router.include_router(traces.router)
