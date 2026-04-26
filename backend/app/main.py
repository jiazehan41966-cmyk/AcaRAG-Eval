from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import setup_logging

setup_logging()
settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    version="0.1.0",
    description="Agentic RAG automated evaluation platform for academic papers.",
)


@app.get("/")
def root():
    return {
        "name": settings.project_name,
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "environment": settings.environment}


app.include_router(api_router, prefix=settings.api_prefix)
