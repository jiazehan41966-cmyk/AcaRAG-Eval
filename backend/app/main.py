from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.api.ui import router as ui_router
from app.core.config import get_settings
from app.core.logging import setup_logging

setup_logging()
settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    version="0.2.0",
    description="Agentic RAG automated evaluation platform for academic papers.",
)


@app.get("/")
def root():
    return {
        "name": settings.project_name,
        "version": "0.2.0",
        "docs": "/docs",
        "ui": "/ui/evaluation",
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "environment": settings.environment}


app.mount("/reports", StaticFiles(directory=str(settings.eval_reports_dir)), name="reports")

frontend_assets = Path(__file__).resolve().parents[3] / "frontend"
if frontend_assets.exists():
    app.mount("/frontend-static", StaticFiles(directory=str(frontend_assets)), name="frontend_static")

app.include_router(api_router, prefix=settings.api_prefix)
app.include_router(ui_router)
