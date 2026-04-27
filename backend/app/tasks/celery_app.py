from __future__ import annotations

from app.core.config import get_settings

try:  # pragma: no cover - dependency/environment dependent
    from celery import Celery
except Exception:  # pragma: no cover
    Celery = None  # type: ignore[assignment,misc]


def _build_celery_app():
    if Celery is None:
        return None

    settings = get_settings()
    app = Celery(
        "paper_rag_agent",
        broker=settings.redis_url,
        backend=settings.redis_url,
    )
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
    )
    return app


celery_app = _build_celery_app()
