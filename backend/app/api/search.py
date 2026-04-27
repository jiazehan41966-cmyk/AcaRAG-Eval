from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter

from app.schemas.search import (
    IndexBuildRequest,
    IndexBuildResponse,
    IndexJobResponse,
    QueryRequest,
    RerankRequest,
    SearchResponse,
)
from app.services.index_service import index_service
from app.services.storage_service import state_store
from app.tasks.index_tasks import enqueue_index_job

router = APIRouter(tags=["search"])


@router.post("/index/build", response_model=IndexBuildResponse)
def build_index(request: IndexBuildRequest):
    if request.run_async:
        job_id = uuid4().hex
        queued = enqueue_index_job(
            job_id=job_id,
            action="build",
            request_payload=request.model_dump(),
        )
        return IndexBuildResponse(
            status="queued",
            task_id=job_id,
            task_backend=queued.get("backend"),
        )

    payload = index_service.build_index(doc_ids=request.doc_ids)
    return IndexBuildResponse(
        status="completed",
        indexed_chunks=payload.get("indexed_chunks"),
        indexed_docs=payload.get("indexed_docs"),
        built_at=payload.get("built_at"),
        embedding_backend=payload.get("embedding_backend"),
        reranker_backend=payload.get("reranker_backend"),
        qdrant_mode=payload.get("qdrant_mode"),
    )


@router.post("/index/rebuild", response_model=IndexBuildResponse)
def rebuild_index(request: IndexBuildRequest):
    if request.run_async:
        job_id = uuid4().hex
        queued = enqueue_index_job(
            job_id=job_id,
            action="rebuild",
            request_payload=request.model_dump(),
        )
        return IndexBuildResponse(
            status="queued",
            task_id=job_id,
            task_backend=queued.get("backend"),
        )

    payload = index_service.rebuild_index(doc_ids=request.doc_ids)
    return IndexBuildResponse(
        status="completed",
        indexed_chunks=payload.get("indexed_chunks"),
        indexed_docs=payload.get("indexed_docs"),
        built_at=payload.get("built_at"),
        embedding_backend=payload.get("embedding_backend"),
        reranker_backend=payload.get("reranker_backend"),
        qdrant_mode=payload.get("qdrant_mode"),
    )


@router.get("/index/jobs", response_model=list[IndexJobResponse])
def list_index_jobs(limit: int = 100):
    jobs = state_store.list_eval_jobs(limit=limit * 3)
    index_jobs = [item for item in jobs if item.get("job_type") == "index"][:limit]
    return [
        IndexJobResponse(
            job_id=item.get("job_id", ""),
            status=item.get("status", "unknown"),
            action=item.get("action"),
            backend=item.get("backend"),
            celery_task_id=item.get("celery_task_id"),
            created_at=item.get("created_at"),
            updated_at=item.get("updated_at"),
            result=item.get("result"),
            error=item.get("error"),
        )
        for item in index_jobs
    ]


@router.get("/index/jobs/{job_id}", response_model=IndexJobResponse)
def get_index_job(job_id: str):
    item = state_store.get_eval_job(job_id) or {
        "job_id": job_id,
        "status": "not_found",
        "error": "job not found",
    }
    if item.get("job_type") not in {None, "index"}:
        item = {
            "job_id": job_id,
            "status": "not_found",
            "error": "index job not found",
        }

    return IndexJobResponse(
        job_id=item.get("job_id", job_id),
        status=item.get("status", "unknown"),
        action=item.get("action"),
        backend=item.get("backend"),
        celery_task_id=item.get("celery_task_id"),
        created_at=item.get("created_at"),
        updated_at=item.get("updated_at"),
        result=item.get("result"),
        error=item.get("error"),
    )


@router.post("/search/vector", response_model=SearchResponse)
def vector_search(request: QueryRequest):
    hits = index_service.vector_search(request.query, request.top_k)
    return SearchResponse(query=request.query, strategy="vector", hits=hits)


@router.post("/search/bm25", response_model=SearchResponse)
def bm25_search(request: QueryRequest):
    hits = index_service.bm25_search(request.query, request.top_k)
    return SearchResponse(query=request.query, strategy="bm25", hits=hits)


@router.post("/search/hybrid", response_model=SearchResponse)
def hybrid_search(request: QueryRequest):
    hits = index_service.hybrid_search(request.query, request.top_k, alpha=request.alpha)
    return SearchResponse(query=request.query, strategy="hybrid", hits=hits)


@router.post("/search/rerank")
def rerank(request: RerankRequest):
    ranked = index_service.rerank(request.query, request.candidates, request.top_k)
    return {
        "query": request.query,
        "strategy": "rerank",
        "hits": ranked,
    }
