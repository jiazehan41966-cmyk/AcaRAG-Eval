from fastapi import APIRouter

from app.schemas.search import IndexBuildRequest, QueryRequest, RerankRequest, SearchResponse
from app.services.index_service import index_service

router = APIRouter(tags=["search"])


@router.post("/index/build")
def build_index(request: IndexBuildRequest):
    return index_service.build_index(doc_ids=request.doc_ids)


@router.post("/index/rebuild")
def rebuild_index(request: IndexBuildRequest):
    return index_service.rebuild_index(doc_ids=request.doc_ids)


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
