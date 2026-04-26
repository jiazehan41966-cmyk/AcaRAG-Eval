from fastapi import APIRouter, Query

from app.services.storage_service import state_store

router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("")
def list_traces(limit: int = Query(default=50, ge=1, le=500)):
    return state_store.list_traces(limit=limit)


@router.get("/{trace_id}")
def get_trace(trace_id: str):
    trace = state_store.get_trace(trace_id)
    if trace is None:
        return {
            "trace_id": trace_id,
            "status": "not_found",
            "message": "trace not found",
        }
    return trace
