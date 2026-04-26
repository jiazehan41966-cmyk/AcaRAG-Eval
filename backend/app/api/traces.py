from fastapi import APIRouter

router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("/{trace_id}")
def get_trace(trace_id: str):
    return {
        "trace_id": trace_id,
        "status": "not_implemented",
        "message": "Langfuse integration will be added in the observability milestone.",
    }
