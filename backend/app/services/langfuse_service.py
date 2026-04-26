from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.core.config import get_settings
from app.services.storage_service import state_store


class LangfuseService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        self._client_error: str | None = None

    def _init_client(self):
        if self._client is not None or self._client_error is not None:
            return self._client

        if not (self.settings.langfuse_public_key and self.settings.langfuse_secret_key):
            self._client_error = "LANGFUSE keys missing"
            return None

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host,
                tracing_enabled=self.settings.langfuse_tracing_enabled,
                environment=self.settings.environment,
            )
            return self._client
        except Exception as exc:  # pragma: no cover
            self._client_error = str(exc)
            return None

    def enabled(self) -> bool:
        return self._init_client() is not None

    def create_trace(self, name: str, input_payload: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        client = self._init_client()
        trace_id = client.create_trace_id() if client else uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        trace_url = client.get_trace_url(trace_id=trace_id) if client else None

        payload = {
            "trace_id": trace_id,
            "name": name,
            "status": "running",
            "input": input_payload,
            "output": None,
            "metadata": metadata or {},
            "events": [],
            "created_at": created_at,
            "updated_at": created_at,
            "langfuse_url": trace_url,
        }
        state_store.save_trace(trace_id, payload)

        if client:
            try:
                client.create_event(
                    trace_context={"trace_id": trace_id},
                    name=f"{name}.start",
                    input=input_payload,
                    metadata=metadata or {},
                )
            except Exception as exc:  # pragma: no cover
                self._client_error = str(exc)

        return payload

    def add_event(
        self,
        trace_id: str,
        name: str,
        input_payload: Any = None,
        output_payload: Any = None,
        metadata: dict[str, Any] | None = None,
        level: str | None = None,
    ) -> None:
        trace = state_store.get_trace(trace_id)
        if trace is None:
            return

        event = {
            "name": name,
            "input": input_payload,
            "output": output_payload,
            "metadata": metadata or {},
            "level": level or "DEFAULT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        trace.setdefault("events", []).append(event)
        trace["updated_at"] = event["timestamp"]
        state_store.save_trace(trace_id, trace)

        client = self._init_client()
        if client:
            try:
                client.create_event(
                    trace_context={"trace_id": trace_id},
                    name=name,
                    input=input_payload,
                    output=output_payload,
                    metadata=metadata or {},
                    level=(level or "DEFAULT"),
                )
            except Exception as exc:  # pragma: no cover
                self._client_error = str(exc)

    def finalize_trace(self, trace_id: str, output_payload: Any = None, metadata: dict[str, Any] | None = None) -> None:
        trace = state_store.get_trace(trace_id)
        if trace is None:
            return

        final_time = datetime.now(timezone.utc).isoformat()
        trace["status"] = "completed"
        trace["output"] = output_payload
        trace["final_metadata"] = metadata or {}
        trace["updated_at"] = final_time
        state_store.save_trace(trace_id, trace)

        client = self._init_client()
        if client:
            try:
                client.create_event(
                    trace_context={"trace_id": trace_id},
                    name=f"{trace.get('name', 'trace')}.end",
                    output=output_payload,
                    metadata=metadata or {},
                )
                client.flush()
            except Exception as exc:  # pragma: no cover
                self._client_error = str(exc)

    def health(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled(),
            "host": self.settings.langfuse_host,
            "error": self._client_error,
        }


langfuse_service = LangfuseService()
