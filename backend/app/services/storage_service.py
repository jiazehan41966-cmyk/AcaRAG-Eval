import json
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.config import get_settings
from app.models.document import ChunkRecord, DocumentRecord


class LocalStateStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.documents_path: Path = settings.state_dir / "documents.json"
        self.chunks_path: Path = settings.state_dir / "chunks.json"
        self.index_path: Path = settings.state_dir / "index.json"
        self.eval_runs_path: Path = settings.state_dir / "eval_runs.json"
        self._lock = Lock()
        self._ensure_files()

    def _ensure_files(self) -> None:
        for path, default in (
            (self.documents_path, {}),
            (self.chunks_path, {}),
            (self.index_path, {"chunk_vectors": {}, "bm25": {}}),
            (self.eval_runs_path, {}),
        ):
            if not path.exists():
                path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return default
        return json.loads(content)

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_document(self, document: DocumentRecord) -> None:
        with self._lock:
            docs = self._read_json(self.documents_path, {})
            docs[document.id] = document.model_dump(mode="json")
            self._write_json(self.documents_path, docs)

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        docs = self._read_json(self.documents_path, {})
        payload = docs.get(doc_id)
        return DocumentRecord.model_validate(payload) if payload else None

    def list_documents(self) -> list[DocumentRecord]:
        docs = self._read_json(self.documents_path, {})
        return [DocumentRecord.model_validate(item) for item in docs.values()]

    def save_chunks(self, doc_id: str, chunks: list[ChunkRecord]) -> None:
        with self._lock:
            all_chunks = self._read_json(self.chunks_path, {})
            all_chunks[doc_id] = [chunk.model_dump(mode="json") for chunk in chunks]
            self._write_json(self.chunks_path, all_chunks)

    def get_chunks(self, doc_id: str) -> list[ChunkRecord]:
        all_chunks = self._read_json(self.chunks_path, {})
        payload = all_chunks.get(doc_id, [])
        return [ChunkRecord.model_validate(item) for item in payload]

    def list_chunks(self, doc_ids: list[str] | None = None) -> list[ChunkRecord]:
        all_chunks = self._read_json(self.chunks_path, {})
        selected_doc_ids = doc_ids if doc_ids else list(all_chunks.keys())
        output: list[ChunkRecord] = []
        for doc_id in selected_doc_ids:
            for item in all_chunks.get(doc_id, []):
                output.append(ChunkRecord.model_validate(item))
        return output

    def save_index(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._write_json(self.index_path, payload)

    def load_index(self) -> dict[str, Any]:
        return self._read_json(self.index_path, {"chunk_vectors": {}, "bm25": {}})

    def save_eval_run(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            runs = self._read_json(self.eval_runs_path, {})
            runs[run_id] = payload
            self._write_json(self.eval_runs_path, runs)

    def get_eval_run(self, run_id: str) -> dict[str, Any] | None:
        runs = self._read_json(self.eval_runs_path, {})
        return runs.get(run_id)


state_store = LocalStateStore()
