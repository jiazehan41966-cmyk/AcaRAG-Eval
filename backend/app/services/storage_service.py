from __future__ import annotations

import json
import os
import re
from hashlib import sha1
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

from app.core.config import get_settings
from app.models.document import ChunkRecord, DocumentRecord

try:  # pragma: no cover - optional dependency at import time
    from sqlalchemy import JSON, DateTime, String, Text, create_engine, delete, desc, select
    from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

    SQLALCHEMY_AVAILABLE = True
except Exception:  # pragma: no cover
    SQLALCHEMY_AVAILABLE = False
    JSON = DateTime = String = Text = object  # type: ignore[assignment,misc]
    Session = object  # type: ignore[assignment,misc]
    DeclarativeBase = object  # type: ignore[assignment,misc]
    Mapped = object  # type: ignore[assignment,misc]
    mapped_column = None  # type: ignore[assignment,misc]
    create_engine = None  # type: ignore[assignment,misc]
    delete = desc = select = None  # type: ignore[assignment,misc]
    sessionmaker = None  # type: ignore[assignment,misc]


class LocalStateStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.documents_path: Path = settings.state_dir / "documents.json"
        self.chunks_path: Path = settings.state_dir / "chunks.json"
        self.index_path: Path = settings.state_dir / "index.json"
        self.eval_runs_path: Path = settings.state_dir / "eval_runs.json"
        self.eval_baselines_path: Path = settings.state_dir / "eval_baselines.json"
        self.traces_path: Path = settings.state_dir / "traces.json"
        self.eval_jobs_path: Path = settings.state_dir / "eval_jobs.json"
        self._lock = RLock()
        self._ensure_files()

    def _ensure_files(self) -> None:
        for path, default in (
            (self.documents_path, {}),
            (self.chunks_path, {}),
            (self.index_path, {"chunk_vectors": {}, "bm25": {}, "graph_index": {}}),
            (self.eval_runs_path, {}),
            (self.eval_baselines_path, {}),
            (self.traces_path, {}),
            (self.eval_jobs_path, {}),
        ):
            if not path.exists():
                path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")

    @contextmanager
    def _file_lock(self, path: Path) -> Iterator[None]:
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)
            if os.name == "nt":  # pragma: no cover - platform-specific branch
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - exercised on non-Windows CI only
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _json_access(self, path: Path) -> Iterator[None]:
        with self._lock:
            with self._file_lock(path):
                yield

    def _read_json_unlocked(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return default
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            recovered, _ = decoder.raw_decode(content)
            backup_path = path.with_suffix(f"{path.suffix}.corrupt")
            backup_path.write_text(content, encoding="utf-8")
            self._write_json_unlocked(path, recovered)
            return recovered

    def _read_json(self, path: Path, default: Any) -> Any:
        with self._json_access(path):
            return self._read_json_unlocked(path, default)

    def _write_json_unlocked(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def _write_json(self, path: Path, payload: Any) -> None:
        with self._json_access(path):
            self._write_json_unlocked(path, payload)

    def save_document(self, document: DocumentRecord) -> None:
        with self._json_access(self.documents_path):
            docs = self._read_json_unlocked(self.documents_path, {})
            docs[document.id] = document.model_dump(mode="json")
            self._write_json_unlocked(self.documents_path, docs)

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        docs = self._read_json(self.documents_path, {})
        payload = docs.get(doc_id)
        return DocumentRecord.model_validate(payload) if payload else None

    def list_documents(self) -> list[DocumentRecord]:
        docs = self._read_json(self.documents_path, {})
        return [DocumentRecord.model_validate(item) for item in docs.values()]

    def save_chunks(self, doc_id: str, chunks: list[ChunkRecord]) -> None:
        with self._json_access(self.chunks_path):
            all_chunks = self._read_json_unlocked(self.chunks_path, {})
            all_chunks[doc_id] = [chunk.model_dump(mode="json") for chunk in chunks]
            self._write_json_unlocked(self.chunks_path, all_chunks)

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
        self._write_json(self.index_path, payload)

    def load_index(self) -> dict[str, Any]:
        return self._read_json(self.index_path, {"chunk_vectors": {}, "bm25": {}, "graph_index": {}})

    def save_eval_run(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._json_access(self.eval_runs_path):
            runs = self._read_json_unlocked(self.eval_runs_path, {})
            runs[run_id] = payload
            self._write_json_unlocked(self.eval_runs_path, runs)

    def get_eval_run(self, run_id: str) -> dict[str, Any] | None:
        runs = self._read_json(self.eval_runs_path, {})
        return runs.get(run_id)

    def list_eval_runs(self) -> list[dict[str, Any]]:
        runs = self._read_json(self.eval_runs_path, {})
        items = list(runs.values())
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return items

    def save_baseline(self, baseline_name: str, run_id: str) -> None:
        with self._json_access(self.eval_baselines_path):
            baselines = self._read_json_unlocked(self.eval_baselines_path, {})
            baselines[baseline_name] = run_id
            self._write_json_unlocked(self.eval_baselines_path, baselines)

    def get_baseline(self, baseline_name: str) -> str | None:
        baselines = self._read_json(self.eval_baselines_path, {})
        return baselines.get(baseline_name)

    def list_baselines(self) -> dict[str, str]:
        return self._read_json(self.eval_baselines_path, {})

    def save_trace(self, trace_id: str, payload: dict[str, Any]) -> None:
        with self._json_access(self.traces_path):
            traces = self._read_json_unlocked(self.traces_path, {})
            traces[trace_id] = payload
            self._write_json_unlocked(self.traces_path, traces)

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        traces = self._read_json(self.traces_path, {})
        return traces.get(trace_id)

    def list_traces(self, limit: int = 100) -> list[dict[str, Any]]:
        traces = self._read_json(self.traces_path, {})
        items = list(traces.values())
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return items[:limit]

    def save_eval_job(self, job_id: str, payload: dict[str, Any]) -> None:
        with self._json_access(self.eval_jobs_path):
            jobs = self._read_json_unlocked(self.eval_jobs_path, {})
            jobs[job_id] = payload
            self._write_json_unlocked(self.eval_jobs_path, jobs)

    def get_eval_job(self, job_id: str) -> dict[str, Any] | None:
        jobs = self._read_json(self.eval_jobs_path, {})
        return jobs.get(job_id)

    def list_eval_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        jobs = self._read_json(self.eval_jobs_path, {})
        items = list(jobs.values())
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return items[:limit]


if SQLALCHEMY_AVAILABLE:

    class Base(DeclarativeBase):
        pass


    class DocumentEntity(Base):
        __tablename__ = "documents"

        id: Mapped[str] = mapped_column(String(64), primary_key=True)
        payload: Mapped[dict] = mapped_column(JSON, nullable=False)
        updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


    class ChunkEntity(Base):
        __tablename__ = "chunks"

        chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
        doc_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
        payload: Mapped[dict] = mapped_column(JSON, nullable=False)


    class IndexEntity(Base):
        __tablename__ = "indexes"

        index_key: Mapped[str] = mapped_column(String(64), primary_key=True)
        payload: Mapped[dict] = mapped_column(JSON, nullable=False)
        updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


    class EvalRunEntity(Base):
        __tablename__ = "eval_runs"

        run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
        payload: Mapped[dict] = mapped_column(JSON, nullable=False)
        created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


    class EvalBaselineEntity(Base):
        __tablename__ = "eval_baselines"

        baseline_name: Mapped[str] = mapped_column(String(128), primary_key=True)
        run_id: Mapped[str] = mapped_column(String(64), nullable=False)


    class TraceEntity(Base):
        __tablename__ = "traces"

        trace_id: Mapped[str] = mapped_column(String(64), primary_key=True)
        payload: Mapped[dict] = mapped_column(JSON, nullable=False)
        created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


    class EvalJobEntity(Base):
        __tablename__ = "eval_jobs"

        job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
        status: Mapped[str] = mapped_column(String(32), nullable=False)
        payload: Mapped[dict] = mapped_column(JSON, nullable=False)
        error: Mapped[str | None] = mapped_column(Text, nullable=True)
        created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
        updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


    class PaperEntity(Base):
        __tablename__ = "papers"

        id: Mapped[str] = mapped_column(String(64), primary_key=True)
        title: Mapped[str | None] = mapped_column(Text, nullable=True)
        authors: Mapped[str | None] = mapped_column(Text, nullable=True)
        year: Mapped[str | None] = mapped_column(String(16), nullable=True)
        venue: Mapped[str | None] = mapped_column(Text, nullable=True)
        abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
        payload: Mapped[dict] = mapped_column(JSON, nullable=False)
        updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


    class KnowledgeEntity(Base):
        __tablename__ = "entities"

        id: Mapped[str] = mapped_column(String(96), primary_key=True)
        paper_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
        name: Mapped[str] = mapped_column(String(256), index=True, nullable=False)
        type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
        payload: Mapped[dict] = mapped_column(JSON, nullable=False)


    class KnowledgeRelation(Base):
        __tablename__ = "relations"

        id: Mapped[str] = mapped_column(String(128), primary_key=True)
        source_entity: Mapped[str] = mapped_column(String(96), index=True, nullable=False)
        target_entity: Mapped[str] = mapped_column(String(96), index=True, nullable=False)
        relation_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
        evidence_chunk_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
        payload: Mapped[dict] = mapped_column(JSON, nullable=False)


    class CitationEntity(Base):
        __tablename__ = "citations"

        id: Mapped[str] = mapped_column(String(128), primary_key=True)
        paper_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
        cited_paper_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
        context: Mapped[str] = mapped_column(Text, nullable=False)
        evidence_chunk_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
        payload: Mapped[dict] = mapped_column(JSON, nullable=False)


class SQLStateStore:
    def __init__(self, database_url: str) -> None:
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy is not available")

        self.engine = create_engine(database_url, future=True, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        Base.metadata.create_all(self.engine)
        self._lock = RLock()
        self._mirror = LocalStateStore()

    @contextmanager
    def session(self) -> Iterator[Session]:
        db = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _stable_id(*parts: str, length: int = 40) -> str:
        raw = "|".join(parts)
        return sha1(raw.encode("utf-8")).hexdigest()[:length]

    @staticmethod
    def _entity_type(name: str) -> str:
        lowered = name.lower()
        if any(marker in lowered for marker in ("bert", "gpt", "t5", "llama", "transformer")):
            return "model"
        if any(marker in lowered for marker in ("dataset", "corpus", "squad", "nq", "msmarco", "wiki")):
            return "dataset"
        if any(marker in lowered for marker in ("accuracy", "f1", "recall", "precision", "bleu", "rouge", "mrr")):
            return "metric"
        if any(marker in lowered for marker in ("retrieval", "rerank", "rag", "graph", "method")):
            return "method"
        return "concept"

    @staticmethod
    def _infer_relation_type(text: str, source_name: str, target_name: str, source_type: str, target_type: str) -> str:
        """Infer semantic relation type from surrounding text and entity types."""
        lowered = text.lower()
        uses_markers = ("使用", "采用", "利用", "based on", "utilize", "employ", "leverage", "using", "apply", "adopted")
        improves_markers = ("改进", "优化", "提升", "outperform", "improve", "enhance", "surpass", "better than", "superior")
        compares_markers = ("对比", "比较", "compare", "versus", " vs ", "contrast", "相比", "不同")
        evaluates_markers = ("evaluate", "测试", "实验", "benchmark", "test on", "evaluated on", "在.*上测试")

        if any(m in lowered for m in improves_markers):
            return "improves"
        if any(m in lowered for m in compares_markers):
            return "compares"
        if source_type in ("method", "model") and target_type in ("dataset", "metric"):
            if any(m in lowered for m in evaluates_markers):
                return "evaluates_on"
        if any(m in lowered for m in uses_markers):
            return "uses"
        return "co_occurs"

    @staticmethod
    def _citation_mentions(text: str) -> list[str]:
        bracket_refs = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", text)
        author_year = re.findall(r"\b([A-Z][A-Za-z]+(?:\s+et al\.)?,\s*(?:19|20)\d{2})\b", text)
        mentions: list[str] = []
        for ref in bracket_refs:
            mentions.extend(item.strip() for item in ref.split(",") if item.strip())
        mentions.extend(author_year)
        return mentions[:12]

    @staticmethod
    def _extract_abstract(chunks: list[ChunkRecord]) -> str | None:
        for chunk in chunks[:12]:
            text = chunk.text.strip()
            if "abstract" in (chunk.section or "").lower() or text.lower().startswith("abstract"):
                return text[:2000]
        return chunks[0].text[:1000] if chunks else None

    def _upsert_paper(self, db: Session, document: DocumentRecord, chunks: list[ChunkRecord] | None = None) -> None:
        metadata = document.metadata or {}
        year_match = re.search(r"\b(19|20)\d{2}\b", str(metadata.get("year") or metadata.get("title") or ""))
        row = db.get(PaperEntity, document.id) or PaperEntity(
            id=document.id,
            payload=document.model_dump(mode="json"),
            updated_at=self._now(),
        )
        row.title = metadata.get("title") or document.filename
        row.authors = str(metadata.get("authors") or "")
        row.year = metadata.get("year") or (year_match.group(0) if year_match else None)
        row.venue = metadata.get("venue")
        row.abstract = metadata.get("abstract") or self._extract_abstract(chunks or [])
        row.payload = document.model_dump(mode="json")
        row.updated_at = self._now()
        db.add(row)

    def _replace_structured_knowledge(self, db: Session, doc_id: str, chunks: list[ChunkRecord]) -> None:
        from app.services.graph_service import graph_service

        existing_entities = db.execute(select(KnowledgeEntity).where(KnowledgeEntity.paper_id == doc_id)).scalars().all()
        existing_entity_ids = [item.id for item in existing_entities]
        if existing_entity_ids:
            db.execute(delete(KnowledgeRelation).where(KnowledgeRelation.source_entity.in_(existing_entity_ids)))
            db.execute(delete(KnowledgeRelation).where(KnowledgeRelation.target_entity.in_(existing_entity_ids)))
        db.execute(delete(KnowledgeEntity).where(KnowledgeEntity.paper_id == doc_id))
        db.execute(delete(CitationEntity).where(CitationEntity.paper_id == doc_id))

        seen_entities: set[str] = set()
        seen_relations: set[str] = set()
        seen_citations: set[str] = set()

        for chunk in chunks:
            smart_entities = graph_service.extract_entities_smart(chunk.text, max_entities=8)
            names = [e["name"] for e in smart_entities]
            types = [e["type"] for e in smart_entities]
            entity_ids: list[str] = []
            for name, etype in zip(names, types):
                entity_id = self._stable_id(doc_id, name, length=32)
                entity_ids.append(entity_id)
                if entity_id in seen_entities:
                    continue
                seen_entities.add(entity_id)
                db.add(
                    KnowledgeEntity(
                        id=entity_id,
                        paper_id=doc_id,
                        name=name,
                        type=etype,
                        payload={"evidence_chunk_id": chunk.id, "page": chunk.page, "section": chunk.section},
                    )
                )

            entity_types_map: dict[str, str] = {}
            for eid, etype in zip(entity_ids, types):
                entity_types_map[eid] = etype

            for source, target in zip(entity_ids, entity_ids[1:]):
                relation_id = self._stable_id(doc_id, chunk.id, source, target, length=40)
                if relation_id in seen_relations:
                    continue
                seen_relations.add(relation_id)
                src_type = entity_types_map.get(source, "concept")
                tgt_type = entity_types_map.get(target, "concept")
                src_name = names[entity_ids.index(source)] if source in entity_ids else ""
                tgt_name = names[entity_ids.index(target)] if target in entity_ids else ""
                rel_type = self._infer_relation_type(chunk.text, src_name, tgt_name, src_type, tgt_type)
                db.add(
                    KnowledgeRelation(
                        id=relation_id,
                        source_entity=source,
                        target_entity=target,
                        relation_type=rel_type,
                        evidence_chunk_id=chunk.id,
                        payload={"page": chunk.page, "section": chunk.section, "relation_type": rel_type},
                    )
                )

            for mention in self._citation_mentions(chunk.text):
                citation_id = self._stable_id(doc_id, chunk.id, mention, length=40)
                if citation_id in seen_citations:
                    continue
                seen_citations.add(citation_id)
                db.add(
                    CitationEntity(
                        id=citation_id,
                        paper_id=doc_id,
                        cited_paper_id=mention,
                        context=chunk.text[:1000],
                        evidence_chunk_id=chunk.id,
                        payload={"page": chunk.page, "section": chunk.section},
                    )
                )

    def save_document(self, document: DocumentRecord) -> None:
        with self._lock, self.session() as db:
            row = db.get(DocumentEntity, document.id) or DocumentEntity(
                id=document.id,
                payload=document.model_dump(mode="json"),
                updated_at=self._now(),
            )
            row.payload = document.model_dump(mode="json")
            row.updated_at = self._now()
            db.add(row)
            self._upsert_paper(db, document)
        self._mirror.save_document(document)

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        with self.session() as db:
            row = db.get(DocumentEntity, doc_id)
            if row is None:
                return None
            return DocumentRecord.model_validate(row.payload)

    def list_documents(self) -> list[DocumentRecord]:
        with self.session() as db:
            rows = db.execute(select(DocumentEntity).order_by(desc(DocumentEntity.updated_at))).scalars().all()
            return [DocumentRecord.model_validate(row.payload) for row in rows]

    def save_chunks(self, doc_id: str, chunks: list[ChunkRecord]) -> None:
        with self._lock, self.session() as db:
            db.execute(delete(ChunkEntity).where(ChunkEntity.doc_id == doc_id))
            for chunk in chunks:
                db.add(
                    ChunkEntity(
                        chunk_id=chunk.id,
                        doc_id=doc_id,
                        payload=chunk.model_dump(mode="json"),
                    )
                )
            document = db.get(DocumentEntity, doc_id)
            if document is not None:
                self._upsert_paper(db, DocumentRecord.model_validate(document.payload), chunks)
            self._replace_structured_knowledge(db, doc_id, chunks)
        self._mirror.save_chunks(doc_id, chunks)

    def get_chunks(self, doc_id: str) -> list[ChunkRecord]:
        with self.session() as db:
            rows = db.execute(select(ChunkEntity).where(ChunkEntity.doc_id == doc_id)).scalars().all()
            payloads = [row.payload for row in rows]
            payloads.sort(key=lambda x: int(x.get("chunk_index", 0)))
            return [ChunkRecord.model_validate(item) for item in payloads]

    def list_chunks(self, doc_ids: list[str] | None = None) -> list[ChunkRecord]:
        with self.session() as db:
            stmt = select(ChunkEntity)
            if doc_ids:
                stmt = stmt.where(ChunkEntity.doc_id.in_(doc_ids))
            rows = db.execute(stmt).scalars().all()
            payloads = [row.payload for row in rows]
            payloads.sort(key=lambda x: (str(x.get("doc_id", "")), int(x.get("chunk_index", 0))))
            return [ChunkRecord.model_validate(item) for item in payloads]

    def save_index(self, payload: dict[str, Any]) -> None:
        with self._lock, self.session() as db:
            row = db.get(IndexEntity, "main") or IndexEntity(index_key="main", payload=payload, updated_at=self._now())
            row.payload = payload
            row.updated_at = self._now()
            db.add(row)
        self._mirror.save_index(payload)

    def load_index(self) -> dict[str, Any]:
        with self.session() as db:
            row = db.get(IndexEntity, "main")
            return row.payload if row else {"chunk_vectors": {}, "bm25": {}, "graph_index": {}}

    def save_eval_run(self, run_id: str, payload: dict[str, Any]) -> None:
        created_at_raw = payload.get("created_at")
        created_at = (
            datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
            if isinstance(created_at_raw, str) and created_at_raw
            else self._now()
        )
        with self._lock, self.session() as db:
            row = db.get(EvalRunEntity, run_id) or EvalRunEntity(run_id=run_id, payload=payload, created_at=created_at)
            row.payload = payload
            row.created_at = created_at
            db.add(row)
        self._mirror.save_eval_run(run_id, payload)

    def get_eval_run(self, run_id: str) -> dict[str, Any] | None:
        with self.session() as db:
            row = db.get(EvalRunEntity, run_id)
            return row.payload if row else None

    def list_eval_runs(self) -> list[dict[str, Any]]:
        with self.session() as db:
            rows = db.execute(select(EvalRunEntity).order_by(desc(EvalRunEntity.created_at))).scalars().all()
            return [row.payload for row in rows]

    def save_baseline(self, baseline_name: str, run_id: str) -> None:
        with self._lock, self.session() as db:
            row = db.get(EvalBaselineEntity, baseline_name) or EvalBaselineEntity(
                baseline_name=baseline_name,
                run_id=run_id,
            )
            row.run_id = run_id
            db.add(row)
        self._mirror.save_baseline(baseline_name, run_id)

    def get_baseline(self, baseline_name: str) -> str | None:
        with self.session() as db:
            row = db.get(EvalBaselineEntity, baseline_name)
            return row.run_id if row else None

    def list_baselines(self) -> dict[str, str]:
        with self.session() as db:
            rows = db.execute(select(EvalBaselineEntity)).scalars().all()
            return {row.baseline_name: row.run_id for row in rows}

    def save_trace(self, trace_id: str, payload: dict[str, Any]) -> None:
        created_at_raw = payload.get("created_at")
        created_at = (
            datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
            if isinstance(created_at_raw, str) and created_at_raw
            else self._now()
        )
        with self._lock, self.session() as db:
            row = db.get(TraceEntity, trace_id) or TraceEntity(trace_id=trace_id, payload=payload, created_at=created_at)
            row.payload = payload
            row.created_at = created_at
            db.add(row)
        self._mirror.save_trace(trace_id, payload)

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self.session() as db:
            row = db.get(TraceEntity, trace_id)
            return row.payload if row else None

    def list_traces(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.session() as db:
            rows = db.execute(select(TraceEntity).order_by(desc(TraceEntity.created_at)).limit(limit)).scalars().all()
            return [row.payload for row in rows]

    def save_eval_job(self, job_id: str, payload: dict[str, Any]) -> None:
        now = self._now()
        status = str(payload.get("status", "queued"))
        with self._lock, self.session() as db:
            row = db.get(EvalJobEntity, job_id) or EvalJobEntity(
                job_id=job_id,
                status=status,
                payload=payload,
                error=payload.get("error"),
                created_at=now,
                updated_at=now,
            )
            row.status = status
            row.payload = payload
            row.error = payload.get("error")
            row.updated_at = now
            db.add(row)
        self._mirror.save_eval_job(job_id, payload)

    def get_eval_job(self, job_id: str) -> dict[str, Any] | None:
        with self.session() as db:
            row = db.get(EvalJobEntity, job_id)
            return row.payload if row else None

    def list_eval_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.session() as db:
            rows = db.execute(select(EvalJobEntity).order_by(desc(EvalJobEntity.created_at)).limit(limit)).scalars().all()
            return [row.payload for row in rows]


def _build_state_store():
    settings = get_settings()
    if settings.database_url:
        try:
            return SQLStateStore(settings.database_url)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize PostgreSQL state store: {exc}") from exc
    return LocalStateStore()


state_store = _build_state_store()
