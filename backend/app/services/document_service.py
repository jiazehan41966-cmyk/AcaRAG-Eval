from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.core.config import get_settings
from app.models.document import DocumentRecord
from app.services.storage_service import state_store


class DocumentService:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def upload_document(self, file: UploadFile) -> DocumentRecord:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in self.settings.allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

        doc_id = uuid4().hex
        save_path = self.settings.raw_dir / f"{doc_id}{suffix}"
        content = await file.read()
        save_path.write_bytes(content)

        record = DocumentRecord(
            id=doc_id,
            filename=file.filename or save_path.name,
            file_path=str(save_path),
            file_ext=suffix,
            metadata={"size_bytes": len(content)},
        )
        state_store.save_document(record)
        return record

    def get_document(self, doc_id: str) -> DocumentRecord:
        record = state_store.get_document(doc_id)
        if not record:
            raise HTTPException(status_code=404, detail="Document not found")
        return record

    def update_document(self, record: DocumentRecord) -> DocumentRecord:
        state_store.save_document(record)
        return record

    def list_documents(self) -> list[DocumentRecord]:
        return state_store.list_documents()

    def list_chunks(self, doc_id: str):
        return state_store.get_chunks(doc_id)


document_service = DocumentService()
