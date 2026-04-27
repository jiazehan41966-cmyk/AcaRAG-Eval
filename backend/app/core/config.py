from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    project_name: str = "Agentic RAG Eval Platform"
    api_prefix: str = "/api"
    environment: str = "dev"

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    parsed_dir: Path = Path("data/parsed")
    index_dir: Path = Path("data/index")
    state_dir: Path = Path("data/state")
    golden_set_dir: Path = Path("data/golden_set")
    eval_reports_dir: Path = Path("data/eval_reports")

    database_url: str | None = None
    redis_url: str = "redis://localhost:6379/0"

    failure_thresholds_path: Path = Path("backend/config/failure_thresholds.json")
    regression_gate_path: Path = Path("backend/config/regression_gate.json")

    default_chunk_size: int = 600
    default_chunk_overlap: int = 120
    default_top_k: int = 5
    embedding_dim: int = 384

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "paper_chunks"
    qdrant_prefer_grpc: bool = False
    qdrant_fallback_local: bool = True
    qdrant_local_path: Path = Path("data/index/qdrant_local")

    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    reranker_model_name: str = "BAAI/bge-reranker-base"
    force_mock_embedding: bool = False

    llm_provider: str = "none"
    llm_model_name: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    default_golden_set_path: Path = Path("data/golden_set/golden_set.jsonl")
    enable_ragas: bool = False
    enable_deepeval: bool = False

    mcp_mode: str = "inprocess"
    mcp_server_url: str = "http://127.0.0.1:8765/mcp"
    mcp_timeout_seconds: int = 15

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_tracing_enabled: bool = True

    allowed_extensions: tuple[str, ...] = (".pdf", ".md", ".txt")

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.raw_dir,
            self.parsed_dir,
            self.index_dir,
            self.state_dir,
            self.golden_set_dir,
            self.eval_reports_dir,
            self.qdrant_local_path,
            self.failure_thresholds_path.parent,
            self.regression_gate_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
