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

    default_chunk_size: int = 600
    default_chunk_overlap: int = 120
    default_top_k: int = 5
    embedding_dim: int = 384

    allowed_extensions: tuple[str, ...] = (".pdf", ".md", ".txt")

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.raw_dir, self.parsed_dir, self.index_dir, self.state_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
