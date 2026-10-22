"""Centralized application settings loaded from YAML and environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime configuration; environment variables override ``config.yaml``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    qdrant_url: str | None = None
    qdrant_api_key: str | None = Field(default=None, repr=False)
    qdrant_collection_name: str = "indian_legal_qa"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    llm_base_url: str | None = None
    llm_api_key: str | None = Field(default=None, repr=False)
    llm_model: str | None = None
    llm_timeout_seconds: float = Field(default=45, gt=0, le=300)
    llm_temperature: float = Field(default=0.1, ge=0, le=2)
    top_k: int = Field(default=8, ge=1, le=100)
    final_context_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float | None = Field(default=None, ge=0, le=1)
    chunk_size_chars: int = Field(default=4000, ge=256)
    chunk_overlap_chars: int = Field(default=300, ge=0)
    upsert_batch_size: int = Field(default=64, ge=1, le=2048)
    exact_index_path: str = "data/reports/exact_search.sqlite3"
    log_level: str = "INFO"

    @field_validator(
        "qdrant_url",
        "qdrant_api_key",
        "llm_base_url",
        "llm_api_key",
        "llm_model",
        mode="before",
    )
    @classmethod
    def normalize_optional_service_settings(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")
        return self

    @property
    def qdrant_configured(self) -> bool:
        return bool(self.qdrant_url and self.qdrant_api_key)

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)

    @property
    def exact_index_file(self) -> Path:
        path = Path(self.exact_index_path)
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once; credentials are read only from the environment."""
    config_path = PROJECT_ROOT / "config.yaml"
    values: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ValueError("config.yaml must contain a YAML mapping")
            values = loaded
    return Settings(**values)
