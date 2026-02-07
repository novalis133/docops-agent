"""Configuration management for DocOps Agent."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ElasticsearchSettings(BaseSettings):
    """Elasticsearch connection settings."""

    model_config = SettingsConfigDict(env_prefix="ES_")

    host: str = Field(default="localhost", description="Elasticsearch host")
    port: int = Field(default=9200, description="Elasticsearch port")
    scheme: str = Field(default="http", description="Connection scheme (http/https)")
    username: Optional[str] = Field(default=None, description="Elasticsearch username")
    password: Optional[str] = Field(default=None, description="Elasticsearch password")
    api_key: Optional[str] = Field(default=None, description="Elasticsearch API key")
    cloud_id: Optional[str] = Field(default=None, description="Elastic Cloud ID")
    verify_certs: bool = Field(default=True, description="Verify SSL certificates")
    ca_certs: Optional[str] = Field(default=None, description="Path to CA certificates")

    # Index names
    documents_index: str = Field(default="docops-documents", description="Documents index name")
    alerts_index: str = Field(default="docops-alerts", description="Alerts index name")
    chunks_index: str = Field(default="docops-chunks", description="Chunks index name")

    @property
    def connection_url(self) -> str:
        """Build Elasticsearch connection URL."""
        if self.cloud_id:
            return self.cloud_id
        return f"{self.scheme}://{self.host}:{self.port}"


class GradientSettings(BaseSettings):
    """DigitalOcean Gradient AI settings."""

    model_config = SettingsConfigDict(env_prefix="GRADIENT_")

    api_key: Optional[str] = Field(default=None, description="Gradient API key")
    workspace_id: Optional[str] = Field(default=None, description="Gradient workspace ID")
    base_url: str = Field(
        default="https://api.gradient.ai/api", description="Gradient API base URL"
    )
    embedding_model: str = Field(
        default="bge-large", description="Model for generating embeddings"
    )
    inference_endpoint: Optional[str] = Field(
        default=None, description="Custom inference endpoint URL"
    )


class EmbeddingSettings(BaseSettings):
    """Embedding generation settings."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    provider: str = Field(
        default="sentence-transformers",
        description="Embedding provider (sentence-transformers, gradient, openai)",
    )
    model_name: str = Field(
        default="all-MiniLM-L6-v2", description="Model name for embeddings"
    )
    dimension: int = Field(default=384, description="Embedding vector dimension")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")


class ChunkingSettings(BaseSettings):
    """Document chunking settings."""

    model_config = SettingsConfigDict(env_prefix="CHUNK_")

    max_chunk_size: int = Field(default=1000, description="Maximum chunk size in characters")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size in characters")
    overlap: int = Field(default=100, description="Overlap between chunks in characters")
    respect_sections: bool = Field(
        default=True, description="Respect document section boundaries"
    )


class AppSettings(BaseSettings):
    """Application-level settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="DocOps Agent", description="Application name")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    # Streamlit settings
    streamlit_port: int = Field(default=8501, description="Streamlit port")

    # Sub-settings
    elasticsearch: ElasticsearchSettings = Field(default_factory=ElasticsearchSettings)
    gradient: GradientSettings = Field(default_factory=GradientSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)


@lru_cache
def get_settings() -> AppSettings:
    """Get cached application settings."""
    return AppSettings()


# Convenience accessor
settings = get_settings()
