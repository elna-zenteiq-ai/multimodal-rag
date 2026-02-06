"""Application configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # NVIDIA API Configuration
    nvidia_api_key: str = ""
    nvidia_model: str = "openai/gpt-oss-120b"
    nvidia_temperature: float = 0.7
    nvidia_top_p: float = 1.0
    nvidia_max_tokens: int = 4096

    # MinIO Configuration
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"
    minio_secure: bool = False

    # Milvus Configuration
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "rag_collection"

    # Embedding Model
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Database Configuration
    database_url: str = "postgresql://raguser:ragpassword@postgres:5432/multimodal_rag"

    # Agent Configuration
    max_summary_context: int = 10  # Max summaries to pass as context
    top_k_relevant_files: int = 5  # Top K relevant files when more than max
    max_files_per_upload: int = 10  # Max files per single upload request

    @property
    def milvus_uri(self) -> str:
        """Get Milvus connection URI."""
        return f"http://{self.milvus_host}:{self.milvus_port}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
