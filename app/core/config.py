"""
app/core — shared configuration.

WHY a config module?
- In production you never hardcode paths, model names, or chunk sizes.
- One place to change settings without touching business logic.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root: GEN_AI_PRACTICE/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """
    Application settings.

    For now we only need folders for uploads + parsed output.
    Later steps will add: embedding model, vector DB URL, LLM API key, etc.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Production RAG Pipeline"
    app_version: str = "0.1.0"

    # Where uploaded PDFs land
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"

    # Where we save structured parse results (JSON) for inspection/debugging
    # WHY save parsed JSON? Because as a fresher you MUST be able to OPEN
    # the parse output and visually verify quality before embedding.
    parsed_dir: Path = PROJECT_ROOT / "data" / "parsed"

    # Step 5 — chunk outputs (inspect before paying for embeddings)
    chunks_dir: Path = PROJECT_ROOT / "data" / "chunks"

    # Step 6 — embedding / vector index outputs (debug JSON still useful)
    vectors_dir: Path = PROJECT_ROOT / "data" / "vectors"

    # Qdrant vector DB
    # mode=local  → embedded on-disk DB at qdrant_path (no Docker)
    # mode=server → connect to qdrant_url (Docker/cloud)
    qdrant_mode: str = "local"
    qdrant_path: Path = PROJECT_ROOT / "data" / "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "rag_chunks"

    # Chunking defaults (chars). Override via .env if needed.
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Embedding model (fastembed). Small + local = good for learning.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Step 8 — LLM (OpenAI-compatible). Set in .env — NEVER hardcode keys here.
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    # If true, fail loudly instead of using extractive_fallback
    require_real_llm: bool = True


settings = Settings()

settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.parsed_dir.mkdir(parents=True, exist_ok=True)
settings.chunks_dir.mkdir(parents=True, exist_ok=True)
settings.vectors_dir.mkdir(parents=True, exist_ok=True)
settings.qdrant_path.mkdir(parents=True, exist_ok=True)
