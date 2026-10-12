from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="REPOMIND_",
        case_sensitive=False,
        extra="ignore",
    )
    
    pinecone_api_key: str = ""
    pinecone_index_name: str = "repomind-openai"
    openai_api_key: str = ""
    app_name: str = "RepoMind"
    app_version: str = "0.1.0"
    environment: Literal["development", "testing", "production"] = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
