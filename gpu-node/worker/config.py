from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str
    redis_url: str
    faiss_index_path: str = "/data/index/faiss/video.index"
    videos_path: str = "/data/videos"
    embeddings_path: str = "/data/embeddings"
    model_path: str = "/app/models/visil"
    frame_rate: float = 1.0
    batch_size: int = 32
    num_workers: int = 1
    log_level: str = "INFO"


settings = Settings()
