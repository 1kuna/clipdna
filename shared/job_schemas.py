from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class IndexVideoPayload(BaseModel):
    video_path: str
    video_id: int
    fps: float = Field(default=1.0, ge=0.1, le=60.0)
    save_embeddings: bool = True


class QueryClipPayload(BaseModel):
    clip_path: str
    clip_id: int
    fps: float = Field(default=1.0, ge=0.1, le=60.0)
    top_k_frames: int = Field(default=100, ge=1)
    top_k_videos: int = Field(default=50, ge=1)
    audio_threshold: float = Field(default=0.4, ge=0.0, le=1.0)


class RebuildIndexPayload(BaseModel):
    embeddings_path: str
    index_path: str


class JobEnvelope(BaseModel):
    job_id: str
    job_type: Literal["index_video", "query_clip", "rebuild_index"]
    payload: dict[str, Any]
