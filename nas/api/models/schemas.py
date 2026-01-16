from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel


class JobStatus(BaseModel):
    job_id: str
    status: str
    job_type: str
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    result: dict[str, Any] | None = None


class SourceVideoOut(BaseModel):
    id: int
    filepath: str
    filename: str
    duration_seconds: float | None = None
    frame_count: int | None = None
    fps: float | None = None
    resolution: str | None = None
    index_status: str


class MatchOut(BaseModel):
    source_video_id: int
    confidence_score: float
    source_start_time: float | None = None
    source_end_time: float | None = None
    match_method: str | None = None
