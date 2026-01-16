from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from config import settings
from services.database import get_db
from services.job_queue import enqueue_job

router = APIRouter(prefix="/index", tags=["index"])


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


class IndexRequest(BaseModel):
    paths: List[str] | None = None
    scan_all: bool = False


class IndexResponse(BaseModel):
    queued: int
    job_ids: List[str]


@router.post("/", response_model=IndexResponse)
async def submit_index(request: IndexRequest) -> IndexResponse:
    db = get_db()
    sources_dir = Path(settings.videos_path) / "sources"

    if request.scan_all or not request.paths:
        candidate_paths = [
            path for path in sources_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS
        ]
    else:
        candidate_paths = [sources_dir / path for path in request.paths]

    job_ids: list[str] = []
    queued = 0

    for path in candidate_paths:
        if not path.exists() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        existing = await db.get_source_by_path(str(path))
        if existing:
            continue

        video_id = await db.create_source_video(filepath=str(path), filename=path.name)
        job_id = str(uuid.uuid4())
        await enqueue_job(
            job_id=job_id,
            job_type="index_video",
            payload={"video_path": str(path), "video_id": video_id},
        )
        job_ids.append(job_id)
        queued += 1

    return IndexResponse(queued=queued, job_ids=job_ids)
