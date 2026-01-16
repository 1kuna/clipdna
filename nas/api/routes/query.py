from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from config import settings
from services.database import get_db
from services.job_queue import enqueue_job

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    clip_path: str


class QueryResponse(BaseModel):
    job_id: str
    status: str
    message: str


class MatchResult(BaseModel):
    source_video_id: int
    source_path: str
    confidence_score: float
    start_time: Optional[float]
    end_time: Optional[float]
    match_method: str
    audio_match: Optional[dict]


@router.post("/", response_model=QueryResponse)
async def submit_query(request: QueryRequest) -> QueryResponse:
    db = get_db()

    clip_path = Path(settings.videos_path) / "clips" / request.clip_path
    if not clip_path.exists():
        raise HTTPException(404, f"Clip not found: {request.clip_path}")

    clip_id = await db.create_query_clip(filepath=str(clip_path), filename=clip_path.name)

    job_id = str(uuid.uuid4())
    await enqueue_job(
        job_id=job_id,
        job_type="query_clip",
        payload={"clip_path": str(clip_path), "clip_id": clip_id},
    )

    return QueryResponse(
        job_id=job_id,
        status="queued",
        message=f"Query job submitted for {request.clip_path}",
    )


@router.post("/upload", response_model=QueryResponse)
async def upload_and_query(file: UploadFile = File(...)) -> QueryResponse:
    db = get_db()

    upload_dir = Path(settings.videos_path) / "clips" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    clip_path = upload_dir / filename

    with open(clip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    clip_id = await db.create_query_clip(filepath=str(clip_path), filename=file.filename)

    job_id = str(uuid.uuid4())
    await enqueue_job(
        job_id=job_id,
        job_type="query_clip",
        payload={"clip_path": str(clip_path), "clip_id": clip_id},
    )

    return QueryResponse(
        job_id=job_id,
        status="queued",
        message=f"Uploaded and queued: {file.filename}",
    )


@router.get("/results/{clip_id}", response_model=List[MatchResult])
async def get_results(clip_id: int) -> List[MatchResult]:
    db = get_db()

    matches = await db.get_matches_for_clip(clip_id)
    if not matches:
        raise HTTPException(404, f"No results found for clip {clip_id}")

    results: list[MatchResult] = []
    for match in matches:
        source = await db.get_source_video(match["source_video_id"])
        results.append(
            MatchResult(
                source_video_id=match["source_video_id"],
                source_path=source["filepath"] if source else "unknown",
                confidence_score=match["confidence_score"],
                start_time=match["source_start_time"],
                end_time=match["source_end_time"],
                match_method=match["match_method"],
                audio_match=None,
            )
        )

    return results
