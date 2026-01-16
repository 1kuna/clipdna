from __future__ import annotations

from fastapi import APIRouter, HTTPException

from models.schemas import JobStatus
from services.job_queue import get_job_status

router = APIRouter(prefix="/status", tags=["status"])


@router.get("/{job_id}", response_model=JobStatus)
async def get_status(job_id: str) -> JobStatus:
    status = await get_job_status(job_id)
    if not status:
        raise HTTPException(404, f"Job not found: {job_id}")
    return JobStatus(**status)
