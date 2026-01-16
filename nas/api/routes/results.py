from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.job_queue import get_job_status

router = APIRouter(prefix="/results", tags=["results"])


@router.get("/{job_id}")
async def get_results(job_id: str) -> dict[str, object]:
    status = await get_job_status(job_id)
    if not status:
        raise HTTPException(404, f"Job not found: {job_id}")
    result = status.get("result")
    if status.get("status") != "complete":
        return {"status": status.get("status"), "result": result}
    return {"status": status.get("status"), "result": result}
