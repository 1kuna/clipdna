from __future__ import annotations

from fastapi import APIRouter

from services.job_queue import get_queue_depth

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check() -> dict[str, object]:
    depth = await get_queue_depth()
    return {"status": "ok", "queue_depth": depth}
