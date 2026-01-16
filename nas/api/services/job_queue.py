from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as redis

from shared.constants import JOB_QUEUE_NAME, JOB_STATUS_QUEUED
from shared.job_schemas import JobEnvelope
from services.database import get_db

logger = logging.getLogger(__name__)


_redis: redis.Redis | None = None


def init_redis(redis_url: str) -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(redis_url, decode_responses=True)
    return _redis


def get_redis() -> redis.Redis:
    if _redis is None:
        raise RuntimeError("Redis is not initialized")
    return _redis


async def enqueue_job(job_id: str, job_type: str, payload: dict[str, Any]) -> None:
    db = get_db()
    await db.create_job(job_id=job_id, job_type=job_type, payload=payload)

    envelope = JobEnvelope(job_id=job_id, job_type=job_type, payload=payload)
    redis_client = get_redis()
    await redis_client.rpush(JOB_QUEUE_NAME, json.dumps(envelope.model_dump()))

    logger.info("Enqueued job %s (%s)", job_id, job_type)


async def get_queue_depth() -> int:
    redis_client = get_redis()
    return await redis_client.llen(JOB_QUEUE_NAME)


async def get_job_status(job_id: str) -> dict[str, Any] | None:
    db = get_db()
    job = await db.get_job(job_id)
    if not job:
        return None
    return {
        "job_id": job["id"],
        "status": job["status"],
        "job_type": job["job_type"],
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "error_message": job.get("error_message"),
        "result": job.get("result"),
    }
