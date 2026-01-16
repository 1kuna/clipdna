from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable

import redis.asyncio as redis

from config import settings
from jobs.index_video import index_video
from jobs.query_clip import query_clip
from jobs.rebuild_index import rebuild_index
from services.database import init_db
from shared.constants import (
    JOB_QUEUE_NAME,
    JOB_STATUS_COMPLETE,
    JOB_STATUS_FAILED,
    JOB_STATUS_PROCESSING,
)
from shared.job_schemas import JobEnvelope
from shared.utils import configure_logging

logger = logging.getLogger(__name__)


JobHandler = Callable[[dict], Awaitable[dict]]


async def handle_job(envelope: JobEnvelope, db) -> None:
    handlers: dict[str, JobHandler] = {
        "index_video": index_video,
        "query_clip": query_clip,
        "rebuild_index": rebuild_index,
    }

    handler = handlers.get(envelope.job_type)
    if handler is None:
        logger.error("Unknown job type: %s", envelope.job_type)
        await db.update_job(
            job_id=envelope.job_id,
            status=JOB_STATUS_FAILED,
            error_message=f"Unknown job type: {envelope.job_type}",
            completed_at=datetime.now(timezone.utc),
        )
        return

    await db.update_job(
        job_id=envelope.job_id,
        status=JOB_STATUS_PROCESSING,
        started_at=datetime.now(timezone.utc),
    )

    try:
        result = await handler(envelope.payload)
        await db.update_job(
            job_id=envelope.job_id,
            status=JOB_STATUS_COMPLETE,
            result=result,
            completed_at=datetime.now(timezone.utc),
        )
        logger.info("Job %s complete", envelope.job_id)
    except Exception as exc:
        logger.exception("Job %s failed", envelope.job_id)
        await db.update_job(
            job_id=envelope.job_id,
            status=JOB_STATUS_FAILED,
            error_message=str(exc),
            completed_at=datetime.now(timezone.utc),
        )


async def worker_loop(worker_id: int) -> None:
    configure_logging(settings.log_level)
    logger.info("Starting ClipDNA Desktop worker %s", worker_id)

    db = init_db(settings.database_url)
    await db.ping()

    redis_client = redis.from_url(settings.redis_url, decode_responses=True)

    while True:
        result = await redis_client.blpop(JOB_QUEUE_NAME, timeout=5)
        if not result:
            continue

        _, raw = result
        payload = json.loads(raw)
        envelope = JobEnvelope(**payload)
        logger.info(
            "Worker %s dequeued job %s (%s)",
            worker_id,
            envelope.job_id,
            envelope.job_type,
        )
        await handle_job(envelope, db)


async def main() -> None:
    tasks = [asyncio.create_task(worker_loop(i)) for i in range(settings.num_workers)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
