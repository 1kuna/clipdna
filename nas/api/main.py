from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import settings
from routes import health, index, query, results, status
from services.database import init_db
from services.job_queue import init_redis
from shared.utils import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    logger.info("Starting ClipDNA Desktop API")

    db = init_db(settings.database_url)
    await db.ping()
    init_redis(settings.redis_url)

    yield


app = FastAPI(title="ClipDNA Desktop API", lifespan=lifespan)

app.include_router(health.router)
app.include_router(query.router)
app.include_router(status.router)
app.include_router(results.router)
app.include_router(index.router)
