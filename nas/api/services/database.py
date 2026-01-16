from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Iterable

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from models.db_models import AudioFingerprint, FrameEmbedding, Job, Match, QueryClip, SourceVideo

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, database_url: str) -> None:
        async_url = _to_async_url(database_url)
        self._engine: AsyncEngine = create_async_engine(async_url, pool_pre_ping=True)
        self._sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def ping(self) -> None:
        async with self._engine.connect() as conn:
            await conn.execute(select(1))
        logger.info("Database connection OK")

    async def create_source_video(self, filepath: str, filename: str) -> int:
        async with self._sessionmaker() as session:
            video = SourceVideo(filepath=filepath, filename=filename)
            session.add(video)
            await session.commit()
            await session.refresh(video)
            return video.id

    async def update_source_video(
        self,
        video_id: int,
        status: str | None = None,
        frame_count: int | None = None,
        fps: float | None = None,
        duration_seconds: float | None = None,
        resolution: str | None = None,
        error_message: str | None = None,
    ) -> None:
        async with self._sessionmaker() as session:
            values: dict[str, Any] = {}
            if status is not None:
                values["index_status"] = status
            if frame_count is not None:
                values["frame_count"] = frame_count
            if fps is not None:
                values["fps"] = fps
            if duration_seconds is not None:
                values["duration_seconds"] = duration_seconds
            if resolution is not None:
                values["resolution"] = resolution
            if error_message is not None:
                values["error_message"] = error_message
            if values:
                await session.execute(
                    update(SourceVideo).where(SourceVideo.id == video_id).values(**values)
                )
                await session.commit()

    async def insert_frame_embeddings(
        self,
        video_id: int,
        frame_numbers: Iterable[int],
        timestamps: Iterable[float],
        faiss_ids: Iterable[int],
    ) -> None:
        records = [
            {
                "source_video_id": video_id,
                "frame_number": frame_number,
                "timestamp_seconds": timestamp,
                "faiss_id": faiss_id,
            }
            for frame_number, timestamp, faiss_id in zip(frame_numbers, timestamps, faiss_ids)
        ]
        if not records:
            return
        async with self._sessionmaker() as session:
            session.add_all([FrameEmbedding(**record) for record in records])
            await session.commit()

    async def insert_audio_fingerprint(
        self,
        video_id: int,
        fingerprint: bytes,
        duration: float | None,
    ) -> None:
        async with self._sessionmaker() as session:
            session.add(
                AudioFingerprint(
                    source_video_id=video_id,
                    fingerprint=fingerprint,
                    duration_seconds=duration,
                )
            )
            await session.commit()

    async def get_audio_fingerprint(self, video_id: int) -> dict[str, Any] | None:
        async with self._sessionmaker() as session:
            result = await session.execute(
                select(AudioFingerprint).where(AudioFingerprint.source_video_id == video_id)
            )
            fp = result.scalar_one_or_none()
            if not fp:
                return None
            return {"fingerprint": fp.fingerprint, "duration": fp.duration_seconds}

    async def get_source_video(self, video_id: int) -> dict[str, Any] | None:
        async with self._sessionmaker() as session:
            result = await session.execute(select(SourceVideo).where(SourceVideo.id == video_id))
            video = result.scalar_one_or_none()
            if not video:
                return None
            return {
                "id": video.id,
                "filepath": video.filepath,
                "filename": video.filename,
                "duration_seconds": video.duration_seconds,
                "frame_count": video.frame_count,
                "fps": video.fps,
                "resolution": video.resolution,
                "index_status": video.index_status,
            }

    async def get_source_by_path(self, filepath: str) -> dict[str, Any] | None:
        async with self._sessionmaker() as session:
            result = await session.execute(
                select(SourceVideo).where(SourceVideo.filepath == filepath)
            )
            video = result.scalar_one_or_none()
            if not video:
                return None
            return {
                "id": video.id,
                "filepath": video.filepath,
                "filename": video.filename,
                "duration_seconds": video.duration_seconds,
                "frame_count": video.frame_count,
                "fps": video.fps,
                "resolution": video.resolution,
                "index_status": video.index_status,
            }

    async def get_frame_metadata(self, video_id: int) -> list[dict[str, Any]]:
        async with self._sessionmaker() as session:
            result = await session.execute(
                select(FrameEmbedding).where(FrameEmbedding.source_video_id == video_id)
            )
            rows = result.scalars().all()
            return [
                {
                    "frame_number": row.frame_number,
                    "timestamp": row.timestamp_seconds,
                    "faiss_id": row.faiss_id,
                }
                for row in rows
            ]

    async def create_query_clip(self, filepath: str, filename: str) -> int:
        async with self._sessionmaker() as session:
            clip = QueryClip(filepath=filepath, filename=filename)
            session.add(clip)
            await session.commit()
            await session.refresh(clip)
            return clip.id

    async def update_query_clip(
        self,
        clip_id: int,
        status: str | None = None,
        error_message: str | None = None,
    ) -> None:
        async with self._sessionmaker() as session:
            values: dict[str, Any] = {}
            if status is not None:
                values["status"] = status
            if error_message is not None:
                values["error_message"] = error_message
            if values:
                await session.execute(
                    update(QueryClip).where(QueryClip.id == clip_id).values(**values)
                )
                await session.commit()

    async def insert_match(
        self,
        clip_id: int,
        source_video_id: int,
        confidence_score: float,
        source_start_time: float | None,
        source_end_time: float | None,
        match_method: str,
    ) -> None:
        async with self._sessionmaker() as session:
            session.add(
                Match(
                    query_clip_id=clip_id,
                    source_video_id=source_video_id,
                    confidence_score=confidence_score,
                    source_start_time=source_start_time,
                    source_end_time=source_end_time,
                    match_method=match_method,
                )
            )
            await session.commit()

    async def get_matches_for_clip(self, clip_id: int) -> list[dict[str, Any]]:
        async with self._sessionmaker() as session:
            result = await session.execute(select(Match).where(Match.query_clip_id == clip_id))
            matches = result.scalars().all()
            return [
                {
                    "source_video_id": match.source_video_id,
                    "confidence_score": match.confidence_score,
                    "source_start_time": match.source_start_time,
                    "source_end_time": match.source_end_time,
                    "match_method": match.match_method,
                }
                for match in matches
            ]

    async def create_job(self, job_id: str, job_type: str, payload: dict[str, Any]) -> None:
        async with self._sessionmaker() as session:
            session.add(Job(id=job_id, job_type=job_type, payload=payload))
            await session.commit()

    async def update_job(
        self,
        job_id: str,
        status: str,
        error_message: str | None = None,
        result: dict[str, Any] | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        async with self._sessionmaker() as session:
            values: dict[str, Any] = {"status": status}
            if error_message is not None:
                values["error_message"] = error_message
            if result is not None:
                values["result"] = result
            if started_at is not None:
                values["started_at"] = started_at
            if completed_at is not None:
                values["completed_at"] = completed_at
            await session.execute(update(Job).where(Job.id == job_id).values(**values))
            await session.commit()

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        async with self._sessionmaker() as session:
            result = await session.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                return None
            return {
                "id": job.id,
                "job_type": job.job_type,
                "status": job.status,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "error_message": job.error_message,
                "result": job.result,
            }


_db: Database | None = None


def _to_async_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    return database_url
def init_db(database_url: str) -> Database:
    global _db
    if _db is None:
        _db = Database(database_url)
    return _db


def get_db() -> Database:
    if _db is None:
        raise RuntimeError("Database is not initialized")
    return _db
