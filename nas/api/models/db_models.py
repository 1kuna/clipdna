from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class SourceVideo(Base):
    __tablename__ = "source_videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filepath: Mapped[str] = mapped_column(String(1024), unique=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    frame_count: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)
    resolution: Mapped[str | None] = mapped_column(String(20))
    indexed_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, server_default=func.now()
    )
    index_status: Mapped[str] = mapped_column(String(20), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text)

    frames: Mapped[list[FrameEmbedding]] = relationship(
        back_populates="source_video", cascade="all, delete-orphan"
    )
    fingerprints: Mapped[list[AudioFingerprint]] = relationship(
        back_populates="source_video", cascade="all, delete-orphan"
    )


class FrameEmbedding(Base):
    __tablename__ = "frame_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_video_id: Mapped[int] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE")
    )
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    faiss_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    source_video: Mapped[SourceVideo] = relationship(back_populates="frames")


class AudioFingerprint(Base):
    __tablename__ = "audio_fingerprints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_video_id: Mapped[int] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE")
    )
    fingerprint: Mapped[bytes] = mapped_column(nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float)

    source_video: Mapped[SourceVideo] = relationship(back_populates="fingerprints")


class QueryClip(Base):
    __tablename__ = "query_clips"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filepath: Mapped[str] = mapped_column(String(1024), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    submitted_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, server_default=func.now()
    )
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text)


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query_clip_id: Mapped[int] = mapped_column(
        ForeignKey("query_clips.id", ondelete="CASCADE")
    )
    source_video_id: Mapped[int] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE")
    )
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    source_start_time: Mapped[float | None] = mapped_column(Float)
    source_end_time: Mapped[float | None] = mapped_column(Float)
    match_method: Mapped[str | None] = mapped_column(String(50))
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, server_default=func.now()
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    created_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column()
    completed_at: Mapped[datetime | None] = mapped_column()
    error_message: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
