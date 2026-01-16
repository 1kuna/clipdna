from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from config import settings
from services.chromaprint import get_chromaprint
from services.database import get_db
from services.faiss_index import FAISSIndexSearcher
from services.frame_extract import extract_all_frames
from services.visil import get_embedder
from shared.constants import FAISS_ID_MULTIPLIER
from shared.job_schemas import QueryClipPayload

logger = logging.getLogger(__name__)


async def query_clip(payload: dict) -> Dict[str, Any]:
    data = QueryClipPayload(**payload)

    clip_path = Path(data.clip_path)
    index_path = Path(settings.faiss_index_path)

    logger.info("Querying clip %s: %s", data.clip_id, clip_path)

    embedder = get_embedder(settings.model_path)
    chromaprint = get_chromaprint()
    searcher = FAISSIndexSearcher(index_path)
    db = get_db()

    try:
        frames, frame_numbers, timestamps = extract_all_frames(
            clip_path, fps=data.fps, use_gpu=True
        )
        logger.info("Extracted %d frames from clip", len(frames))
        if not frames:
            raise ValueError("No frames extracted from clip")

        clip_embeddings = embedder.embed_frames(frames, batch_size=settings.batch_size)

        logger.info("Searching FAISS index (k=%d)", data.top_k_frames)
        distances, faiss_ids = searcher.search(clip_embeddings, k=data.top_k_frames)

        video_scores: dict[int, list[float]] = defaultdict(list)
        video_frame_matches: dict[int, list[dict[str, float]]] = defaultdict(list)

        for clip_frame_idx, (frame_distances, frame_ids) in enumerate(
            zip(distances, faiss_ids)
        ):
            for dist, fid in zip(frame_distances, frame_ids):
                if fid < 0:
                    continue

                video_id = int(fid // FAISS_ID_MULTIPLIER)
                source_frame_idx = int(fid % FAISS_ID_MULTIPLIER)

                video_scores[video_id].append(float(dist))
                video_frame_matches[video_id].append(
                    {
                        "clip_frame": float(clip_frame_idx),
                        "source_frame": float(source_frame_idx),
                        "distance": float(dist),
                    }
                )

        video_rankings: list[dict[str, Any]] = []
        for video_id, scores in video_scores.items():
            avg_score = float(np.mean(scores))
            max_score = float(np.max(scores))
            match_count = len(scores)
            video_rankings.append(
                {
                    "video_id": video_id,
                    "avg_similarity": avg_score,
                    "max_similarity": max_score,
                    "matched_frames": match_count,
                    "frame_matches": video_frame_matches[video_id][:20],
                }
            )

        video_rankings.sort(key=lambda x: x["avg_similarity"], reverse=True)
        candidates = video_rankings[: data.top_k_videos]

        logger.info("Found %d candidate videos", len(candidates))

        logger.info("Generating clip audio fingerprint")
        try:
            clip_audio = chromaprint.generate_fingerprint(clip_path)
            for candidate in candidates:
                source_audio = await db.get_audio_fingerprint(candidate["video_id"])
                if source_audio:
                    result = chromaprint.find_clip_in_source(
                        clip_audio.fingerprint,
                        source_audio["fingerprint"],
                        source_duration=source_audio.get("duration"),
                        threshold=data.audio_threshold,
                    )
                    if result:
                        start_time, end_time, confidence = result
                        candidate["audio_match"] = {
                            "start_time": start_time,
                            "end_time": end_time,
                            "confidence": confidence,
                        }
                    else:
                        candidate["audio_match"] = None
                else:
                    candidate["audio_match"] = None
        except Exception as exc:
            logger.warning("Audio fingerprinting failed: %s", exc)
            for candidate in candidates:
                candidate["audio_match"] = None

        for candidate in candidates:
            video_score = candidate["avg_similarity"]
            audio_score = (
                candidate["audio_match"]["confidence"]
                if candidate.get("audio_match")
                else 0.0
            )
            candidate["final_score"] = 0.6 * video_score + 0.4 * audio_score

        candidates.sort(key=lambda x: x["final_score"], reverse=True)

        for candidate in candidates:
            await _estimate_timestamps(candidate, db, timestamps)

        await db.update_query_clip(clip_id=data.clip_id, status="complete")

        for candidate in candidates[:10]:
            await db.insert_match(
                clip_id=data.clip_id,
                source_video_id=candidate["video_id"],
                confidence_score=candidate["final_score"],
                source_start_time=candidate.get("estimated_start"),
                source_end_time=candidate.get("estimated_end"),
                match_method="visil+chromaprint" if candidate.get("audio_match") else "visil",
            )

        return {
            "status": "success",
            "clip_id": data.clip_id,
            "candidates": candidates,
            "total_candidates": len(video_rankings),
        }
    except Exception as exc:
        logger.exception("Failed to query clip %s", data.clip_id)
        await db.update_query_clip(clip_id=data.clip_id, status="failed", error_message=str(exc))
        raise


async def _estimate_timestamps(
    candidate: Dict[str, Any],
    db,
    clip_timestamps: List[float],
) -> None:
    if not candidate.get("frame_matches"):
        return

    frame_metadata = await db.get_frame_metadata(candidate["video_id"])
    if not frame_metadata:
        return

    source_timestamps = {
        frame["frame_number"]: frame["timestamp"] for frame in frame_metadata
    }

    matches = candidate["frame_matches"]
    source_times: list[float] = []

    for match in matches:
        source_frame = int(match["source_frame"])
        if source_frame in source_timestamps:
            source_times.append(source_timestamps[source_frame])

    if source_times and clip_timestamps:
        candidate["estimated_start"] = min(source_times)
        candidate["estimated_end"] = max(source_times) + (
            clip_timestamps[-1] - clip_timestamps[0]
        )
