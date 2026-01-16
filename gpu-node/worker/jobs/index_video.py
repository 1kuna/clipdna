from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config import settings
from services.chromaprint import get_chromaprint
from services.database import get_db
from services.faiss_index import FAISSIndexBuilder
from services.frame_extract import FrameExtractor, extract_all_frames
from services.visil import get_embedder
from shared.constants import FAISS_ID_MULTIPLIER
from shared.job_schemas import IndexVideoPayload
from shared.utils import ensure_dir

logger = logging.getLogger(__name__)


async def index_video(payload: dict) -> dict:
    data = IndexVideoPayload(**payload)

    video_path = Path(data.video_path)
    embeddings_path = Path(settings.embeddings_path)
    index_path = Path(settings.faiss_index_path)

    logger.info("Indexing video %s (%s)", data.video_id, video_path)

    embedder = get_embedder(settings.model_path)
    extractor = FrameExtractor(use_gpu=True)
    chromaprint = get_chromaprint()
    db = get_db()

    try:
        video_info = extractor.get_video_info(video_path)
        logger.info("Video info: %s", video_info)

        frames, frame_numbers, timestamps = extract_all_frames(
            video_path, fps=data.fps, use_gpu=True
        )
        logger.info("Extracted %d frames", len(frames))
        if not frames:
            raise ValueError("No frames extracted from source video")

        embeddings = embedder.embed_frames(frames, batch_size=settings.batch_size)
        logger.info("Generated embeddings: %s", embeddings.shape)

        audio_result = chromaprint.generate_fingerprint(video_path)
        logger.info(
            "Audio fingerprint: %d bytes, %.2fs",
            len(audio_result.fingerprint),
            audio_result.duration,
        )

        if data.save_embeddings:
            ensure_dir(embeddings_path)
            emb_file = embeddings_path / f"{data.video_id}.npz"
            np.savez_compressed(
                emb_file,
                embeddings=embeddings,
                frame_numbers=frame_numbers,
                timestamps=timestamps,
            )
            logger.info("Saved embeddings to %s", emb_file)

        faiss_ids = np.array(
            [data.video_id * FAISS_ID_MULTIPLIER + i for i in range(len(embeddings))],
            dtype=np.int64,
        )

        builder = FAISSIndexBuilder(embedding_dim=embeddings.shape[1])

        if index_path.exists():
            import faiss

            index = faiss.read_index(str(index_path))
            builder.add_to_index(index, embeddings, faiss_ids)
        else:
            index = builder.build_index(embeddings, faiss_ids)

        builder.save_index(index, index_path)

        await db.update_source_video(
            video_id=data.video_id,
            status="complete",
            frame_count=len(frames),
            fps=video_info.get("fps"),
            duration_seconds=video_info.get("duration_seconds"),
            resolution=video_info.get("resolution"),
        )

        await db.insert_frame_embeddings(
            video_id=data.video_id,
            frame_numbers=frame_numbers,
            timestamps=timestamps,
            faiss_ids=faiss_ids.tolist(),
        )

        await db.insert_audio_fingerprint(
            video_id=data.video_id,
            fingerprint=audio_result.fingerprint,
            duration=audio_result.duration,
        )

        return {
            "status": "success",
            "video_id": data.video_id,
            "frames_indexed": len(frames),
            "embedding_shape": list(embeddings.shape),
            "audio_duration": audio_result.duration,
        }
    except Exception as exc:
        logger.exception("Failed to index video %s", data.video_id)
        await db.update_source_video(
            video_id=data.video_id,
            status="failed",
            error_message=str(exc),
        )
        raise
