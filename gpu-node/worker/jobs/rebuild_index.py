from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from services.faiss_index import FAISSIndexBuilder
from shared.constants import FAISS_ID_MULTIPLIER
from shared.job_schemas import RebuildIndexPayload

logger = logging.getLogger(__name__)


async def rebuild_index(payload: dict) -> dict:
    data = RebuildIndexPayload(**payload)

    embeddings_path = Path(data.embeddings_path)
    index_path = Path(data.index_path)

    if not embeddings_path.exists():
        raise FileNotFoundError(str(embeddings_path))

    embeddings_list: list[np.ndarray] = []
    ids_list: list[np.ndarray] = []
    total_frames = 0

    for emb_file in sorted(embeddings_path.glob("*.npz")):
        video_id = int(emb_file.stem)
        loaded = np.load(emb_file)
        embeddings = loaded["embeddings"].astype(np.float32)
        frame_count = len(embeddings)
        faiss_ids = np.array(
            [video_id * FAISS_ID_MULTIPLIER + i for i in range(frame_count)],
            dtype=np.int64,
        )
        embeddings_list.append(embeddings)
        ids_list.append(faiss_ids)
        total_frames += frame_count

    if not embeddings_list:
        raise ValueError("No embeddings found to rebuild index")

    all_embeddings = np.vstack(embeddings_list)
    all_ids = np.concatenate(ids_list)

    builder = FAISSIndexBuilder(embedding_dim=all_embeddings.shape[1])
    index = builder.build_index(all_embeddings, all_ids)
    builder.save_index(index, index_path)

    logger.info("Rebuilt FAISS index with %d frames", total_frames)
    return {"status": "success", "frames_indexed": total_frames}
