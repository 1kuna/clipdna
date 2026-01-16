from __future__ import annotations

from typing import Final

JOB_QUEUE_NAME: Final[str] = "clipdna:jobs"
JOB_RESULT_PREFIX: Final[str] = "clipdna:job_result"

JOB_TYPE_INDEX_VIDEO: Final[str] = "index_video"
JOB_TYPE_QUERY_CLIP: Final[str] = "query_clip"
JOB_TYPE_REBUILD_INDEX: Final[str] = "rebuild_index"

JOB_STATUS_QUEUED: Final[str] = "queued"
JOB_STATUS_PROCESSING: Final[str] = "processing"
JOB_STATUS_COMPLETE: Final[str] = "complete"
JOB_STATUS_FAILED: Final[str] = "failed"

FAISS_ID_MULTIPLIER: Final[int] = 1_000_000

DEFAULT_FAISS_INDEX_PATH: Final[str] = "/data/index/faiss/video.index"
DEFAULT_VIDEOS_PATH: Final[str] = "/data/videos"
DEFAULT_EMBEDDINGS_PATH: Final[str] = "/data/embeddings"
