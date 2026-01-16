from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FrameExtractor:
    def __init__(self, use_gpu: bool = True) -> None:
        self.use_gpu = use_gpu

    def _get_context(self):
        import decord

        if self.use_gpu:
            try:
                return decord.gpu(0)
            except Exception:
                logger.warning("GPU decode unavailable, falling back to CPU")
        return decord.cpu(0)

    def get_video_info(self, video_path: str | Path) -> dict[str, float | int | str]:
        import decord

        path = Path(video_path)
        vr = decord.VideoReader(str(path), ctx=self._get_context())
        fps = float(vr.get_avg_fps())
        frame_count = len(vr)
        duration_seconds = frame_count / fps if fps > 0 else 0
        width, height = vr[0].shape[1], vr[0].shape[0]
        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration_seconds,
            "resolution": f"{width}x{height}",
        }


def extract_all_frames(
    video_path: str | Path,
    fps: float = 1.0,
    use_gpu: bool = True,
) -> Tuple[List[np.ndarray], List[int], List[float]]:
    import decord

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    ctx = decord.gpu(0) if use_gpu else decord.cpu(0)
    vr = decord.VideoReader(str(path), ctx=ctx)

    source_fps = float(vr.get_avg_fps())
    if source_fps <= 0:
        raise ValueError("Invalid source FPS")

    step = max(int(round(source_fps / fps)), 1)
    indices = list(range(0, len(vr), step))
    if not indices:
        return [], [], []

    frames = vr.get_batch(indices).asnumpy()
    frame_numbers = indices
    timestamps = [idx / source_fps for idx in indices]

    return list(frames), frame_numbers, timestamps
