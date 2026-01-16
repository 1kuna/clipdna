from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FingerprintResult:
    fingerprint: bytes
    duration: float


class ChromaprintService:
    def generate_fingerprint(self, video_path: str | Path) -> FingerprintResult:
        path = Path(video_path)
        command = ["fpcalc", "-json", str(path)]
        logger.debug("Running fpcalc: %s", " ".join(command))
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)

        fingerprint = data.get("fingerprint", "")
        duration = float(data.get("duration", 0.0))

        if not fingerprint:
            raise ValueError("Empty fingerprint")

        return FingerprintResult(fingerprint=fingerprint.encode("utf-8"), duration=duration)

    def find_clip_in_source(
        self,
        clip_fp: bytes,
        source_fp: bytes,
        source_duration: float | None = None,
        threshold: float = 0.4,
    ) -> Tuple[float, float, float] | None:
        clip_values = _decode_fingerprint(clip_fp)
        source_values = _decode_fingerprint(source_fp)

        if not clip_values or not source_values:
            return None

        clip_arr = np.array(clip_values, dtype=np.float32)
        source_arr = np.array(source_values, dtype=np.float32)

        if len(source_arr) < len(clip_arr):
            return None

        clip_centered = clip_arr - clip_arr.mean()
        source_centered = source_arr - source_arr.mean()

        correlation = np.correlate(source_centered, clip_centered, mode="valid")
        clip_norm = np.linalg.norm(clip_centered)
        if clip_norm == 0:
            return None

        window_energy = np.convolve(
            source_centered**2,
            np.ones(len(clip_centered), dtype=np.float32),
            mode="valid",
        )
        denom = clip_norm * np.sqrt(window_energy) + 1e-9
        scores = correlation / denom

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        confidence = (best_score + 1.0) / 2.0
        if confidence < threshold:
            return None

        if source_duration and len(source_arr) > 0:
            seconds_per_fp = source_duration / len(source_arr)
        else:
            seconds_per_fp = 1.0

        start_time = best_idx * seconds_per_fp
        clip_duration = len(clip_arr) * seconds_per_fp
        end_time = start_time + clip_duration

        return start_time, end_time, confidence


def _decode_fingerprint(fp: bytes) -> list[int]:
    text = fp.decode("utf-8", errors="ignore").strip()
    if not text:
        return []
    return [int(token) for token in text.split(",") if token]


_chromaprint: ChromaprintService | None = None


def get_chromaprint() -> ChromaprintService:
    global _chromaprint
    if _chromaprint is None:
        _chromaprint = ChromaprintService()
    return _chromaprint
