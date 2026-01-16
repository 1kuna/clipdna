import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpu-node" / "worker"))

from services.chromaprint import ChromaprintService


def test_chromaprint_alignment():
    service = ChromaprintService()
    clip_fp = b"1,2,3"
    source_fp = b"0,1,2,3,4,5"

    result = service.find_clip_in_source(
        clip_fp,
        source_fp,
        source_duration=6.0,
        threshold=0.1,
    )

    assert result is not None
    start_time, end_time, confidence = result
    assert start_time >= 0
    assert end_time > start_time
    assert 0 <= confidence <= 1
