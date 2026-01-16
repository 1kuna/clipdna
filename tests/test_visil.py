import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpu-node" / "worker"))

from services.visil import ViSiLEmbedder


def test_visil_embedder_generates_embeddings():
    embedder = ViSiLEmbedder(device="cpu", use_pretrained=False)
    frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)]
    embeddings = embedder.embed_frames(frames, batch_size=1)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0
