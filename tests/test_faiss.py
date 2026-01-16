import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpu-node" / "worker"))

from services.faiss_index import FAISSIndexBuilder, FAISSIndexSearcher


def test_faiss_build_and_search(tmp_path):
    embeddings = np.random.rand(10, 8).astype(np.float32)
    ids = np.arange(10, dtype=np.int64)

    builder = FAISSIndexBuilder(embedding_dim=8)
    index = builder.build_index(embeddings, ids)

    index_path = tmp_path / "index.faiss"
    builder.save_index(index, index_path)

    searcher = FAISSIndexSearcher(index_path)
    distances, found_ids = searcher.search(embeddings[:2], k=3)

    assert distances.shape == (2, 3)
    assert found_ids.shape == (2, 3)
