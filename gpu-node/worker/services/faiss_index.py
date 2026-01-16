from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    return embeddings / norms


class FAISSIndexBuilder:
    def __init__(
        self,
        embedding_dim: int,
        ef_construction: int = 200,
        ef_search: int = 128,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def build_index(self, embeddings: np.ndarray, ids: np.ndarray):
        import faiss

        embeddings = normalize_embeddings(embeddings.astype("float32"))
        index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        index.add_with_ids(embeddings, ids)
        return index

    def add_to_index(self, index, embeddings: np.ndarray, ids: np.ndarray) -> None:
        embeddings = normalize_embeddings(embeddings.astype("float32"))
        index.add_with_ids(embeddings, ids)

    def save_index(self, index, index_path: Path | str) -> None:
        import faiss

        path = Path(index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
        logger.info("Saved FAISS index to %s", path)


class FAISSIndexSearcher:
    def __init__(self, index_path: Path | str) -> None:
        self.index_path = Path(index_path)
        self._index = None

    def load(self) -> None:
        if self._index is not None:
            return
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

        import faiss

        self._index = faiss.read_index(str(self.index_path))
        logger.info("Loaded FAISS index from %s", self.index_path)

    def search(self, embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        self.load()
        if self._index is None:
            raise RuntimeError("FAISS index not loaded")
        embeddings = normalize_embeddings(embeddings.astype("float32"))
        return self._index.search(embeddings, k)
