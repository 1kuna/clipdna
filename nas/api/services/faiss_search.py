from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


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

        logger.info("Loading FAISS index from %s", self.index_path)
        self._index = faiss.read_index(str(self.index_path))

    def search(self, embeddings: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        self.load()
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D array")
        if self._index is None:
            raise RuntimeError("FAISS index not loaded")
        distances, ids = self._index.search(embeddings.astype("float32"), k)
        return distances, ids
