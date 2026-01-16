from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

logger = logging.getLogger(__name__)


class ViSiLEmbedder:
    """Generate frame embeddings using a ViSiL-style backbone."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
        embedding_dim: int = 512,
        use_pretrained: bool | None = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        if use_pretrained is None:
            env_value = os.getenv("VISIL_USE_PRETRAINED", "1")
            use_pretrained = env_value != "0"
        self.use_pretrained = use_pretrained

        logger.info("Initializing ViSiL embedder on %s", self.device)

        self.backbone = self._load_backbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.pca_matrix: np.ndarray | None = None
        self.pca_mean: np.ndarray | None = None
        if model_path:
            pca_path = model_path / "pca.npz"
            if pca_path.exists():
                self._load_pca(pca_path)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_backbone(self) -> nn.Module:
        weights = ResNet50_Weights.IMAGENET1K_V2 if self.use_pretrained else None
        model = resnet50(weights=weights)
        layers = list(model.children())[:-3]
        backbone = nn.Sequential(*layers)
        backbone = backbone.to(self.device)
        backbone.eval()
        return backbone

    def _load_pca(self, pca_path: Path) -> None:
        data = np.load(pca_path)
        self.pca_matrix = data.get("components")
        self.pca_mean = data.get("mean")
        logger.info("Loaded PCA parameters from %s", pca_path)

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame must be HxWx3 array")
        tensor = self.transform(frame).unsqueeze(0)
        return tensor

    def _apply_pca(self, embeddings: np.ndarray) -> np.ndarray:
        if self.pca_matrix is None or self.pca_mean is None:
            return embeddings
        centered = embeddings - self.pca_mean
        projected = centered @ self.pca_matrix.T
        return projected[:, : self.embedding_dim]

    def embed_frames(self, frames: Iterable[np.ndarray], batch_size: int = 32) -> np.ndarray:
        frame_list = list(frames)
        if not frame_list:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        tensors: list[torch.Tensor] = []
        for frame in frame_list:
            tensors.append(self._preprocess(frame))

        all_embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(tensors), batch_size):
                batch = torch.cat(tensors[start : start + batch_size]).to(self.device)
                features = self.backbone(batch)
                pooled = self.pool(features).flatten(1)
                emb = pooled.cpu().numpy()
                all_embeddings.append(emb)

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        embeddings = self._apply_pca(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        return embeddings.astype(np.float32)


_embedder: ViSiLEmbedder | None = None


def get_embedder(model_path: str | Path | None = None) -> ViSiLEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = ViSiLEmbedder(model_path=Path(model_path) if model_path else None)
    return _embedder
