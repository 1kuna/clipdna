# Video Clip Matcher

A video fingerprinting system for matching short clips to their source videos using ViSiL embeddings, FAISS indexing, and Chromaprint audio fingerprinting.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NAS (Docker)                                   │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐  │
│  │ PostgreSQL  │  │   Redis     │  │   MinIO     │  │   API Service     │  │
│  │ + pgvector  │  │   (queue)   │  │ (optional)  │  │   (FastAPI)       │  │
│  │             │  │             │  │             │  │                   │  │
│  │ - metadata  │  │ - job queue │  │ - index     │  │ - /query          │  │
│  │ - matches   │  │ - results   │  │   backups   │  │ - /status         │  │
│  │ - audio fps │  │             │  │             │  │ - /results        │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Shared Storage                               │   │
│  │  /data/videos/sources/     - Source videos (thousands)               │   │
│  │  /data/videos/clips/       - Clips to match (hundreds)               │   │
│  │  /data/index/faiss/        - FAISS index files (mmap'd)              │   │
│  │  /data/index/chromaprint/  - Audio fingerprint database              │   │
│  │  /data/embeddings/         - Raw embeddings (optional backup)        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ NFS mount: /mnt/nas
                                      │ Redis connection
                                      │ PostgreSQL connection
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            GPU Node (Docker)                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Worker Service                                 │   │
│  │                                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │   │
│  │  │   ViSiL     │  │   Decord    │  │ Chromaprint │                  │   │
│  │  │  (PyTorch)  │  │ (GPU decode)│  │  (fpcalc)   │                  │   │
│  │  │             │  │             │  │             │                  │   │
│  │  │ - ResNet50  │  │ - frame     │  │ - audio     │                  │   │
│  │  │ - attention │  │   extract   │  │   fingerpr. │                  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │   │
│  │                                                                      │   │
│  │  Jobs:                                                               │   │
│  │  - index_video: Generate embeddings for source video                 │   │
│  │  - query_clip: Generate embeddings for clip, search FAISS            │   │
│  │  - rebuild_index: Rebuild FAISS from stored embeddings               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
video-clip-matcher/
│
├── README.md
├── PROJECT_OUTLINE.md          # This file
│
├── nas/                        # Runs on NAS
│   ├── docker-compose.yml
│   ├── .env.example
│   │
│   ├── api/                    # FastAPI service
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── main.py             # FastAPI app entrypoint
│   │   ├── config.py           # Settings via pydantic
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── query.py        # POST /query - submit clip for matching
│   │   │   ├── status.py       # GET /status/{job_id} - check job status
│   │   │   ├── results.py      # GET /results/{job_id} - get match results
│   │   │   ├── index.py        # POST /index - trigger indexing jobs
│   │   │   └── health.py       # GET /health
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── job_queue.py    # Redis job queue interface
│   │   │   ├── database.py     # PostgreSQL operations
│   │   │   └── faiss_search.py # FAISS index loading and search
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── schemas.py      # Pydantic request/response models
│   │   │   └── db_models.py    # SQLAlchemy ORM models
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── ffmpeg.py       # FFmpeg signature verification
│   │
│   ├── db/
│   │   ├── init.sql            # PostgreSQL schema initialization
│   │   └── migrations/         # Alembic migrations (optional)
│   │
│   └── nginx/                  # Optional reverse proxy
│       └── nginx.conf
│
├── gpu-node/                   # Runs on GPU Node (DGX Spark, PC, etc.)
│   ├── docker-compose.yml
│   ├── .env.example
│   │
│   ├── worker/                 # GPU worker service
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── main.py             # Worker entrypoint (connects to Redis)
│   │   ├── config.py           # Settings
│   │   ├── jobs/
│   │   │   ├── __init__.py
│   │   │   ├── index_video.py  # Index a source video
│   │   │   ├── query_clip.py   # Query a clip against index
│   │   │   └── rebuild_index.py# Rebuild FAISS index
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── visil.py        # ViSiL model wrapper
│   │   │   ├── frame_extract.py# Decord-based frame extraction
│   │   │   ├── chromaprint.py  # Audio fingerprinting
│   │   │   ├── faiss_index.py  # FAISS index building
│   │   │   └── database.py     # PostgreSQL client
│   │   └── models/
│   │       └── visil/          # ViSiL model weights (downloaded on first run)
│   │
│   └── scripts/
│       ├── download_weights.py # Download ViSiL pretrained weights
│       └── test_gpu.py         # Verify GPU/CUDA setup
│
├── shared/                     # Shared code (mounted in both services)
│   ├── __init__.py
│   ├── constants.py            # Shared constants
│   ├── job_schemas.py          # Job payload schemas
│   └── utils.py                # Common utilities
│
├── scripts/                    # Utility scripts (run manually)
│   ├── index_all_sources.py    # Batch index all source videos
│   ├── query_all_clips.py      # Batch query all clips
│   ├── export_results.py       # Export matches to CSV/JSON
│   ├── verify_matches.py       # Manual verification helper
│   └── benchmark.py            # Performance testing
│
├── tests/
│   ├── conftest.py
│   ├── test_visil.py
│   ├── test_faiss.py
│   ├── test_chromaprint.py
│   └── test_api.py
│
└── docs/
    ├── setup.md                # Installation guide
    ├── usage.md                # How to use
    ├── architecture.md         # Detailed architecture docs
    └── tuning.md               # Threshold tuning guide
```

---

## Component Specifications

### PostgreSQL Schema

```sql
-- nas/db/init.sql

-- Source videos that have been indexed
CREATE TABLE source_videos (
    id SERIAL PRIMARY KEY,
    filepath VARCHAR(1024) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    duration_seconds FLOAT,
    frame_count INTEGER,
    fps FLOAT,
    resolution VARCHAR(20),          -- e.g., "1920x1080"
    indexed_at TIMESTAMP DEFAULT NOW(),
    index_status VARCHAR(20) DEFAULT 'pending',  -- pending, processing, complete, failed
    error_message TEXT
);

-- Frame-level embeddings metadata (actual vectors in FAISS)
CREATE TABLE frame_embeddings (
    id SERIAL PRIMARY KEY,
    source_video_id INTEGER REFERENCES source_videos(id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    timestamp_seconds FLOAT NOT NULL,
    faiss_id BIGINT NOT NULL,        -- ID in FAISS index
    UNIQUE(source_video_id, frame_number)
);
CREATE INDEX idx_frame_embeddings_faiss ON frame_embeddings(faiss_id);

-- Audio fingerprints
CREATE TABLE audio_fingerprints (
    id SERIAL PRIMARY KEY,
    source_video_id INTEGER REFERENCES source_videos(id) ON DELETE CASCADE,
    fingerprint BYTEA NOT NULL,       -- Chromaprint raw fingerprint
    duration_seconds FLOAT
);

-- Query clips
CREATE TABLE query_clips (
    id SERIAL PRIMARY KEY,
    filepath VARCHAR(1024) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    duration_seconds FLOAT,
    submitted_at TIMESTAMP DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'pending',  -- pending, processing, complete, failed
    error_message TEXT
);

-- Match results
CREATE TABLE matches (
    id SERIAL PRIMARY KEY,
    query_clip_id INTEGER REFERENCES query_clips(id) ON DELETE CASCADE,
    source_video_id INTEGER REFERENCES source_videos(id) ON DELETE CASCADE,
    confidence_score FLOAT NOT NULL,       -- 0.0 - 1.0
    source_start_time FLOAT,               -- Where in source the clip starts
    source_end_time FLOAT,                 -- Where in source the clip ends
    match_method VARCHAR(50),              -- 'visil', 'chromaprint', 'both'
    verified BOOLEAN DEFAULT FALSE,        -- Manual verification flag
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_matches_clip ON matches(query_clip_id);
CREATE INDEX idx_matches_confidence ON matches(confidence_score DESC);

-- Job queue tracking (supplements Redis)
CREATE TABLE jobs (
    id VARCHAR(36) PRIMARY KEY,            -- UUID
    job_type VARCHAR(50) NOT NULL,         -- index_video, query_clip, rebuild_index
    payload JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'queued',   -- queued, processing, complete, failed
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    result JSONB
);
```

### Docker Compose - NAS

```yaml
# nas/docker-compose.yml

version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: vcm-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-vcm}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?required}
      POSTGRES_DB: ${POSTGRES_DB:-videoclipdb}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-vcm}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: vcm-redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    container_name: vcm-api
    restart: unless-stopped
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER:-vcm}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-videoclipdb}
      REDIS_URL: redis://redis:6379/0
      FAISS_INDEX_PATH: /data/index/faiss/video.index
      VIDEOS_PATH: /data/videos
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
    volumes:
      - ${DATA_PATH:?required}:/data
      - ../shared:/app/shared:ro
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

volumes:
  postgres_data:
  redis_data:
```

### Docker Compose - GPU Node

```yaml
# gpu-node/docker-compose.yml

version: '3.8'

services:
  worker:
    build:
      context: ./worker
      dockerfile: Dockerfile
    container_name: vcm-worker
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER:-vcm}:${POSTGRES_PASSWORD}@${NAS_HOST:?required}:5432/${POSTGRES_DB:-videoclipdb}
      REDIS_URL: redis://${NAS_HOST}:6379/0
      FAISS_INDEX_PATH: /data/index/faiss/video.index
      VIDEOS_PATH: /data/videos
      EMBEDDINGS_PATH: /data/embeddings
      MODEL_PATH: /app/models/visil
      FRAME_RATE: ${FRAME_RATE:-1}           # Frames per second to extract
      BATCH_SIZE: ${BATCH_SIZE:-32}          # Frames per batch for ViSiL
      NUM_WORKERS: ${NUM_WORKERS:-1}         # Parallel jobs (usually 1 for GPU)
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
    volumes:
      - ${NAS_MOUNT_PATH:?required}:/data    # NFS mount to NAS storage
      - ./worker/models:/app/models          # Model weights (persistent)
      - ../shared:/app/shared:ro
    # For local development without NFS:
    # volumes:
    #   - /path/to/local/data:/data
```

### Dockerfiles

```dockerfile
# nas/api/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# FFmpeg for signature verification
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# gpu-node/worker/Dockerfile

FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    ffmpeg \
    libchromaprint-tools \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Download ViSiL weights on build (optional, can also do at runtime)
# RUN python3 scripts/download_weights.py

COPY . .

CMD ["python3", "main.py"]
```

### Requirements Files

```text
# nas/api/requirements.txt

fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
sqlalchemy>=2.0.0
asyncpg>=0.29.0
psycopg2-binary>=2.9.9
redis>=5.0.0
faiss-cpu>=1.7.4
numpy>=1.26.0
python-multipart>=0.0.6
httpx>=0.26.0
```

```text
# gpu-node/worker/requirements.txt

torch>=2.1.0
torchvision>=0.16.0
numpy>=1.26.0
faiss-gpu>=1.7.4
decord>=0.6.0
opencv-python-headless>=4.9.0
Pillow>=10.2.0
redis>=5.0.0
psycopg2-binary>=2.9.9
sqlalchemy>=2.0.0
pyacoustid>=1.3.0
pydantic>=2.5.0
tqdm>=4.66.0
```

---

## Core Service Implementations

### ViSiL Service (GPU Node)

```python
# gpu-node/worker/services/visil.py

"""
ViSiL (Video Similarity Learning) wrapper for frame embedding generation.

Paper: https://arxiv.org/abs/1908.07410
Repo: https://github.com/MKLab-ITI/visil
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class ViSiLEmbedder:
    """
    Generates frame embeddings using ViSiL's approach:
    - ResNet50 backbone (layer3 features)
    - Region-level features with attention
    - Whitening for better similarity computation
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
        embedding_dim: int = 512,  # After PCA reduction
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        
        logger.info(f"Initializing ViSiL embedder on {self.device}")
        
        # Load backbone
        self.backbone = self._load_backbone()
        
        # Load whitening/PCA parameters if available
        self.pca_matrix = None
        self.pca_mean = None
        if model_path and (model_path / "pca.npz").exists():
            self._load_pca(model_path / "pca.npz")
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def _load_backbone(self) -> nn.Module:
        """Load ResNet50 and extract intermediate features."""
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # We want features from layer3 (before final pooling)
        # Output: [batch, 1024, 14, 14] for 224x224 input
        layers = list(model.children())[:-3]  # Remove layer4, avgpool, fc
        backbone = nn.Sequential(*layers)
        backbone = backbone.to(self.device)
        backbone.eval()
        
        return backbone
    
    def _load_pca(self, pca_path: Path):
        """Load PCA/whitening parameters."""
        data = np.load(pca_path)
        self.pca_matrix = torch.tensor(data["matrix"], device=self.device, dtype=torch.float32)
        self.pca_mean = torch.tensor(data["mean"], device=self.device, dtype=torch.float32)
        logger.info(f"Loaded PCA matrix: {self.pca_matrix.shape}")
    
    @torch.no_grad()
    def embed_frames(self, frames: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of frames.
        
        Args:
            frames: List of BGR/RGB numpy arrays (H, W, 3)
            batch_size: Batch size for inference
            
        Returns:
            embeddings: numpy array of shape (num_frames, embedding_dim)
        """
        if not frames:
            return np.array([])
        
        embeddings = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # Preprocess
            batch_tensors = torch.stack([
                self.transform(frame) for frame in batch_frames
            ]).to(self.device)
            
            # Extract features
            features = self.backbone(batch_tensors)  # [B, 1024, 14, 14]
            
            # Global average pooling over spatial dimensions
            features = features.mean(dim=[2, 3])  # [B, 1024]
            
            # Apply PCA/whitening if available
            if self.pca_matrix is not None:
                features = features - self.pca_mean
                features = features @ self.pca_matrix.T
            
            # L2 normalize
            features = nn.functional.normalize(features, p=2, dim=1)
            
            embeddings.append(features.cpu().numpy())
        
        return np.vstack(embeddings).astype(np.float32)
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2))


# Singleton instance
_embedder: Optional[ViSiLEmbedder] = None

def get_embedder(model_path: Optional[Path] = None) -> ViSiLEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = ViSiLEmbedder(model_path=model_path)
    return _embedder
```

### Frame Extraction Service (GPU Node)

```python
# gpu-node/worker/services/frame_extract.py

"""
GPU-accelerated frame extraction using Decord.
Falls back to OpenCV if Decord unavailable.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator
import logging

logger = logging.getLogger(__name__)

try:
    from decord import VideoReader, cpu, gpu
    from decord import bridge
    bridge.set_bridge("native")
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    import cv2
    logger.warning("Decord not available, falling back to OpenCV")


class FrameExtractor:
    """Extract frames from video at specified rate."""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and DECORD_AVAILABLE
        if self.use_gpu:
            try:
                # Test GPU context
                self.ctx = gpu(0)
                logger.info("Using Decord GPU acceleration")
            except:
                self.ctx = cpu(0)
                self.use_gpu = False
                logger.info("GPU unavailable, using Decord CPU")
        elif DECORD_AVAILABLE:
            self.ctx = cpu(0)
    
    def get_video_info(self, video_path: Path) -> dict:
        """Get video metadata."""
        if DECORD_AVAILABLE:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            return {
                "frame_count": len(vr),
                "fps": float(vr.get_avg_fps()),
                "duration_seconds": len(vr) / vr.get_avg_fps(),
                "resolution": f"{vr[0].shape[1]}x{vr[0].shape[0]}",  # WxH
            }
        else:
            cap = cv2.VideoCapture(str(video_path))
            info = {
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
                "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
            }
            cap.release()
            return info
    
    def extract_frames(
        self,
        video_path: Path,
        fps: float = 1.0,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Extract frames at specified FPS.
        
        Yields:
            (frame_number, timestamp_seconds, frame_array)
        """
        video_path = Path(video_path)
        
        if DECORD_AVAILABLE:
            yield from self._extract_decord(video_path, fps, max_frames)
        else:
            yield from self._extract_opencv(video_path, fps, max_frames)
    
    def _extract_decord(
        self,
        video_path: Path,
        fps: float,
        max_frames: Optional[int],
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Decord-based extraction."""
        vr = VideoReader(str(video_path), ctx=self.ctx)
        video_fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        # Calculate frame indices to extract
        frame_interval = max(1, int(video_fps / fps))
        frame_indices = list(range(0, total_frames, frame_interval))
        
        if max_frames:
            frame_indices = frame_indices[:max_frames]
        
        # Batch fetch for efficiency
        batch_size = 32
        for i in range(0, len(frame_indices), batch_size):
            batch_indices = frame_indices[i:i + batch_size]
            frames = vr.get_batch(batch_indices).asnumpy()
            
            for j, frame_idx in enumerate(batch_indices):
                timestamp = frame_idx / video_fps
                yield (frame_idx, timestamp, frames[j])
    
    def _extract_opencv(
        self,
        video_path: Path,
        fps: float,
        max_frames: Optional[int],
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """OpenCV fallback extraction."""
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps))
        
        frame_idx = 0
        extracted = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / video_fps
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield (frame_idx, timestamp, frame_rgb)
                
                extracted += 1
                if max_frames and extracted >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()


def extract_all_frames(
    video_path: Path,
    fps: float = 1.0,
    use_gpu: bool = True,
) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """
    Convenience function to extract all frames at once.
    
    Returns:
        (frames, frame_numbers, timestamps)
    """
    extractor = FrameExtractor(use_gpu=use_gpu)
    
    frames = []
    frame_numbers = []
    timestamps = []
    
    for frame_num, timestamp, frame in extractor.extract_frames(video_path, fps=fps):
        frames.append(frame)
        frame_numbers.append(frame_num)
        timestamps.append(timestamp)
    
    return frames, frame_numbers, timestamps
```

### FAISS Index Service (GPU Node + NAS)

```python
# gpu-node/worker/services/faiss_index.py

"""
FAISS index management for video frame embeddings.
Supports building on GPU, serving from CPU (mmap'd on NAS).
"""

import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import threading

logger = logging.getLogger(__name__)


class FAISSIndexBuilder:
    """Build FAISS index on GPU node."""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        index_type: str = "HNSW",  # or "IVF_PQ", "Flat"
        use_gpu: bool = True,
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        if self.use_gpu:
            logger.info(f"FAISS using GPU (found {faiss.get_num_gpus()} GPUs)")
        else:
            logger.info("FAISS using CPU")
    
    def build_index(
        self,
        embeddings: np.ndarray,
        ids: Optional[np.ndarray] = None,
    ) -> faiss.Index:
        """
        Build index from embeddings.
        
        Args:
            embeddings: (N, embedding_dim) float32 array
            ids: Optional (N,) int64 array of IDs
            
        Returns:
            Trained FAISS index
        """
        n_vectors = embeddings.shape[0]
        logger.info(f"Building {self.index_type} index for {n_vectors} vectors")
        
        if self.index_type == "HNSW":
            # HNSW: Best recall/speed tradeoff, no training needed
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # M=32 connections
            index.hnsw.efConstruction = 200  # Higher = better quality, slower build
            index.hnsw.efSearch = 128  # Can adjust at search time
            
        elif self.index_type == "IVF_PQ":
            # IVF with Product Quantization: More memory efficient
            n_list = min(4096, n_vectors // 10)  # Number of clusters
            m = 64  # Number of subquantizers
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, n_list, m, 8)
            
            # IVF requires training
            if self.use_gpu:
                gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
                gpu_index.train(embeddings)
                index = faiss.index_gpu_to_cpu(gpu_index)
            else:
                index.train(embeddings)
                
        else:  # Flat
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine sim
        
        # Wrap with IDMap if custom IDs provided
        if ids is not None:
            index_with_ids = faiss.IndexIDMap(index)
            index_with_ids.add_with_ids(embeddings, ids)
            return index_with_ids
        else:
            index.add(embeddings)
            return index
    
    def save_index(self, index: faiss.Index, path: Path):
        """Save index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
        logger.info(f"Saved index to {path}")
    
    def add_to_index(
        self,
        index: faiss.Index,
        embeddings: np.ndarray,
        ids: np.ndarray,
    ) -> faiss.Index:
        """Add new embeddings to existing index."""
        if isinstance(index, faiss.IndexIDMap):
            index.add_with_ids(embeddings, ids)
        else:
            index.add(embeddings)
        return index


class FAISSIndexSearcher:
    """Search FAISS index (runs on NAS, CPU-only, memory-mapped)."""
    
    def __init__(self, index_path: Optional[Path] = None):
        self.index: Optional[faiss.Index] = None
        self.index_path = index_path
        self._lock = threading.Lock()
        
        if index_path and index_path.exists():
            self.load_index(index_path)
    
    def load_index(self, path: Path, mmap: bool = True):
        """Load index from disk, optionally memory-mapped."""
        with self._lock:
            if mmap:
                # Memory-map for lower RAM usage on NAS
                self.index = faiss.read_index(str(path), faiss.IO_FLAG_MMAP)
            else:
                self.index = faiss.read_index(str(path))
            self.index_path = path
            logger.info(f"Loaded index from {path} (ntotal={self.index.ntotal})")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 100,
        ef_search: int = 128,  # HNSW search parameter
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.
        
        Args:
            query_embeddings: (N, dim) float32 array
            k: Number of neighbors to return
            ef_search: HNSW search effort (higher = better recall, slower)
            
        Returns:
            (distances, ids): Both shape (N, k)
        """
        if self.index is None:
            raise RuntimeError("No index loaded")
        
        # Set HNSW search parameter if applicable
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = ef_search
        
        with self._lock:
            distances, ids = self.index.search(query_embeddings, k)
        
        return distances, ids
    
    def search_single(
        self,
        query_embedding: np.ndarray,
        k: int = 100,
    ) -> List[Tuple[int, float]]:
        """Search for a single query, return list of (id, distance) tuples."""
        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, ids = self.search(query, k=k)
        
        return [(int(ids[0, i]), float(distances[0, i])) for i in range(k) if ids[0, i] >= 0]


# Global searcher instance for API
_searcher: Optional[FAISSIndexSearcher] = None

def get_searcher(index_path: Optional[Path] = None) -> FAISSIndexSearcher:
    global _searcher
    if _searcher is None:
        _searcher = FAISSIndexSearcher(index_path)
    return _searcher
```

### Chromaprint Service (GPU Node)

```python
# gpu-node/worker/services/chromaprint.py

"""
Audio fingerprinting using Chromaprint/AcoustID.
Provides segment-level matching capability.
"""

import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ChromaprintService:
    """Generate and compare audio fingerprints."""
    
    def __init__(self, fpcalc_path: str = "fpcalc"):
        self.fpcalc_path = fpcalc_path
        self._verify_fpcalc()
    
    def _verify_fpcalc(self):
        """Verify fpcalc is available."""
        try:
            result = subprocess.run(
                [self.fpcalc_path, "-version"],
                capture_output=True,
                text=True
            )
            logger.info(f"Chromaprint available: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError(
                "fpcalc not found. Install with: apt-get install libchromaprint-tools"
            )
    
    def generate_fingerprint(
        self,
        video_path: Path,
        length: Optional[int] = None,  # Max duration in seconds
    ) -> Tuple[bytes, float]:
        """
        Generate audio fingerprint for video.
        
        Returns:
            (fingerprint_bytes, duration_seconds)
        """
        cmd = [self.fpcalc_path, "-raw", "-json"]
        if length:
            cmd.extend(["-length", str(length)])
        cmd.append(str(video_path))
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"fpcalc failed: {result.stderr}")
        
        data = json.loads(result.stdout)
        
        # Convert fingerprint array to bytes
        fp_array = np.array(data["fingerprint"], dtype=np.int32)
        fp_bytes = fp_array.tobytes()
        
        return fp_bytes, data["duration"]
    
    def compare_fingerprints(
        self,
        fp1: bytes,
        fp2: bytes,
        threshold: float = 0.5,
    ) -> Tuple[bool, float, Optional[float]]:
        """
        Compare two fingerprints.
        
        Returns:
            (is_match, similarity_score, offset_seconds)
        """
        arr1 = np.frombuffer(fp1, dtype=np.int32)
        arr2 = np.frombuffer(fp2, dtype=np.int32)
        
        # Compute match using sliding window cross-correlation
        # This finds the best alignment between the two fingerprints
        if len(arr1) == 0 or len(arr2) == 0:
            return False, 0.0, None
        
        # For shorter fingerprint, slide over longer one
        if len(arr1) < len(arr2):
            query, reference = arr1, arr2
        else:
            query, reference = arr2, arr1
        
        best_score = 0.0
        best_offset = 0
        
        # Slide query over reference
        for offset in range(len(reference) - len(query) + 1):
            ref_slice = reference[offset:offset + len(query)]
            
            # Count matching bits using XOR and popcount
            xor = np.bitwise_xor(query, ref_slice)
            diff_bits = np.array([bin(x).count('1') for x in xor]).sum()
            total_bits = len(query) * 32  # 32 bits per int
            
            similarity = 1.0 - (diff_bits / total_bits)
            
            if similarity > best_score:
                best_score = similarity
                best_offset = offset
        
        # Convert offset to seconds (fingerprint is ~8 items per second)
        offset_seconds = best_offset / 8.0
        is_match = best_score >= threshold
        
        return is_match, best_score, offset_seconds if is_match else None
    
    def find_clip_in_source(
        self,
        clip_fp: bytes,
        source_fp: bytes,
        threshold: float = 0.5,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find where a clip appears in a source video.
        
        Returns:
            (start_time, end_time, confidence) or None if not found
        """
        is_match, score, offset = self.compare_fingerprints(clip_fp, source_fp, threshold)
        
        if not is_match:
            return None
        
        # Estimate clip duration from fingerprint length
        clip_arr = np.frombuffer(clip_fp, dtype=np.int32)
        clip_duration = len(clip_arr) / 8.0  # ~8 items per second
        
        return (offset, offset + clip_duration, score)


# Module-level instance
_chromaprint: Optional[ChromaprintService] = None

def get_chromaprint() -> ChromaprintService:
    global _chromaprint
    if _chromaprint is None:
        _chromaprint = ChromaprintService()
    return _chromaprint
```

### Index Video Job (GPU Node)

```python
# gpu-node/worker/jobs/index_video.py

"""
Job: Index a source video.
Generates ViSiL embeddings and Chromaprint fingerprint.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

from services.visil import get_embedder
from services.frame_extract import extract_all_frames, FrameExtractor
from services.chromaprint import get_chromaprint
from services.faiss_index import FAISSIndexBuilder
from services.database import get_db

logger = logging.getLogger(__name__)


async def index_video(
    video_path: str,
    video_id: int,
    fps: float = 1.0,
    save_embeddings: bool = True,
    embeddings_path: str = "/data/embeddings",
    index_path: str = "/data/index/faiss/video.index",
) -> Dict[str, Any]:
    """
    Index a source video: extract frames, generate embeddings, update index.
    
    Args:
        video_path: Path to source video
        video_id: Database ID for this video
        fps: Frames per second to extract
        save_embeddings: Whether to save raw embeddings to disk
        embeddings_path: Directory for embedding files
        index_path: Path to FAISS index
        
    Returns:
        Job result dict
    """
    video_path = Path(video_path)
    embeddings_path = Path(embeddings_path)
    index_path = Path(index_path)
    
    logger.info(f"Indexing video {video_id}: {video_path}")
    
    # Get services
    embedder = get_embedder()
    extractor = FrameExtractor(use_gpu=True)
    chromaprint = get_chromaprint()
    db = get_db()
    
    try:
        # 1. Get video info
        video_info = extractor.get_video_info(video_path)
        logger.info(f"Video info: {video_info}")
        
        # 2. Extract frames
        logger.info(f"Extracting frames at {fps} FPS...")
        frames, frame_numbers, timestamps = extract_all_frames(
            video_path, fps=fps, use_gpu=True
        )
        logger.info(f"Extracted {len(frames)} frames")
        
        # 3. Generate embeddings
        logger.info("Generating ViSiL embeddings...")
        embeddings = embedder.embed_frames(frames, batch_size=32)
        logger.info(f"Generated embeddings: {embeddings.shape}")
        
        # 4. Generate audio fingerprint
        logger.info("Generating audio fingerprint...")
        audio_fp, audio_duration = chromaprint.generate_fingerprint(video_path)
        logger.info(f"Audio fingerprint: {len(audio_fp)} bytes, {audio_duration:.1f}s")
        
        # 5. Save embeddings to disk (for index rebuilding)
        if save_embeddings:
            emb_file = embeddings_path / f"{video_id}.npz"
            emb_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                emb_file,
                embeddings=embeddings,
                frame_numbers=frame_numbers,
                timestamps=timestamps,
            )
            logger.info(f"Saved embeddings to {emb_file}")
        
        # 6. Add to FAISS index
        # Generate unique IDs: video_id * 1_000_000 + frame_index
        faiss_ids = np.array([
            video_id * 1_000_000 + i for i in range(len(embeddings))
        ], dtype=np.int64)
        
        builder = FAISSIndexBuilder(embedding_dim=embeddings.shape[1])
        
        if index_path.exists():
            # Load existing index and add to it
            import faiss
            index = faiss.read_index(str(index_path))
            builder.add_to_index(index, embeddings, faiss_ids)
        else:
            # Create new index
            index = builder.build_index(embeddings, faiss_ids)
        
        builder.save_index(index, index_path)
        
        # 7. Update database
        await db.update_source_video(
            video_id=video_id,
            status="complete",
            frame_count=len(frames),
            fps=video_info["fps"],
            duration_seconds=video_info["duration_seconds"],
            resolution=video_info["resolution"],
        )
        
        # Store frame metadata
        await db.insert_frame_embeddings(
            video_id=video_id,
            frame_numbers=frame_numbers,
            timestamps=timestamps,
            faiss_ids=faiss_ids.tolist(),
        )
        
        # Store audio fingerprint
        await db.insert_audio_fingerprint(
            video_id=video_id,
            fingerprint=audio_fp,
            duration=audio_duration,
        )
        
        return {
            "status": "success",
            "video_id": video_id,
            "frames_indexed": len(frames),
            "embedding_shape": list(embeddings.shape),
            "audio_duration": audio_duration,
        }
        
    except Exception as e:
        logger.exception(f"Failed to index video {video_id}")
        await db.update_source_video(
            video_id=video_id,
            status="failed",
            error_message=str(e),
        )
        raise
```

### Query Clip Job (GPU Node)

```python
# gpu-node/worker/jobs/query_clip.py

"""
Job: Query a clip against the indexed source videos.
Returns candidate matches with confidence scores.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import logging

from services.visil import get_embedder
from services.frame_extract import extract_all_frames
from services.chromaprint import get_chromaprint
from services.faiss_index import FAISSIndexSearcher
from services.database import get_db

logger = logging.getLogger(__name__)


async def query_clip(
    clip_path: str,
    clip_id: int,
    fps: float = 1.0,
    top_k_frames: int = 100,
    top_k_videos: int = 50,
    index_path: str = "/data/index/faiss/video.index",
    audio_threshold: float = 0.4,
) -> Dict[str, Any]:
    """
    Query a clip against indexed source videos.
    
    Pipeline:
    1. Extract clip frames and generate embeddings
    2. Search FAISS for similar frames
    3. Aggregate frame matches to video-level candidates
    4. Verify with audio fingerprinting
    5. Return ranked matches
    
    Args:
        clip_path: Path to query clip
        clip_id: Database ID for this clip
        fps: Frames per second to extract
        top_k_frames: Number of nearest neighbors per frame
        top_k_videos: Number of candidate videos to return
        index_path: Path to FAISS index
        audio_threshold: Chromaprint match threshold
        
    Returns:
        Job result with matches
    """
    clip_path = Path(clip_path)
    index_path = Path(index_path)
    
    logger.info(f"Querying clip {clip_id}: {clip_path}")
    
    # Get services
    embedder = get_embedder()
    chromaprint = get_chromaprint()
    searcher = FAISSIndexSearcher(index_path)
    db = get_db()
    
    try:
        # 1. Extract frames and generate embeddings
        logger.info("Extracting clip frames...")
        frames, frame_numbers, timestamps = extract_all_frames(clip_path, fps=fps)
        logger.info(f"Extracted {len(frames)} frames from clip")
        
        logger.info("Generating clip embeddings...")
        clip_embeddings = embedder.embed_frames(frames, batch_size=32)
        
        # 2. Search FAISS for each frame
        logger.info(f"Searching FAISS index (k={top_k_frames})...")
        distances, faiss_ids = searcher.search(clip_embeddings, k=top_k_frames)
        
        # 3. Aggregate frame matches to video-level scores
        # faiss_id format: video_id * 1_000_000 + frame_index
        video_scores = defaultdict(list)
        video_frame_matches = defaultdict(list)
        
        for clip_frame_idx, (frame_distances, frame_ids) in enumerate(zip(distances, faiss_ids)):
            for dist, fid in zip(frame_distances, frame_ids):
                if fid < 0:  # Invalid result
                    continue
                    
                video_id = int(fid // 1_000_000)
                source_frame_idx = int(fid % 1_000_000)
                
                video_scores[video_id].append(float(dist))
                video_frame_matches[video_id].append({
                    "clip_frame": clip_frame_idx,
                    "source_frame": source_frame_idx,
                    "distance": float(dist),
                })
        
        # Score videos by average similarity of matched frames
        video_rankings = []
        for video_id, scores in video_scores.items():
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            match_count = len(scores)
            
            video_rankings.append({
                "video_id": video_id,
                "avg_similarity": avg_score,
                "max_similarity": max_score,
                "matched_frames": match_count,
                "frame_matches": video_frame_matches[video_id][:20],  # Top 20 matches
            })
        
        # Sort by average similarity (higher = better for cosine sim)
        video_rankings.sort(key=lambda x: x["avg_similarity"], reverse=True)
        candidates = video_rankings[:top_k_videos]
        
        logger.info(f"Found {len(candidates)} candidate videos")
        
        # 4. Verify top candidates with audio fingerprinting
        logger.info("Generating clip audio fingerprint...")
        try:
            clip_audio_fp, clip_audio_duration = chromaprint.generate_fingerprint(clip_path)
            
            logger.info("Verifying candidates with audio...")
            for candidate in candidates:
                source_audio = await db.get_audio_fingerprint(candidate["video_id"])
                if source_audio:
                    result = chromaprint.find_clip_in_source(
                        clip_audio_fp, 
                        source_audio["fingerprint"],
                        threshold=audio_threshold,
                    )
                    if result:
                        start_time, end_time, confidence = result
                        candidate["audio_match"] = {
                            "start_time": start_time,
                            "end_time": end_time,
                            "confidence": confidence,
                        }
                    else:
                        candidate["audio_match"] = None
                else:
                    candidate["audio_match"] = None
        except Exception as e:
            logger.warning(f"Audio fingerprinting failed: {e}")
            for candidate in candidates:
                candidate["audio_match"] = None
        
        # 5. Compute final scores combining video and audio
        for candidate in candidates:
            video_score = candidate["avg_similarity"]
            audio_score = (
                candidate["audio_match"]["confidence"] 
                if candidate.get("audio_match") 
                else 0.0
            )
            
            # Weighted combination: 60% video, 40% audio
            candidate["final_score"] = 0.6 * video_score + 0.4 * audio_score
        
        # Re-sort by final score
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        # 6. Estimate timestamps from frame matches
        for candidate in candidates:
            await _estimate_timestamps(candidate, db, timestamps)
        
        # 7. Store results in database
        await db.update_query_clip(clip_id=clip_id, status="complete")
        
        for rank, candidate in enumerate(candidates[:10]):  # Store top 10
            await db.insert_match(
                clip_id=clip_id,
                source_video_id=candidate["video_id"],
                confidence_score=candidate["final_score"],
                source_start_time=candidate.get("estimated_start"),
                source_end_time=candidate.get("estimated_end"),
                match_method="visil+chromaprint" if candidate.get("audio_match") else "visil",
            )
        
        return {
            "status": "success",
            "clip_id": clip_id,
            "candidates": candidates,
            "total_candidates": len(video_rankings),
        }
        
    except Exception as e:
        logger.exception(f"Failed to query clip {clip_id}")
        await db.update_query_clip(
            clip_id=clip_id,
            status="failed",
            error_message=str(e),
        )
        raise


async def _estimate_timestamps(
    candidate: Dict,
    db,
    clip_timestamps: List[float],
) -> None:
    """Estimate where in the source video the clip appears."""
    if not candidate.get("frame_matches"):
        return
    
    # Get source video frame timestamps from database
    frame_metadata = await db.get_frame_metadata(candidate["video_id"])
    if not frame_metadata:
        return
    
    # Map source frame indices to timestamps
    source_timestamps = {fm["frame_number"]: fm["timestamp"] for fm in frame_metadata}
    
    # Find best matching sequence
    matches = candidate["frame_matches"]
    source_times = []
    
    for match in matches:
        source_frame = match["source_frame"]
        if source_frame in source_timestamps:
            source_times.append(source_timestamps[source_frame])
    
    if source_times:
        candidate["estimated_start"] = min(source_times)
        candidate["estimated_end"] = max(source_times) + (clip_timestamps[-1] - clip_timestamps[0])
```

### API Routes (NAS)

```python
# nas/api/routes/query.py

"""
API routes for querying clips.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import uuid
import shutil

from services.job_queue import enqueue_job
from services.database import get_db
from config import settings

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    clip_path: str  # Path relative to /data/videos/clips/
    
    
class QueryResponse(BaseModel):
    job_id: str
    status: str
    message: str


class MatchResult(BaseModel):
    source_video_id: int
    source_path: str
    confidence_score: float
    start_time: Optional[float]
    end_time: Optional[float]
    match_method: str
    audio_match: Optional[dict]


@router.post("/", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """
    Submit a clip for matching against source videos.
    Returns a job_id to check status.
    """
    db = get_db()
    
    # Validate clip exists
    clip_path = Path(settings.videos_path) / "clips" / request.clip_path
    if not clip_path.exists():
        raise HTTPException(404, f"Clip not found: {request.clip_path}")
    
    # Create clip record
    clip_id = await db.create_query_clip(
        filepath=str(clip_path),
        filename=clip_path.name,
    )
    
    # Enqueue job
    job_id = str(uuid.uuid4())
    await enqueue_job(
        job_id=job_id,
        job_type="query_clip",
        payload={
            "clip_path": str(clip_path),
            "clip_id": clip_id,
        },
    )
    
    return QueryResponse(
        job_id=job_id,
        status="queued",
        message=f"Query job submitted for {request.clip_path}",
    )


@router.post("/upload", response_model=QueryResponse)
async def upload_and_query(file: UploadFile = File(...)):
    """
    Upload a clip file and submit for matching.
    """
    db = get_db()
    
    # Save uploaded file
    upload_dir = Path(settings.videos_path) / "clips" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    clip_path = upload_dir / filename
    
    with open(clip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Create clip record
    clip_id = await db.create_query_clip(
        filepath=str(clip_path),
        filename=file.filename,
    )
    
    # Enqueue job
    job_id = str(uuid.uuid4())
    await enqueue_job(
        job_id=job_id,
        job_type="query_clip",
        payload={
            "clip_path": str(clip_path),
            "clip_id": clip_id,
        },
    )
    
    return QueryResponse(
        job_id=job_id,
        status="queued",
        message=f"Uploaded and queued: {file.filename}",
    )


@router.get("/results/{clip_id}", response_model=List[MatchResult])
async def get_results(clip_id: int):
    """
    Get match results for a clip.
    """
    db = get_db()
    
    matches = await db.get_matches_for_clip(clip_id)
    if not matches:
        raise HTTPException(404, f"No results found for clip {clip_id}")
    
    results = []
    for match in matches:
        source = await db.get_source_video(match["source_video_id"])
        results.append(MatchResult(
            source_video_id=match["source_video_id"],
            source_path=source["filepath"] if source else "unknown",
            confidence_score=match["confidence_score"],
            start_time=match["source_start_time"],
            end_time=match["source_end_time"],
            match_method=match["match_method"],
            audio_match=None,  # Could store this too
        ))
    
    return results
```

---

## Setup & Deployment

### NAS Setup

```bash
# 1. Clone repository
git clone <repo-url> /path/to/video-clip-matcher
cd video-clip-matcher/nas

# 2. Configure environment
cp .env.example .env
# Edit .env:
#   POSTGRES_PASSWORD=<secure-password>
#   DATA_PATH=/volume1/video-data  # Path to your video storage

# 3. Create data directories
mkdir -p ${DATA_PATH}/videos/sources
mkdir -p ${DATA_PATH}/videos/clips
mkdir -p ${DATA_PATH}/index/faiss
mkdir -p ${DATA_PATH}/embeddings

# 4. Start services
docker-compose up -d

# 5. Verify
curl http://localhost:8000/health
```

### GPU Node Setup

```bash
# 1. Clone repository (or mount from NAS)
git clone <repo-url> /path/to/video-clip-matcher
cd video-clip-matcher/gpu-node

# 2. Configure environment
cp .env.example .env
# Edit .env:
#   NAS_HOST=192.168.1.100          # NAS IP address
#   POSTGRES_PASSWORD=<same-as-nas>
#   NAS_MOUNT_PATH=/mnt/nas         # Where NAS is mounted

# 3. Mount NAS storage (if using NFS)
sudo mount -t nfs ${NAS_HOST}:/volume1/video-data /mnt/nas

# 4. Download ViSiL weights
cd worker
python scripts/download_weights.py

# 5. Start worker
docker-compose up -d

# 6. Verify GPU access
docker exec -it vcm-worker nvidia-smi
```

### Usage Workflow

```bash
# 1. Add source videos to NAS
cp /path/to/videos/*.mp4 /mnt/nas/videos/sources/

# 2. Index all source videos (run once)
python scripts/index_all_sources.py

# 3. Add clips to match
cp /path/to/clips/*.mp4 /mnt/nas/videos/clips/

# 4. Query all clips
python scripts/query_all_clips.py

# 5. Export results
python scripts/export_results.py --output matches.csv

# Or use API directly:
curl -X POST http://nas-ip:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"clip_path": "my_clip.mp4"}'
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection string |
| `REDIS_URL` | - | Redis connection string |
| `FAISS_INDEX_PATH` | `/data/index/faiss/video.index` | Path to FAISS index |
| `VIDEOS_PATH` | `/data/videos` | Root path for video storage |
| `EMBEDDINGS_PATH` | `/data/embeddings` | Path for raw embeddings |
| `MODEL_PATH` | `/app/models/visil` | Path to ViSiL weights |
| `FRAME_RATE` | `1` | Frames per second to extract |
| `BATCH_SIZE` | `32` | Batch size for ViSiL inference |
| `LOG_LEVEL` | `INFO` | Logging level |

### Tuning Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `fps` (extraction) | 1.0 | Higher = more precision, slower indexing |
| `top_k_frames` | 100 | FAISS neighbors per frame |
| `top_k_videos` | 50 | Candidate videos to verify |
| `audio_threshold` | 0.4 | Chromaprint match threshold |
| `efSearch` (HNSW) | 128 | Higher = better recall, slower |
| `efConstruction` | 200 | Higher = better index quality |

---

## Estimated Performance

For your scale (hundreds of clips, thousands of source videos):

| Operation | Time | Hardware |
|-----------|------|----------|
| Index 1 source (2 min video) | ~30-60s | GPU Node |
| Index 1000 sources | ~14-28 hours | GPU Node |
| Query 1 clip | ~2-5s | GPU Node |
| Query 500 clips (batch) | ~15-40 min | GPU Node |
| FAISS search (1 query) | ~1-10ms | NAS (CPU) |
| Audio verification | ~100-500ms | NAS (CPU) |

### Storage Requirements

| Data | Size (1000 videos) |
|------|-------------------|
| FAISS index (HNSW) | ~500MB - 1GB |
| Raw embeddings | ~2-4GB |
| Audio fingerprints | ~100MB |
| PostgreSQL | ~500MB |

---

## Future Enhancements

- [ ] Web UI for manual match verification
- [ ] FFmpeg signature filter as additional verification step
- [ ] Incremental index updates (add videos without full rebuild)
- [ ] Distributed workers for parallel indexing
- [ ] Scene-level matching (detect scene boundaries first)
- [ ] Export to video editing software (EDL/XML)
