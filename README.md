# ClipDNA Desktop

ClipDNA Desktop is a video fingerprinting system that matches short clips to their source videos using ViSiL embeddings, FAISS indexing, and Chromaprint audio fingerprinting. This repository is structured to coexist with a future ClipDNA iOS app (see `CLIPDNA_IOS_SPEC.md`).

## Quick Start

### NAS (API + Database)

```bash
cd nas
cp .env.example .env
# Edit .env with your POSTGRES_PASSWORD, DATA_PATH, and API_PORT
mkdir -p ${DATA_PATH}/videos/sources
mkdir -p ${DATA_PATH}/videos/clips
mkdir -p ${DATA_PATH}/index/faiss
mkdir -p ${DATA_PATH}/embeddings

docker-compose up -d
curl http://localhost:${API_PORT}/health
```

### GPU Node (Worker)

```bash
cd gpu-node
cp .env.example .env
# Edit .env with NAS_HOST, POSTGRES_PASSWORD, NAS_MOUNT_PATH

# Mount NAS storage (NFS or SMB)
# sudo mount -t nfs ${NAS_HOST}:/volume1/clipdna-data /mnt/nas

docker-compose up -d
```

### Index and Query

```bash
# Point scripts at your API port
export CLIPDNA_API_URL="http://localhost:${API_PORT}"

# Index all sources
python scripts/index_all_sources.py

# Query all clips
python scripts/query_all_clips.py
```

## Layout

- `nas/` FastAPI API + Postgres + Redis
- `gpu-node/` GPU worker for embedding and matching
- `shared/` Shared schemas and utilities
- `scripts/` Batch scripts
- `tests/` Pytest suite
- `docs/` Setup and usage details

## Setup Wizard (recommended)

Run the wizard in a persistent terminal (works well with DGX Spark custom apps):

```bash
./scripts/clipdna_wizard.py nas
```

For the GPU worker:

```bash
./scripts/clipdna_wizard.py dgx
```

If the DGX Spark UI requires a bash launch script, use:

```bash
CLIPDNA_MODE=nas ./scripts/clipdna_spark.sh
```

```bash
CLIPDNA_MODE=dgx ./scripts/clipdna_spark.sh
```

## iOS App

The upcoming ClipDNA iOS app will live alongside this stack and share naming conventions (ClipDNA branding, environment variables). See `CLIPDNA_IOS_SPEC.md` for the mobile spec.

## DGX Spark Custom App (example)

Use a random high port so you avoid conflicts; it only needs to match the API port you set in `nas/.env`.

- Name: `ClipDNA Desktop API`
- Port: `49321` (or any open port you choose)
- Launch script:

```bash
cd /path/to/clipdna-desktop/nas
cp .env.example .env
sed -i '' 's/API_PORT=49321/API_PORT=49321/' .env
docker-compose up -d
```
