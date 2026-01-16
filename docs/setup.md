# Setup

## NAS

1. Copy the environment file:

```bash
cd nas
cp .env.example .env
```

2. Update `.env` with your credentials, storage path, and `API_PORT`.
3. Create required directories:

```bash
mkdir -p ${DATA_PATH}/videos/sources
mkdir -p ${DATA_PATH}/videos/clips
mkdir -p ${DATA_PATH}/index/faiss
mkdir -p ${DATA_PATH}/embeddings
```

4. Start services:

```bash
docker-compose up -d
```

5. Verify:

```bash
curl http://localhost:${API_PORT}/health
```

## GPU Node

1. Copy the environment file:

```bash
cd gpu-node
cp .env.example .env
```

2. Mount the NAS data directory (NFS/SMB) to `NAS_MOUNT_PATH`.
3. Start worker:

```bash
docker-compose up -d
```

4. Verify GPU access:

```bash
docker exec -it clipdna-desktop-worker nvidia-smi
```
