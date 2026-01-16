# Architecture

ClipDNA Desktop is split across two nodes:

- **NAS**: PostgreSQL + Redis + FastAPI API
- **GPU Node**: ViSiL embedding worker and FAISS indexing

Shared storage (`/data`) is mounted to both services, and Redis is used as the job queue. The GPU node performs embedding extraction, FAISS indexing, and Chromaprint matching, then persists results back to PostgreSQL.
