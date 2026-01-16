-- ClipDNA Desktop schema initialization

-- Source videos that have been indexed
CREATE TABLE IF NOT EXISTS source_videos (
    id SERIAL PRIMARY KEY,
    filepath VARCHAR(1024) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    duration_seconds FLOAT,
    frame_count INTEGER,
    fps FLOAT,
    resolution VARCHAR(20),
    indexed_at TIMESTAMP DEFAULT NOW(),
    index_status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT
);

-- Frame-level embeddings metadata (actual vectors in FAISS)
CREATE TABLE IF NOT EXISTS frame_embeddings (
    id SERIAL PRIMARY KEY,
    source_video_id INTEGER REFERENCES source_videos(id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    timestamp_seconds FLOAT NOT NULL,
    faiss_id BIGINT NOT NULL,
    UNIQUE(source_video_id, frame_number)
);
CREATE INDEX IF NOT EXISTS idx_frame_embeddings_faiss ON frame_embeddings(faiss_id);

-- Audio fingerprints
CREATE TABLE IF NOT EXISTS audio_fingerprints (
    id SERIAL PRIMARY KEY,
    source_video_id INTEGER REFERENCES source_videos(id) ON DELETE CASCADE,
    fingerprint BYTEA NOT NULL,
    duration_seconds FLOAT
);

-- Query clips
CREATE TABLE IF NOT EXISTS query_clips (
    id SERIAL PRIMARY KEY,
    filepath VARCHAR(1024) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    duration_seconds FLOAT,
    submitted_at TIMESTAMP DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT
);

-- Match results
CREATE TABLE IF NOT EXISTS matches (
    id SERIAL PRIMARY KEY,
    query_clip_id INTEGER REFERENCES query_clips(id) ON DELETE CASCADE,
    source_video_id INTEGER REFERENCES source_videos(id) ON DELETE CASCADE,
    confidence_score FLOAT NOT NULL,
    source_start_time FLOAT,
    source_end_time FLOAT,
    match_method VARCHAR(50),
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_matches_clip ON matches(query_clip_id);
CREATE INDEX IF NOT EXISTS idx_matches_confidence ON matches(confidence_score DESC);

-- Job queue tracking (supplements Redis)
CREATE TABLE IF NOT EXISTS jobs (
    id VARCHAR(36) PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'queued',
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    result JSONB
);
