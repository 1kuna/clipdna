# Tuning

## Key Parameters

- `FRAME_RATE` (worker): frames per second extracted
- `BATCH_SIZE` (worker): ViSiL inference batch size
- `top_k_frames` (query): FAISS neighbors per clip frame
- `top_k_videos` (query): candidate videos returned
- `audio_threshold` (query): Chromaprint confidence threshold

## Recommended Starting Values

- `FRAME_RATE=1`
- `BATCH_SIZE=32`
- `top_k_frames=100`
- `top_k_videos=50`
- `audio_threshold=0.4`
