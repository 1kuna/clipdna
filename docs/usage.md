# Usage

## Index Source Videos

Copy source videos into `${DATA_PATH}/videos/sources/` then run:

```bash
python scripts/index_all_sources.py
```

## Query Clips

Copy clip files into `${DATA_PATH}/videos/clips/` then run:

```bash
python scripts/query_all_clips.py
```

## Export Results

```bash
python scripts/export_results.py --output matches.csv --format csv
```

## API Examples

```bash
curl -X POST http://localhost:${API_PORT}/query/ \
  -H "Content-Type: application/json" \
  -d '{"clip_path": "my_clip.mp4"}'
```
