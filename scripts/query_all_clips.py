from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue queries for all clip videos")
    parser.add_argument(
        "--api-url",
        default=os.getenv("CLIPDNA_API_URL", "http://localhost:49321"),
        help="Base URL for ClipDNA Desktop API",
    )
    parser.add_argument(
        "--videos-path",
        default=os.getenv("CLIPDNA_VIDEOS_PATH", "/data/videos"),
        help="Root videos path (must include clips/)",
    )
    args = parser.parse_args()

    clips_dir = Path(args.videos_path) / "clips"
    clip_files = [
        path for path in clips_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not clip_files:
        print("No clips found")
        return

    for clip in clip_files:
        relative = clip.relative_to(clips_dir)
        response = httpx.post(
            f"{args.api_url}/query/",
            json={"clip_path": str(relative)},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        print(f"Queued {relative}: {payload.get('job_id')}")


if __name__ == "__main__":
    main()
