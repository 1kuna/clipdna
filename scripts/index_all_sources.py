from __future__ import annotations

import argparse
import os

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue indexing for all source videos")
    parser.add_argument(
        "--api-url",
        default=os.getenv("CLIPDNA_API_URL", "http://localhost:49321"),
        help="Base URL for ClipDNA Desktop API",
    )
    args = parser.parse_args()

    response = httpx.post(f"{args.api_url}/index/", json={"scan_all": True}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    print(f"Queued {payload.get('queued')} videos")
    print(payload)


if __name__ == "__main__":
    main()
