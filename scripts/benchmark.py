from __future__ import annotations

import argparse
import os
import time

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ClipDNA Desktop API")
    parser.add_argument(
        "--api-url",
        default=os.getenv("CLIPDNA_API_URL", "http://localhost:49321"),
        help="Base URL for ClipDNA Desktop API",
    )
    parser.add_argument("--clip-path", required=True, help="Clip path relative to clips/")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    timings = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        response = httpx.post(
            f"{args.api_url}/query/",
            json={"clip_path": args.clip_path},
            timeout=60,
        )
        response.raise_for_status()
        timings.append(time.perf_counter() - start)

    avg = sum(timings) / len(timings)
    print(f"Average /query response: {avg:.3f}s")
    print(f"Timings: {timings}")


if __name__ == "__main__":
    main()
