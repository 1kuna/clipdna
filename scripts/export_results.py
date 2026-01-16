from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from sqlalchemy import create_engine, text


def main() -> None:
    parser = argparse.ArgumentParser(description="Export match results to CSV or JSON")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL connection string",
    )
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    args = parser.parse_args()

    if not args.database_url:
        raise SystemExit("DATABASE_URL is required")

    engine = create_engine(args.database_url)
    query = text(
        """
        SELECT
            m.id AS match_id,
            qc.id AS clip_id,
            qc.filepath AS clip_path,
            sv.id AS source_id,
            sv.filepath AS source_path,
            m.confidence_score,
            m.source_start_time,
            m.source_end_time,
            m.match_method,
            m.created_at
        FROM matches m
        JOIN query_clips qc ON m.query_clip_id = qc.id
        JOIN source_videos sv ON m.source_video_id = sv.id
        ORDER BY m.confidence_score DESC
        """
    )

    with engine.connect() as conn:
        rows = [dict(row) for row in conn.execute(query)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=str)
    else:
        if not rows:
            print("No results found")
            return
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Exported {len(rows)} results to {output_path}")


if __name__ == "__main__":
    main()
