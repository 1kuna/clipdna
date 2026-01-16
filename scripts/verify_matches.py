from __future__ import annotations

import argparse
import os

from sqlalchemy import create_engine, text


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect matches for a clip")
    parser.add_argument("--clip-id", type=int, required=True)
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL connection string",
    )
    args = parser.parse_args()

    if not args.database_url:
        raise SystemExit("DATABASE_URL is required")

    engine = create_engine(args.database_url)
    query = text(
        """
        SELECT
            m.id,
            m.confidence_score,
            m.source_start_time,
            m.source_end_time,
            m.match_method,
            sv.filepath AS source_path
        FROM matches m
        JOIN source_videos sv ON m.source_video_id = sv.id
        WHERE m.query_clip_id = :clip_id
        ORDER BY m.confidence_score DESC
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query, {"clip_id": args.clip_id}).fetchall()

    if not rows:
        print("No matches found")
        return

    for row in rows:
        print(
            f"Match {row.id}: score={row.confidence_score:.3f} "
            f"time={row.source_start_time}-{row.source_end_time} "
            f"method={row.match_method} source={row.source_path}"
        )


if __name__ == "__main__":
    main()
