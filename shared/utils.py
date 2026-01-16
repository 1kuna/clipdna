from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def sanitize_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in {"-", "_", "."})


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
