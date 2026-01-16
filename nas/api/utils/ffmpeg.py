from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def probe_video(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration:stream=width,height,avg_frame_rate",
        "-of",
        "json",
        str(file_path),
    ]
    logger.debug("Running ffprobe: %s", " ".join(command))
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return data
