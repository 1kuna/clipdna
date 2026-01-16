import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL") or not os.getenv("REDIS_URL"),
    reason="DATABASE_URL and REDIS_URL required for API integration test",
)
def test_health_endpoint():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "nas" / "api"))
    from main import app

    with TestClient(app) as client:
        response = client.get("/health/")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
