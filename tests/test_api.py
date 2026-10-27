"""API tests. They train a quick model first if none exists."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from pdm.config import RISK_MODEL_PATH  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module=None):
    import os
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path_factory.mktemp('db') / 't.db'}"
    if not RISK_MODEL_PATH.exists():
        import sys
        from pdm import train
        sys.argv = ["train", "--skip-torch"]
        train.main()
    from pdm.api import app
    with TestClient(app) as c:
        yield c


def _readings(vib):
    return [{"timestamp": f"2025-03-01T{h:02d}:00:00", "temperature": 70, "vibration": vib,
             "pressure": 100, "rpm": 1500, "current": 12} for h in range(24)]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["models_loaded"] is True


def test_predict_normal_vs_degraded(client):
    normal = client.post("/predict", json={"machine_id": "M1", "readings": _readings(0.30)}).json()
    bad_readings = _readings(0.30)
    for i, r in enumerate(bad_readings[-12:]):
        r["vibration"] = 0.6 + 0.05 * i
        r["temperature"] = 76 + 0.6 * i
        r["current"] = 14 + 0.3 * i
    bad = client.post("/predict", json={"machine_id": "M1", "readings": bad_readings}).json()
    assert bad["failure_risk"] > normal["failure_risk"]


def test_empty_readings_rejected(client):
    assert client.post("/predict", json={"machine_id": "M1", "readings": []}).status_code == 422
