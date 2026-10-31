"""API tests against the generated demo scene (run create_demo_scene.py first)."""
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    subprocess.run([sys.executable, "create_demo_scene.py"], cwd=ROOT, check=True)
    from backend.main import app
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_initialize_and_path(client):
    r = client.post("/api/v1/scenes/demo_office/initialize")
    assert r.status_code == 200 and r.json()["walkable_area_m2"] > 20
    p = client.post("/api/v1/navigation/plan-path", json={
        "scene_id": "demo_office", "start_xyz": [1, 1.5, 1], "goal_xyz": [8, 0, 6]})
    assert p.status_code == 200
    assert p.json()["min_obstacle_clearance_m"] >= 0.28


def test_poi_search(client):
    client.post("/api/v1/scenes/demo_office/initialize")
    r = client.post("/api/v1/semantics/search", json={"scene_id": "demo_office", "query_text": "espresso machine"})
    assert r.json()[0]["label"] == "espresso machine"
    assert client.post("/api/v1/semantics/search", json={
        "scene_id": "demo_office", "query_text": "zebra"}).json() == []


def test_unknown_scene(client):
    assert client.post("/api/v1/scenes/nope/initialize").status_code == 404


def test_map_status_and_capture(client):
    import base64
    import numpy as np
    client.post("/api/v1/scenes/demo_office/initialize")
    m = client.get("/api/v1/scenes/demo_office/map").json()
    status = np.frombuffer(base64.b64decode(m["status_b64"]), dtype=np.uint8)
    assert status.size == m["nx"] * m["nz"]
    assert set(np.unique(status)) <= {0, 1, 2}
    assert (status == 1).sum() > 0 and (status == 2).sum() > 0


def test_map_requires_init(client):
    assert client.get("/api/v1/scenes/not_loaded/map").status_code == 400
