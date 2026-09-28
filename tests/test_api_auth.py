"""
src/api/auth.py's optional API-key check. Uses a fresh TestClient (not the
shared api_client fixture) because it needs to mutate src.api.auth.API_KEY
at import-patched-module level before the app's routes evaluate the
dependency, and restore it afterward so other tests (which assume auth is
disabled) aren't affected.
"""

import pytest
from fastapi.testclient import TestClient

import src.api.auth as auth_module
from src.api.app import app


@pytest.fixture
def with_api_key(monkeypatch):
    monkeypatch.setattr(auth_module, "API_KEY", "test-secret-key")
    yield "test-secret-key"


def test_trigger_run_succeeds_without_api_key_when_disabled(api_client, monkeypatch):
    import src.api.routers.runs as runs_module
    monkeypatch.setattr(runs_module, "run_pipeline_job", lambda job_id, data_path: None)

    r = api_client.post("/runs")
    assert r.status_code == 202


def test_trigger_run_requires_api_key_when_enabled(with_api_key, monkeypatch):
    import src.api.routers.runs as runs_module
    monkeypatch.setattr(runs_module, "run_pipeline_job", lambda job_id, data_path: None)

    with TestClient(app) as client:
        r = client.post("/runs")
        assert r.status_code == 401

        r = client.post("/runs", headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401

        r = client.post("/runs", headers={"X-API-Key": with_api_key})
        assert r.status_code == 202


def test_score_endpoint_requires_api_key_when_enabled(with_api_key):
    with TestClient(app) as client:
        r = client.post("/customers/1/score", json={})
        assert r.status_code == 401

        r = client.post("/customers/1/score", json={}, headers={"X-API-Key": with_api_key})
        # 401 must come from the auth dependency, not fall through to it;
        # any non-401 here (404 unknown customer, 503 no models, 200) means
        # the key was accepted and the request reached the real handler.
        assert r.status_code != 401


def test_read_endpoints_never_require_api_key(with_api_key, api_client):
    # GET endpoints have no require_api_key dependency at all -- read
    # access stays open regardless of API_KEY.
    r = api_client.get("/runs")
    assert r.status_code == 200
