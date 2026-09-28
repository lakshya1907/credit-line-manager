"""
src/api/jobs.py's registry, unit-tested directly (not through a real
run_all.main() -- that's a real ~10s retrain, exercised for real manually;
see CLAUDE.md's Database/API sections) and the POST /runs endpoint
contract with run_pipeline_job mocked out to a fast stub.
"""

import sys
import types

import pytest

from src.api.jobs import create_job, get_job, run_pipeline_job


def test_create_job_starts_in_running_state():
    job_id = create_job()
    job = get_job(job_id)
    assert job["status"] == "running"
    assert job["run_id"] is None
    assert job["finished_at"] is None


def test_get_job_returns_none_for_unknown_id():
    assert get_job("not-a-real-job-id") is None


def test_get_job_returns_a_copy_not_the_live_dict():
    job_id = create_job()
    snapshot = get_job(job_id)
    snapshot["status"] = "tampered"
    assert get_job(job_id)["status"] == "running"


def test_run_pipeline_job_marks_completed_on_success(monkeypatch):
    fake_run_all = types.ModuleType("run_all")
    fake_run_all.main = lambda raw_path: "fake-run-id-123"
    fake_sync = types.ModuleType("sync_run_to_db")
    fake_sync.sync_run = lambda run_id, session: None
    monkeypatch.setitem(sys.modules, "run_all", fake_run_all)
    monkeypatch.setitem(sys.modules, "sync_run_to_db", fake_sync)

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("src.db.session.get_session_factory", lambda engine: (lambda: _FakeSession()))
    monkeypatch.setattr("src.db.session.get_engine", lambda: object())

    job_id = create_job()
    run_pipeline_job(job_id, "data/raw/uci_credit.csv")

    job = get_job(job_id)
    assert job["status"] == "completed"
    assert job["run_id"] == "fake-run-id-123"
    assert job["finished_at"] is not None


def test_run_pipeline_job_marks_failed_on_exception(monkeypatch):
    fake_run_all = types.ModuleType("run_all")

    def _boom(raw_path):
        raise RuntimeError("training blew up")

    fake_run_all.main = _boom
    monkeypatch.setitem(sys.modules, "run_all", fake_run_all)

    job_id = create_job()
    run_pipeline_job(job_id, "data/raw/uci_credit.csv")

    job = get_job(job_id)
    assert job["status"] == "failed"
    assert "training blew up" in job["error"]


def test_post_runs_returns_202_with_a_running_job(api_client, monkeypatch):
    import src.api.routers.runs as runs_module

    monkeypatch.setattr(runs_module, "run_pipeline_job", lambda job_id, data_path: None)  # no-op background task

    r = api_client.post("/runs")
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "running"
    assert body["job_id"]
