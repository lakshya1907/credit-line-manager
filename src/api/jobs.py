"""
src/api/jobs.py
──────────────────
In-memory background-job tracking for POST /runs.

Deliberate scope limit: this is a single-process, in-memory dict, not a
persisted job queue (no Celery/Redis/RQ). That's the right size for this
project today (single instance, low request volume) -- but it means job
status is lost on process restart, and it would NOT work correctly behind
multiple uvicorn workers (each worker has its own dict, so a status
lookup could land on a worker that never ran the job). If this API is ever
run with more than one worker, replace this with a persisted jobs table
(or a real queue) rather than trying to patch around it.
"""

import datetime
import threading
import uuid
from typing import Optional

_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def create_job() -> str:
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "run_id": None,
            "error": None,
            "started_at": datetime.datetime.now(datetime.timezone.utc),
            "finished_at": None,
        }
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def _update_job(job_id: str, **updates) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(updates)


def run_pipeline_job(job_id: str, raw_path: str) -> None:
    """The actual background task: run the full pipeline, then sync the
    resulting run straight into Postgres so it's immediately queryable via
    GET /runs/{run_id} -- without this, POST /runs would produce a run
    nothing in the API could see until someone separately ran
    sync_run_to_db.py."""
    import run_all
    import sync_run_to_db
    from src.db.session import get_engine, get_session_factory

    try:
        run_id = run_all.main(raw_path)
        session_factory = get_session_factory(get_engine())
        with session_factory() as session:
            sync_run_to_db.sync_run(run_id, session)
        _update_job(
            job_id, status="completed", run_id=run_id,
            finished_at=datetime.datetime.now(datetime.timezone.utc),
        )
    except Exception as e:
        _update_job(
            job_id, status="failed", error=str(e),
            finished_at=datetime.datetime.now(datetime.timezone.utc),
        )
