from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from src.api.deps import get_db
from src.api.jobs import create_job, get_job, run_pipeline_job
from src.api.schemas import JobStatus, ModelRunDetail, ModelRunSummary, PaginatedRecommendations
from src.db.models import ModelRun, Recommendation

router = APIRouter(prefix="/runs", tags=["runs"])

# run_all.py's own default raw-data path; imported lazily inside the route
# (not at module import time) so importing this router never requires
# run_all.py's heavier dependency chain to have succeeded first.
_DEFAULT_RAW = "data/raw/uci_credit.csv"


@router.get("", response_model=list[ModelRunSummary])
def list_runs(limit: int = Query(20, le=200), offset: int = 0, db: Session = Depends(get_db)):
    """Run history, newest first -- the whole reason this schema exists:
    before it, run_all.py overwrote its own history on every run."""
    stmt = select(ModelRun).order_by(ModelRun.started_at.desc()).limit(limit).offset(offset)
    return db.execute(stmt).scalars().all()


@router.get("/{run_id}", response_model=ModelRunDetail)
def get_run(run_id: str, db: Session = Depends(get_db)):
    stmt = (
        select(ModelRun)
        .where(ModelRun.run_id == run_id)
        .options(
            selectinload(ModelRun.portfolio_runs),
            selectinload(ModelRun.stress_test_results),
            selectinload(ModelRun.segment_metrics),
            selectinload(ModelRun.backtest_results),
            selectinload(ModelRun.fairness_checks),
        )
    )
    run = db.execute(stmt).scalar_one_or_none()
    if run is None:
        raise HTTPException(404, f"run_id {run_id!r} not found")
    return run


@router.get("/{run_id}/recommendations", response_model=PaginatedRecommendations)
def get_recommendations(
    run_id: str,
    action: Optional[str] = Query(None, pattern="^(increase|decrease|hold)$"),
    customer_id: Optional[int] = None,
    limit: int = Query(50, le=1000),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """Filtered/paginated recommendation lookup -- exactly the kind of
    query that was awkward against a flat CSV and is what Postgres
    indexing (see ix_recommendations_action on the model) is for."""
    if db.get(ModelRun, run_id) is None:
        raise HTTPException(404, f"run_id {run_id!r} not found")

    stmt = select(Recommendation).where(Recommendation.model_run_id == run_id)
    if action:
        stmt = stmt.where(Recommendation.action == action)
    if customer_id is not None:
        stmt = stmt.where(Recommendation.customer_id == customer_id)

    total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = db.execute(stmt.order_by(Recommendation.id).limit(limit).offset(offset)).scalars().all()
    return PaginatedRecommendations(total=total, limit=limit, offset=offset, items=rows)


@router.post("", response_model=JobStatus, status_code=202)
def trigger_run(background_tasks: BackgroundTasks, data_path: str = Query(_DEFAULT_RAW)):
    """Triggers a full `python run_all.py` pipeline run (retrains models,
    re-scores, re-syncs to Postgres) as a background job -- returns
    immediately with a job_id to poll via GET /jobs/{job_id}, since even
    at ~10s (post-vectorization) a synchronous request isn't the right UX
    for "retrain and re-run everything". See src/api/jobs.py for the
    in-memory job-tracking scope limit."""
    job_id = create_job()
    background_tasks.add_task(run_pipeline_job, job_id, data_path)
    return get_job(job_id)


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_run_job(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"job_id {job_id!r} not found")
    return job
