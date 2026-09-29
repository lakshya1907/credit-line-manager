from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session, selectinload

from src.api.auth import require_api_key
from src.api.deps import get_db
from src.api.jobs import create_job, get_job, run_pipeline_job
from src.api.schemas import (
    CustomerSamplePoint, DistributionBucket, ExposureSummaryOut, JobStatus, ModelRunDetail,
    ModelRunSummary, PaginatedRecommendations, PortfolioDistributionsOut,
)
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

    # Portfolio-wide exposure sums, over every customer -- deliberately
    # not the same thing as PortfolioRunOut.used_ead (which only totals
    # approved increases). Set as a plain instance attribute (not a mapped
    # column) so ModelRunDetail's from_attributes pickup works without a
    # separate response-building function for one extra field.
    totals = db.execute(
        select(
            func.coalesce(func.sum(Recommendation.current_limit), 0.0),
            func.coalesce(func.sum(Recommendation.recommended_limit), 0.0),
            func.coalesce(func.sum(Recommendation.ead_current), 0.0),
            func.coalesce(func.sum(Recommendation.ead_recommended), 0.0),
        ).where(Recommendation.model_run_id == run_id)
    ).one()
    run.exposure_summary = ExposureSummaryOut(
        total_current_limit=totals[0], total_recommended_limit=totals[1],
        total_current_ead=totals[2], total_recommended_ead=totals[3],
    )
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


# ── bucket definitions, shared by the distributions endpoint's queries ──
_RISK_BUCKETS = [
    (Recommendation.pd_current < 0.05, "<5%"),
    (Recommendation.pd_current < 0.10, "5-10%"),
    (Recommendation.pd_current < 0.15, "10-15%"),
    (Recommendation.pd_current < 0.20, "15-20%"),
    (Recommendation.pd_current < 0.30, "20-30%"),
]
_RISK_BUCKET_ORDER = ["<5%", "5-10%", "10-15%", "15-20%", "20-30%", "30%+"]

_UTIL_EXPR = Recommendation.ead_current / func.nullif(Recommendation.current_limit, 0.0)
_UTIL_BUCKETS = [
    (_UTIL_EXPR < 0.10, "<10%"),
    (_UTIL_EXPR < 0.30, "10-30%"),
    (_UTIL_EXPR < 0.50, "30-50%"),
    (_UTIL_EXPR < 0.70, "50-70%"),
    (_UTIL_EXPR < 0.90, "70-90%"),
]
_UTIL_BUCKET_ORDER = ["<10%", "10-30%", "30-50%", "50-70%", "70-90%", "90%+"]

_CHANGE_PCT_EXPR = (
    (Recommendation.recommended_limit - Recommendation.current_limit)
    / func.nullif(Recommendation.current_limit, 0.0)
)
# Every candidate limit in LIMIT_MULTIPLIERS (src/config.py) is >=10% away
# from the current limit -- 0.8/0.9/1.1/1.25/1.5x -- so a finer-grained
# "0-10%" bucket would always be empty by construction, not because
# nothing happened to land there; three buckets, not five.
_CHANGE_BUCKETS = [
    (_CHANGE_PCT_EXPR <= -0.001, "Decrease"),
    (_CHANGE_PCT_EXPR < 0.001, "No change"),
]
_CHANGE_BUCKET_ORDER = ["Decrease", "No change", "Increase"]


def _bucket_counts(db: Session, run_id: str, buckets: list, fallback_label: str, order: list[str]) -> list[DistributionBucket]:
    label_expr = case(*buckets, else_=fallback_label)
    rows = db.execute(
        select(label_expr.label("bucket"), func.count())
        .where(Recommendation.model_run_id == run_id)
        .group_by(label_expr)
    ).all()
    counts = {bucket: count for bucket, count in rows}
    return [DistributionBucket(bucket=b, count=counts.get(b, 0)) for b in order]


@router.get("/{run_id}/distributions", response_model=PortfolioDistributionsOut)
def get_distributions(run_id: str, db: Session = Depends(get_db)):
    """Bucketed counts (SQL GROUP BY over the full recommendation set, not
    a sample) for the Overview/Analytics pages' histograms. utilization is
    a derived field (ead_current / current_limit, computed in SQL, not
    stored) -- there is no persisted utilization column, this is the same
    definition used throughout the pipeline (see src/features.py)."""
    if db.get(ModelRun, run_id) is None:
        raise HTTPException(404, f"run_id {run_id!r} not found")

    return PortfolioDistributionsOut(
        risk_distribution=_bucket_counts(db, run_id, _RISK_BUCKETS, "30%+", _RISK_BUCKET_ORDER),
        utilization_distribution=_bucket_counts(db, run_id, _UTIL_BUCKETS, "90%+", _UTIL_BUCKET_ORDER),
        limit_change_distribution=_bucket_counts(db, run_id, _CHANGE_BUCKETS, "Increase", _CHANGE_BUCKET_ORDER),
    )


@router.get("/{run_id}/customer-sample", response_model=list[CustomerSamplePoint])
def get_customer_sample(run_id: str, n: int = Query(1500, le=5000), db: Session = Depends(get_db)):
    """A bounded random sample for scatter plots (PD vs. utilization,
    limit vs. recommended limit, PD vs. EP) -- the alternative is shipping
    all ~30k rows to the browser for a chart that can't usefully render
    that many points anyway. Not used for the bucketed distributions
    above, which aggregate the full table in SQL instead for accuracy."""
    if db.get(ModelRun, run_id) is None:
        raise HTTPException(404, f"run_id {run_id!r} not found")

    utilization_expr = (Recommendation.ead_current / func.nullif(Recommendation.current_limit, 0.0)).label("utilization")
    stmt = (
        select(
            Recommendation.customer_id, Recommendation.pd_current, utilization_expr,
            Recommendation.current_limit, Recommendation.recommended_limit,
            Recommendation.ep_uplift, Recommendation.action,
        )
        .where(Recommendation.model_run_id == run_id)
        .order_by(func.random())
        .limit(n)
    )
    rows = db.execute(stmt).all()
    return [
        CustomerSamplePoint(
            customer_id=r.customer_id, pd_current=r.pd_current, utilization=r.utilization,
            current_limit=r.current_limit, recommended_limit=r.recommended_limit,
            ep_uplift=r.ep_uplift, action=r.action,
        )
        for r in rows
    ]


@router.post("", response_model=JobStatus, status_code=202, dependencies=[Depends(require_api_key)])
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
