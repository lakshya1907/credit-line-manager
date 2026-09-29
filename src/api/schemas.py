"""
src/api/schemas.py
────────────────────
Pydantic response/request models. The *_Out models mirror src/db/models.py
one-for-one (via from_attributes) -- deliberately no generic
"whatever's in the table" serialization, so the API's contract is explicit
and doesn't silently change if a DB column is added/renamed.
"""

from __future__ import annotations

import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    database: str
    models_loaded: bool


class PortfolioRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    policy_name: str
    el_budget: float
    ead_budget: float
    pd_increase_max: float
    pd_decrease_min: float
    used_el: float
    used_ead: float
    n_increase_applied: int
    n_decrease: int
    n_hold: int
    total_ep_uplift: float


class StressTestResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    pd_shock: str
    n_increase: int
    n_decrease: int
    n_hold: int
    total_ep_uplift: float
    el_used: float
    el_budget_pct: float


class SegmentMetricOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    segment: str
    segment_value: str
    n_customers: int
    increase_rate: float
    decrease_rate: float
    hold_rate: float
    avg_pd_current: float
    avg_ep_uplift: float
    total_ep_uplift: float


class BacktestResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    window_months: int
    n_features: int
    val_roc_auc: float
    val_pr_auc: float


class FairnessCheckOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    segment: str
    min_max_approval_ratio: float
    flag: str


class ExposureSummaryOut(BaseModel):
    """Portfolio-wide sums across every customer's recommendation for a
    run (not just approved increases, unlike PortfolioRunOut's used_ead) --
    answers "how does total book exposure change under the recommended
    plan", which nothing else in the API currently does."""

    total_current_limit: float
    total_recommended_limit: float
    total_current_ead: float
    total_recommended_ead: float


class ModelRunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    started_at: datetime.datetime
    finished_at: datetime.datetime
    wall_time_seconds: float
    git_commit: Optional[str] = None
    pd_roc_auc: float
    pd_pr_auc: float
    ead_mae: float


class ModelRunDetail(ModelRunSummary):
    config: dict
    exposure_summary: ExposureSummaryOut
    portfolio_runs: list[PortfolioRunOut]
    stress_test_results: list[StressTestResultOut]
    segment_metrics: list[SegmentMetricOut]
    backtest_results: list[BacktestResultOut]
    fairness_checks: list[FairnessCheckOut]


class RecommendationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    customer_id: int
    current_limit: float
    recommended_limit: float
    action: str
    pd_current: float
    pd_recommended: float
    ead_current: float
    ead_recommended: float
    ep_uplift: float
    el_uplift_proxy: float
    ead_uplift: float
    reason_codes: Optional[str] = None


class PaginatedRecommendations(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[RecommendationOut]


class CustomerHistoryEntry(BaseModel):
    run_id: str
    started_at: datetime.datetime
    action: str
    current_limit: float
    recommended_limit: float
    pd_current: float
    pd_recommended: float
    ep_uplift: float


class ScoreRequest(BaseModel):
    new_limit: Optional[float] = None


class ScoreResponse(BaseModel):
    customer_id: int
    current_limit: float
    evaluated_limit: float
    action: str
    guardrail_blocked: bool
    pd_current: float
    pd_evaluated: float
    ead_current: float
    ead_evaluated: float
    ep_current: float
    ep_evaluated: float
    ep_uplift: float


class JobStatus(BaseModel):
    job_id: str
    status: str
    run_id: Optional[str] = None
    error: Optional[str] = None
    started_at: datetime.datetime
    finished_at: Optional[datetime.datetime] = None
