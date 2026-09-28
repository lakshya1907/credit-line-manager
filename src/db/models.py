"""
src/db/models.py
──────────────────
SQLAlchemy schema for run history.

Every table here mirrors an artifact run_all.py / run_analytics.py already
produce (see CLAUDE.md's "Run history" and "Analytics" sections) -- this is
a migration target for files that already exist, not a speculative schema:

    model_runs         <- data/processed/runs/<run_id>/run_manifest.json
    portfolio_runs      <- portfolio_opt.portfolio_select()'s summary dict,
                           one row per named policy (run_analytics.py's
                           policy_compare.py can produce several per
                           model_run against the same trained models)
    recommendations     <- recommendations_raw.csv + audit_log.csv's
                           per-row content (their run-level metadata --
                           model versions, policy budgets -- lives on
                           ModelRun/PortfolioRun here instead of being
                           repeated on every customer row)
    stress_test_results <- stress_test_results.csv
    segment_metrics      <- data/processed/analytics/segment_summary.csv
    backtest_results     <- data/processed/analytics/backtest_window_months.csv
    fairness_checks       <- data/processed/analytics/fairness_check.csv

Nothing here is read by anything yet except sync_run_to_db.py (which
writes) and tests -- that's deliberate; a FastAPI read layer is a later
stage, not this one. See sync_run_to_db.py for how a run's files become
these rows.
"""

from __future__ import annotations

import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class ModelRun(Base):
    """One row per `python run_all.py` execution."""
    __tablename__ = "model_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    started_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    wall_time_seconds: Mapped[float] = mapped_column(Float)
    git_commit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    raw_data_path: Mapped[str] = mapped_column(String(512))
    config: Mapped[dict] = mapped_column(JSON)

    pd_roc_auc: Mapped[float] = mapped_column(Float)
    pd_pr_auc: Mapped[float] = mapped_column(Float)
    pd_brier_raw: Mapped[float] = mapped_column(Float)
    pd_brier_calibrated: Mapped[float] = mapped_column(Float)
    ead_mae: Mapped[float] = mapped_column(Float)

    portfolio_runs: Mapped[list["PortfolioRun"]] = relationship(back_populates="model_run", cascade="all, delete-orphan")
    recommendations: Mapped[list["Recommendation"]] = relationship(back_populates="model_run", cascade="all, delete-orphan")
    stress_test_results: Mapped[list["StressTestResult"]] = relationship(back_populates="model_run", cascade="all, delete-orphan")
    segment_metrics: Mapped[list["SegmentMetric"]] = relationship(back_populates="model_run", cascade="all, delete-orphan")
    backtest_results: Mapped[list["BacktestResult"]] = relationship(back_populates="model_run", cascade="all, delete-orphan")
    fairness_checks: Mapped[list["FairnessCheck"]] = relationship(back_populates="model_run", cascade="all, delete-orphan")


class PortfolioRun(Base):
    """One row per portfolio_select() outcome for a model_run. Usually one
    ("default"), but run_analytics.py's policy_compare.py can produce
    several per model_run (different budgets/guardrails against the same
    trained models) -- policy_name distinguishes them."""
    __tablename__ = "portfolio_runs"
    __table_args__ = (UniqueConstraint("model_run_id", "policy_name", name="uq_portfolio_run_policy"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_run_id: Mapped[str] = mapped_column(ForeignKey("model_runs.run_id", ondelete="CASCADE"), index=True)
    policy_name: Mapped[str] = mapped_column(String(64), default="default")

    el_budget: Mapped[float] = mapped_column(Float)
    ead_budget: Mapped[float] = mapped_column(Float)
    pd_increase_max: Mapped[float] = mapped_column(Float)
    pd_decrease_min: Mapped[float] = mapped_column(Float)

    used_el: Mapped[float] = mapped_column(Float)
    used_ead: Mapped[float] = mapped_column(Float)
    n_increase_applied: Mapped[int] = mapped_column(Integer)
    n_decrease: Mapped[int] = mapped_column(Integer)
    n_hold: Mapped[int] = mapped_column(Integer)
    total_ep_uplift: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    model_run: Mapped["ModelRun"] = relationship(back_populates="portfolio_runs")


class Recommendation(Base):
    """One row per (model_run, customer): the decision engine's raw
    (pre-portfolio-budget) recommendation plus explainability. The
    portfolio-approved action can differ (see the portfolio_opt.py fix
    documented in CLAUDE.md/git log) -- that's PortfolioRun's job to
    capture when a per-policy recommendations table is needed; this table
    is the engine's unconstrained output, matching what
    recommendations_raw.csv / audit_log.csv hold today."""
    __tablename__ = "recommendations"
    __table_args__ = (
        UniqueConstraint("model_run_id", "customer_id", name="uq_recommendation_customer"),
        Index("ix_recommendations_action", "model_run_id", "action"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_run_id: Mapped[str] = mapped_column(ForeignKey("model_runs.run_id", ondelete="CASCADE"), index=True)
    customer_id: Mapped[int] = mapped_column(Integer, index=True)

    current_limit: Mapped[float] = mapped_column(Float)
    recommended_limit: Mapped[float] = mapped_column(Float)
    action: Mapped[str] = mapped_column(String(16))
    pd_current: Mapped[float] = mapped_column(Float)
    pd_recommended: Mapped[float] = mapped_column(Float)
    ead_current: Mapped[float] = mapped_column(Float)
    ead_recommended: Mapped[float] = mapped_column(Float)
    ep_current: Mapped[float] = mapped_column(Float)
    ep_recommended: Mapped[float] = mapped_column(Float)
    ep_uplift: Mapped[float] = mapped_column(Float)
    el_uplift_proxy: Mapped[float] = mapped_column(Float)
    ead_uplift: Mapped[float] = mapped_column(Float)

    top_features: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason_codes: Mapped[str | None] = mapped_column(Text, nullable=True)
    top_shap_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)

    model_run: Mapped["ModelRun"] = relationship(back_populates="recommendations")


class StressTestResult(Base):
    __tablename__ = "stress_test_results"
    __table_args__ = (UniqueConstraint("model_run_id", "pd_shock", name="uq_stress_test_shock"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_run_id: Mapped[str] = mapped_column(ForeignKey("model_runs.run_id", ondelete="CASCADE"), index=True)
    pd_shock: Mapped[str] = mapped_column(String(16))
    n_increase: Mapped[int] = mapped_column(Integer)
    n_decrease: Mapped[int] = mapped_column(Integer)
    n_hold: Mapped[int] = mapped_column(Integer)
    total_ep_uplift: Mapped[float] = mapped_column(Float)
    el_used: Mapped[float] = mapped_column(Float)
    el_budget_pct: Mapped[float] = mapped_column(Float)

    model_run: Mapped["ModelRun"] = relationship(back_populates="stress_test_results")


class SegmentMetric(Base):
    __tablename__ = "segment_metrics"
    __table_args__ = (UniqueConstraint("model_run_id", "segment", "segment_value", name="uq_segment_metric"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_run_id: Mapped[str] = mapped_column(ForeignKey("model_runs.run_id", ondelete="CASCADE"), index=True)
    segment: Mapped[str] = mapped_column(String(32))
    segment_value: Mapped[str] = mapped_column(String(32))
    n_customers: Mapped[int] = mapped_column(Integer)
    increase_rate: Mapped[float] = mapped_column(Float)
    decrease_rate: Mapped[float] = mapped_column(Float)
    hold_rate: Mapped[float] = mapped_column(Float)
    avg_pd_current: Mapped[float] = mapped_column(Float)
    avg_ep_uplift: Mapped[float] = mapped_column(Float)
    total_ep_uplift: Mapped[float] = mapped_column(Float)

    model_run: Mapped["ModelRun"] = relationship(back_populates="segment_metrics")


class BacktestResult(Base):
    __tablename__ = "backtest_results"
    __table_args__ = (UniqueConstraint("model_run_id", "window_months", name="uq_backtest_window"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_run_id: Mapped[str] = mapped_column(ForeignKey("model_runs.run_id", ondelete="CASCADE"), index=True)
    window_months: Mapped[int] = mapped_column(Integer)
    n_features: Mapped[int] = mapped_column(Integer)
    val_roc_auc: Mapped[float] = mapped_column(Float)
    val_pr_auc: Mapped[float] = mapped_column(Float)

    model_run: Mapped["ModelRun"] = relationship(back_populates="backtest_results")


class FairnessCheck(Base):
    __tablename__ = "fairness_checks"
    __table_args__ = (UniqueConstraint("model_run_id", "segment", name="uq_fairness_segment"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_run_id: Mapped[str] = mapped_column(ForeignKey("model_runs.run_id", ondelete="CASCADE"), index=True)
    segment: Mapped[str] = mapped_column(String(32))
    min_max_approval_ratio: Mapped[float] = mapped_column(Float)
    flag: Mapped[str] = mapped_column(String(16))

    model_run: Mapped["ModelRun"] = relationship(back_populates="fairness_checks")
