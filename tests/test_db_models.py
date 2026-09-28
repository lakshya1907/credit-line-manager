"""
Schema/relationship tests for src/db/models.py, run against an in-memory
SQLite engine so they're portable (no live Postgres required to run the
suite). The actual DDL (types, FKs, cascades, unique constraints, the
alembic upgrade/downgrade path) was verified by hand against a real local
Postgres -- see CLAUDE.md's Database section -- these tests cover the ORM
layer's behavior (relationships, cascade deletes, constraints), which is
portable across dialects for the simple types this schema uses.
"""

import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, ModelRun, PortfolioRun, Recommendation, StressTestResult


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    with factory() as s:
        yield s


def make_model_run(run_id="run-1"):
    now = datetime.datetime.now(datetime.timezone.utc)
    return ModelRun(
        run_id=run_id, started_at=now, finished_at=now, wall_time_seconds=1.0,
        git_commit="abc123", raw_data_path="data/raw/uci_credit.csv",
        config={"el_budget": 1.0}, pd_roc_auc=0.7, pd_pr_auc=0.5,
        pd_brier_raw=0.2, pd_brier_calibrated=0.1, ead_mae=100.0,
    )


def test_model_run_round_trips(session):
    mr = make_model_run()
    session.add(mr)
    session.commit()

    fetched = session.get(ModelRun, "run-1")
    assert fetched.pd_roc_auc == 0.7
    assert fetched.config == {"el_budget": 1.0}


def test_portfolio_run_relationship(session):
    mr = make_model_run()
    mr.portfolio_runs.append(PortfolioRun(
        policy_name="default", el_budget=1.0, ead_budget=2.0,
        pd_increase_max=0.08, pd_decrease_min=0.2,
        used_el=0.5, used_ead=1.0, n_increase_applied=1, n_decrease=2, n_hold=3,
        total_ep_uplift=10.0,
    ))
    session.add(mr)
    session.commit()

    fetched = session.get(ModelRun, "run-1")
    assert len(fetched.portfolio_runs) == 1
    assert fetched.portfolio_runs[0].policy_name == "default"
    assert fetched.portfolio_runs[0].model_run is fetched


def test_duplicate_policy_name_for_same_run_violates_unique_constraint(session):
    mr = make_model_run()
    mr.portfolio_runs.append(PortfolioRun(
        policy_name="default", el_budget=1.0, ead_budget=2.0, pd_increase_max=0.08,
        pd_decrease_min=0.2, used_el=0.0, used_ead=0.0, n_increase_applied=0,
        n_decrease=0, n_hold=0, total_ep_uplift=0.0,
    ))
    mr.portfolio_runs.append(PortfolioRun(
        policy_name="default", el_budget=1.0, ead_budget=2.0, pd_increase_max=0.08,
        pd_decrease_min=0.2, used_el=0.0, used_ead=0.0, n_increase_applied=0,
        n_decrease=0, n_hold=0, total_ep_uplift=0.0,
    ))
    session.add(mr)
    with pytest.raises(Exception):
        session.commit()


def test_deleting_model_run_cascades_to_children(session):
    mr = make_model_run()
    mr.recommendations.append(Recommendation(
        customer_id=1, current_limit=1000.0, recommended_limit=1000.0, action="hold",
        pd_current=0.1, pd_recommended=0.1, ead_current=100.0, ead_recommended=100.0,
        ep_current=0.0, ep_recommended=0.0, ep_uplift=0.0, el_uplift_proxy=0.0, ead_uplift=0.0,
    ))
    mr.stress_test_results.append(StressTestResult(
        pd_shock="+0%", n_increase=0, n_decrease=0, n_hold=1,
        total_ep_uplift=0.0, el_used=0.0, el_budget_pct=0.0,
    ))
    session.add(mr)
    session.commit()

    session.delete(session.get(ModelRun, "run-1"))
    session.commit()

    assert session.query(Recommendation).count() == 0
    assert session.query(StressTestResult).count() == 0


def test_duplicate_customer_id_for_same_run_violates_unique_constraint(session):
    mr = make_model_run()
    for _ in range(2):
        mr.recommendations.append(Recommendation(
            customer_id=1, current_limit=1000.0, recommended_limit=1000.0, action="hold",
            pd_current=0.1, pd_recommended=0.1, ead_current=100.0, ead_recommended=100.0,
            ep_current=0.0, ep_recommended=0.0, ep_uplift=0.0, el_uplift_proxy=0.0, ead_uplift=0.0,
        ))
    session.add(mr)
    with pytest.raises(Exception):
        session.commit()
