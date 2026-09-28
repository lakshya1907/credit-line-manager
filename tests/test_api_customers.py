import numpy as np
import pandas as pd
import pytest

from tests.conftest import require_db
from tests.test_decision_engine import ConstantEADModel, ConstantPDModel, PD_CALIBRATOR, make_row


@pytest.fixture
def scoring_client(api_client):
    """Swaps app.state.ml for small, fast, deterministic fake models (the
    same fakes decision_engine's own tests use), instead of the real
    30k-row feature matrix + real XGBoost models -- fast and portable,
    while still exercising the actual endpoint logic end to end. Restores
    the real state afterward."""
    original = api_client.app.state.ml

    X = pd.DataFrame([
        make_row(10000.0, customer_id=1),
        make_row(20000.0, customer_id=2),
    ]).set_index("customer_id", drop=False)

    api_client.app.state.ml = {
        "models_loaded": True,
        "pd_model": ConstantPDModel(0.01),
        "pd_calibrator": PD_CALIBRATOR,
        "ead_model": ConstantEADModel(2000.0),
        "X": X,
    }
    yield api_client
    api_client.app.state.ml = original


def test_score_unknown_customer_404s(scoring_client):
    r = scoring_client.post("/customers/999999/score", json={})
    assert r.status_code == 404


def test_score_with_no_body_runs_best_candidate_search(scoring_client):
    r = scoring_client.post("/customers/1/score", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == 1
    assert body["current_limit"] == 10000.0
    # pd_value=0.01 is comfortably profitable to increase (see
    # test_decision_engine.py's identical fixture), so best-candidate
    # search should recommend increasing.
    assert body["action"] == "increase"
    assert body["evaluated_limit"] > body["current_limit"]


def test_score_with_new_limit_evaluates_that_exact_limit(scoring_client):
    r = scoring_client.post("/customers/1/score", json={"new_limit": 15000.0})
    assert r.status_code == 200
    body = r.json()
    assert body["evaluated_limit"] == 15000.0
    assert body["action"] == "increase"
    assert body["guardrail_blocked"] is False


def test_score_rejects_non_positive_new_limit(scoring_client):
    r = scoring_client.post("/customers/1/score", json={"new_limit": -100.0})
    assert r.status_code == 422


def test_score_reports_guardrail_blocked_for_a_risky_new_limit(api_client):
    from src.config import PD_INCREASE_MAX

    original = api_client.app.state.ml
    X = pd.DataFrame([make_row(10000.0, customer_id=1)]).set_index("customer_id", drop=False)
    api_client.app.state.ml = {
        "models_loaded": True,
        "pd_model": ConstantPDModel(PD_INCREASE_MAX + 0.05),  # always "too risky to increase"
        "pd_calibrator": PD_CALIBRATOR,
        "ead_model": ConstantEADModel(2000.0),
        "X": X,
    }
    try:
        r = api_client.post("/customers/1/score", json={"new_limit": 15000.0})  # an increase
        assert r.status_code == 200
        body = r.json()
        assert body["guardrail_blocked"] is True
    finally:
        api_client.app.state.ml = original


def test_score_when_models_not_loaded_returns_503(api_client):
    original = api_client.app.state.ml
    api_client.app.state.ml = {"models_loaded": False}
    try:
        r = api_client.post("/customers/1/score", json={})
        assert r.status_code == 503
    finally:
        api_client.app.state.ml = original


def test_customer_history_404s_for_unknown_customer(api_client, db_available):
    require_db(db_available)
    r = api_client.get("/customers/999999999/history")
    assert r.status_code == 404


def test_customer_history_returns_entries_oldest_first(api_client, db_available):
    require_db(db_available)
    runs = api_client.get("/runs", params={"limit": 1}).json()
    if not runs:
        pytest.skip("no synced runs to test against")
    recs = api_client.get(f"/runs/{runs[0]['run_id']}/recommendations", params={"limit": 1}).json()
    if not recs["items"]:
        pytest.skip("no recommendations to test against")
    customer_id = recs["items"][0]["customer_id"]

    r = api_client.get(f"/customers/{customer_id}/history")
    assert r.status_code == 200
    body = r.json()
    assert len(body) >= 1
    started_ats = [row["started_at"] for row in body]
    assert started_ats == sorted(started_ats)
