import pandas as pd
import pytest

from src.stress_test import run_stress_scenario, apply_pd_shock, apply_ead_shock
from tests.test_decision_engine import (
    ConstantPDModel, ConstantEADModel, PD_CALIBRATOR, make_row,
)


def test_run_stress_scenario_forwards_shock_into_decision_engine():
    # run_stress_scenario is a thin wrapper around recommend_limits(...,
    # pd_shock=...); confirm the shock actually changes the outcome the
    # same way a direct recommend_limits(pd_shock=...) call would (this is
    # the "real re-simulation" path used by run_all.py's batch pipeline).
    L0 = 10000.0
    pd_value = 0.01
    X = pd.DataFrame([make_row(L0)])
    pd_model = ConstantPDModel(pd_value)
    ead_model = ConstantEADModel(2000.0)

    baseline = run_stress_scenario(X, pd_model, PD_CALIBRATOR, ead_model, pd_shock=0.0)
    shocked = run_stress_scenario(X, pd_model, PD_CALIBRATOR, ead_model, pd_shock=8.0)

    assert baseline.iloc[0]["action"] == "increase"
    assert shocked.iloc[0]["action"] != "increase"


def make_fast_rec_df():
    return pd.DataFrame([{
        "action": "increase",
        "pd_current": 0.05,
        "pd_recommended": 0.04,
        "ead_current": 1000.0,
        "ead_recommended": 1500.0,
    }])


def test_apply_pd_shock_rescales_but_never_changes_the_action():
    df = make_fast_rec_df()
    out = apply_pd_shock(df.copy(), shock=0.5)
    assert out.iloc[0]["action"] == df.iloc[0]["action"]
    assert out.iloc[0]["pd_current"] == pytest.approx(0.05 * 1.5)
    assert out.iloc[0]["pd_recommended"] == pytest.approx(0.04 * 1.5)


def test_apply_pd_shock_clips_to_one():
    df = make_fast_rec_df()
    out = apply_pd_shock(df.copy(), shock=50.0)
    assert out.iloc[0]["pd_current"] == pytest.approx(1.0)
    assert out.iloc[0]["pd_recommended"] == pytest.approx(1.0)


def test_apply_ead_shock_rescales_ead_and_recomputes_el_uplift_proxy():
    df = make_fast_rec_df()
    out = apply_ead_shock(df.copy(), shock=0.2)
    assert out.iloc[0]["ead_current"] == pytest.approx(1000.0 * 1.2)
    assert out.iloc[0]["ead_recommended"] == pytest.approx(1500.0 * 1.2)
    expected_el = out.iloc[0]["pd_recommended"] * out.iloc[0]["ead_recommended"] - \
        out.iloc[0]["pd_current"] * out.iloc[0]["ead_current"]
    assert out.iloc[0]["el_uplift_proxy"] == pytest.approx(expected_el)
