import pandas as pd
import pytest

from src.analytics.policy_compare import compare_policies
from tests.test_decision_engine import ConstantPDModel, ConstantEADModel, PD_CALIBRATOR, make_row


def test_compare_policies_returns_one_row_per_named_policy():
    X = pd.DataFrame([make_row(10000.0), make_row(20000.0)])
    pd_model = ConstantPDModel(0.01)  # comfortably profitable to increase, see test_decision_engine
    ead_model = ConstantEADModel(2000.0)

    policies = {
        "baseline": {},
        "tight_guardrail": {"pd_increase_max": 0.0},  # blocks every increase
    }
    result = compare_policies(X, pd_model, PD_CALIBRATOR, ead_model, policies)

    assert list(result["policy"]) == ["baseline", "tight_guardrail"]
    assert result[result["policy"] == "baseline"].iloc[0]["n_increase_applied"] > 0
    assert result[result["policy"] == "tight_guardrail"].iloc[0]["n_increase_applied"] == 0


def test_compare_policies_respects_budget_override():
    X = pd.DataFrame([make_row(10000.0)])
    pd_model = ConstantPDModel(0.01)
    ead_model = ConstantEADModel(2000.0)

    policies = {
        "generous_budget": {"ead_budget": 1e12},
        "starved_budget": {"ead_budget": 0.0},
    }
    result = compare_policies(X, pd_model, PD_CALIBRATOR, ead_model, policies)

    generous = result[result["policy"] == "generous_budget"].iloc[0]
    starved = result[result["policy"] == "starved_budget"].iloc[0]
    assert generous["n_increase_applied"] >= starved["n_increase_applied"]
    assert starved["n_increase_applied"] == 0


def test_compare_policies_unspecified_overrides_fall_back_to_config_defaults():
    from src.config import EL_BUDGET, PD_DECREASE_MIN
    X = pd.DataFrame([make_row(10000.0)])
    pd_model = ConstantPDModel(0.01)
    ead_model = ConstantEADModel(2000.0)

    result = compare_policies(X, pd_model, PD_CALIBRATOR, ead_model, {"only_ead_override": {"ead_budget": 999.0}})
    row = result.iloc[0]
    assert row["el_budget"] == EL_BUDGET
    assert row["pd_decrease_min"] == PD_DECREASE_MIN
    assert row["ead_budget"] == 999.0
