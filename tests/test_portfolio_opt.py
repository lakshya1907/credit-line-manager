import pandas as pd
import pytest

from src.portfolio_opt import portfolio_select


def make_rec_df(rows):
    """rows: list of dicts with action, ep_uplift, el_uplift_proxy, ead_uplift."""
    defaults = dict(
        current_limit=10000.0,
        recommended_limit=10000.0,
        pd_current=0.05,
        pd_recommended=0.05,
        ead_current=1000.0,
        ead_recommended=1000.0,
        ep_current=0.0,
        ep_recommended=0.0,
    )
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_decreases_and_holds_always_pass_through_unaffected():
    df = make_rec_df([
        dict(action="decrease", ep_uplift=10.0, el_uplift_proxy=-500.0, ead_uplift=-2000.0),
        dict(action="hold", ep_uplift=0.0, el_uplift_proxy=0.0, ead_uplift=0.0),
    ])
    out, summary = portfolio_select(df, el_budget=0.0, ead_budget=0.0)
    assert (out["action"] == df["action"]).all()
    assert summary["n_decrease"] == 1
    assert summary["n_hold"] == 1


def test_increase_approved_when_within_budget():
    df = make_rec_df([
        dict(action="increase", ep_uplift=100.0, el_uplift_proxy=50.0, ead_uplift=200.0),
    ])
    out, summary = portfolio_select(df, el_budget=1000.0, ead_budget=1000.0)
    assert out.iloc[0]["action"] == "increase"
    assert summary["n_increase_applied"] == 1
    assert summary["used_el"] == pytest.approx(50.0)
    assert summary["used_ead"] == pytest.approx(200.0)


def test_increase_rejected_when_it_would_exceed_budget():
    df = make_rec_df([
        dict(action="increase", ep_uplift=100.0, el_uplift_proxy=5000.0, ead_uplift=200.0),
    ])
    out, summary = portfolio_select(df, el_budget=1000.0, ead_budget=1000.0)
    row = out.iloc[0]
    assert row["action"] == "hold"
    assert row["recommended_limit"] == row["current_limit"]
    assert summary["n_increase_applied"] == 0
    assert summary["used_el"] == 0.0


def test_negative_el_uplift_is_not_floored_to_zero():
    # Regression test: portfolio_select used to do
    # `d_el = max(el_uplift_proxy, 0.0)`, silently discarding any customer
    # whose net incremental EL was negative and letting them consume zero
    # budget regardless of how large that negative delta was. used_el must
    # reflect the true signed sum.
    df = make_rec_df([
        dict(action="increase", ep_uplift=10.0, el_uplift_proxy=-300.0, ead_uplift=100.0),
        dict(action="increase", ep_uplift=10.0, el_uplift_proxy=200.0, ead_uplift=100.0),
    ])
    out, summary = portfolio_select(df, el_budget=1000.0, ead_budget=1000.0)
    assert summary["used_el"] == pytest.approx(-300.0 + 200.0)
    assert summary["n_increase_applied"] == 2


def test_negative_el_uplift_increase_is_never_blocked_by_el_budget():
    # A customer whose el_uplift_proxy is negative should never itself be
    # the reason the EL budget is exceeded.
    df = make_rec_df([
        dict(action="increase", ep_uplift=5.0, el_uplift_proxy=-100.0, ead_uplift=50.0),
    ])
    out, summary = portfolio_select(df, el_budget=0.0, ead_budget=1000.0)
    assert out.iloc[0]["action"] == "increase"


def test_increases_are_approved_in_descending_roi_order():
    # roi = ep_uplift / (|el_uplift_proxy| + eps); budget only fits one.
    df = make_rec_df([
        dict(action="increase", ep_uplift=10.0, el_uplift_proxy=100.0, ead_uplift=50.0),   # roi 0.1
        dict(action="increase", ep_uplift=50.0, el_uplift_proxy=100.0, ead_uplift=50.0),   # roi 0.5 (best)
        dict(action="increase", ep_uplift=20.0, el_uplift_proxy=100.0, ead_uplift=50.0),   # roi 0.2
    ])
    out, summary = portfolio_select(df, el_budget=100.0, ead_budget=1000.0)
    approved = out[out["action"] == "increase"]
    assert len(approved) == 1
    assert approved.iloc[0]["ep_uplift"] == 50.0


def test_summary_total_ep_uplift_sums_all_rows_including_rejected():
    df = make_rec_df([
        dict(action="increase", ep_uplift=100.0, el_uplift_proxy=5000.0, ead_uplift=200.0),  # rejected
        dict(action="decrease", ep_uplift=10.0, el_uplift_proxy=-50.0, ead_uplift=-100.0),
    ])
    out, summary = portfolio_select(df, el_budget=0.0, ead_budget=0.0)
    # total_ep_uplift sums ep_uplift across the *output* rows regardless of
    # whether an increase was rejected back to hold.
    assert summary["total_ep_uplift"] == pytest.approx(100.0 + 10.0)


def test_output_preserves_original_row_order():
    df = make_rec_df([
        dict(action="hold", ep_uplift=0.0, el_uplift_proxy=0.0, ead_uplift=0.0),
        dict(action="increase", ep_uplift=1.0, el_uplift_proxy=1.0, ead_uplift=1.0),
        dict(action="decrease", ep_uplift=2.0, el_uplift_proxy=-1.0, ead_uplift=-1.0),
    ])
    out, _ = portfolio_select(df, el_budget=1000.0, ead_budget=1000.0)
    assert list(out.index) == list(df.index)
