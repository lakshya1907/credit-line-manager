import pytest

from tests.conftest import require_db


def test_list_runs(api_client, db_available):
    require_db(db_available)
    r = api_client.get("/runs", params={"limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    if body:
        assert "run_id" in body[0]
        assert "pd_roc_auc" in body[0]


def test_list_runs_ordered_newest_first(api_client, db_available):
    require_db(db_available)
    r = api_client.get("/runs", params={"limit": 50})
    started_ats = [row["started_at"] for row in r.json()]
    assert started_ats == sorted(started_ats, reverse=True)


def test_get_run_detail_includes_nested_collections(api_client, db_available):
    require_db(db_available)
    runs = api_client.get("/runs", params={"limit": 1}).json()
    if not runs:
        pytest.skip("no synced runs to test against")
    run_id = runs[0]["run_id"]

    r = api_client.get(f"/runs/{run_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == run_id
    for key in ["portfolio_runs", "stress_test_results", "segment_metrics", "backtest_results", "fairness_checks"]:
        assert key in body
        assert isinstance(body[key], list)


def test_get_run_detail_exposure_summary_is_portfolio_wide(api_client, db_available):
    # exposure_summary sums every recommendation for the run, not just
    # approved increases -- so it must be >= any single PortfolioRun's
    # used_ead (which only totals approved increases).
    require_db(db_available)
    runs = api_client.get("/runs", params={"limit": 1}).json()
    if not runs:
        pytest.skip("no synced runs to test against")
    run_id = runs[0]["run_id"]

    body = api_client.get(f"/runs/{run_id}").json()
    summary = body["exposure_summary"]
    for key in ["total_current_limit", "total_recommended_limit", "total_current_ead", "total_recommended_ead"]:
        assert key in summary
        assert summary[key] >= 0

    default_policy = next((p for p in body["portfolio_runs"] if p["policy_name"] == "default"), None)
    if default_policy:
        assert summary["total_recommended_ead"] >= default_policy["used_ead"]


def test_get_run_detail_404_for_unknown_run(api_client, db_available):
    require_db(db_available)
    r = api_client.get("/runs/this-run-id-does-not-exist")
    assert r.status_code == 404


def test_get_recommendations_filtered_by_action(api_client, db_available):
    require_db(db_available)
    runs = api_client.get("/runs", params={"limit": 1}).json()
    if not runs:
        pytest.skip("no synced runs to test against")
    run_id = runs[0]["run_id"]

    r = api_client.get(f"/runs/{run_id}/recommendations", params={"action": "increase", "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["limit"] == 5
    assert all(row["action"] == "increase" for row in body["items"])
    assert body["total"] >= len(body["items"])


def test_get_recommendations_404_for_unknown_run(api_client, db_available):
    require_db(db_available)
    r = api_client.get("/runs/this-run-id-does-not-exist/recommendations")
    assert r.status_code == 404


def test_get_recommendations_rejects_invalid_action(api_client, db_available):
    require_db(db_available)
    runs = api_client.get("/runs", params={"limit": 1}).json()
    if not runs:
        pytest.skip("no synced runs to test against")
    run_id = runs[0]["run_id"]

    r = api_client.get(f"/runs/{run_id}/recommendations", params={"action": "not_a_real_action"})
    assert r.status_code == 422


def test_get_job_404_for_unknown_job(api_client):
    r = api_client.get("/runs/jobs/not-a-real-job-id")
    assert r.status_code == 404


def test_get_distributions_buckets_sum_to_total_customers(api_client, db_available):
    require_db(db_available)
    runs = api_client.get("/runs", params={"limit": 1}).json()
    if not runs:
        pytest.skip("no synced runs to test against")
    run_id = runs[0]["run_id"]

    recs_total = api_client.get(f"/runs/{run_id}/recommendations", params={"limit": 1}).json()["total"]
    body = api_client.get(f"/runs/{run_id}/distributions").json()

    for key in ["risk_distribution", "utilization_distribution", "limit_change_distribution"]:
        assert key in body
        assert sum(b["count"] for b in body[key]) == recs_total


def test_get_distributions_404_for_unknown_run(api_client, db_available):
    require_db(db_available)
    r = api_client.get("/runs/this-run-id-does-not-exist/distributions")
    assert r.status_code == 404


def test_get_customer_sample_respects_n_and_shape(api_client, db_available):
    require_db(db_available)
    runs = api_client.get("/runs", params={"limit": 1}).json()
    if not runs:
        pytest.skip("no synced runs to test against")
    run_id = runs[0]["run_id"]

    r = api_client.get(f"/runs/{run_id}/customer-sample", params={"n": 10})
    assert r.status_code == 200
    body = r.json()
    assert len(body) <= 10
    if body:
        for key in ["customer_id", "pd_current", "current_limit", "recommended_limit", "ep_uplift", "action"]:
            assert key in body[0]


def test_get_customer_sample_404_for_unknown_run(api_client, db_available):
    require_db(db_available)
    r = api_client.get("/runs/this-run-id-does-not-exist/customer-sample")
    assert r.status_code == 404
