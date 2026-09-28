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
