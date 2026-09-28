import datetime
import json
import os

import pandas as pd
import pytest

import run_all


def test_new_run_id_is_sortable_and_unique():
    t = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    run_id = run_all._new_run_id(t)
    assert run_id.startswith("20260102T030405Z_")

    # Same timestamp, called twice, must not collide (random suffix).
    run_id2 = run_all._new_run_id(t)
    assert run_id != run_id2


def test_config_snapshot_includes_the_policy_knobs_that_drive_decisions():
    snapshot = run_all._config_snapshot()
    for key in ["el_budget", "ead_budget", "pd_increase_max", "pd_decrease_min",
                "limit_multipliers", "robust_mode"]:
        assert key in snapshot


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Point run_all's module-level path constants at a tmp dir so
    archive_run() never touches the real data/processed/ tree."""
    proc_dir = tmp_path / "data" / "processed"
    report_dir = tmp_path / "reports"
    proc_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    rec_raw = proc_dir / "recommendations_raw.csv"
    rec_final = proc_dir / "recommendations_final.csv"
    stress = proc_dir / "stress_test_results.csv"
    audit = proc_dir / "audit_log.csv"
    metrics = report_dir / "pipeline_metrics.txt"
    global_imp = report_dir / "shap_global_importance.csv"

    for f in [rec_raw, rec_final, stress, audit, metrics, global_imp]:
        f.write_text("dummy content")

    runs_dir = proc_dir / "runs"
    runs_index = proc_dir / "runs_index.csv"

    monkeypatch.setattr(run_all, "REC_RAW_PATH", str(rec_raw))
    monkeypatch.setattr(run_all, "REC_FINAL_PATH", str(rec_final))
    monkeypatch.setattr(run_all, "STRESS_PATH", str(stress))
    monkeypatch.setattr(run_all, "AUDIT_LOG_PATH", str(audit))
    monkeypatch.setattr(run_all, "METRICS_PATH", str(metrics))
    monkeypatch.setattr(run_all, "GLOBAL_IMP_PATH", str(global_imp))
    monkeypatch.setattr(run_all, "RUNS_DIR", str(runs_dir))
    monkeypatch.setattr(run_all, "RUNS_INDEX_PATH", str(runs_index))

    return {"runs_dir": runs_dir, "runs_index": runs_index}


def make_stress_df():
    return pd.DataFrame([
        {"pd_shock": "+0%", "n_increase": 10, "n_decrease": 5, "n_hold": 2,
         "total_ep_uplift": 100.0, "el_used": -50.0, "el_budget_pct": -1.0},
    ])


def test_archive_run_copies_outputs_writes_manifest_and_appends_index(isolated_paths):
    started = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    finished = started + datetime.timedelta(seconds=5)
    portfolio_summary = {
        "used_el": -1.0, "used_ead": 2.0, "el_budget": 3.0, "ead_budget": 4.0,
        "n_increase_applied": 1, "n_decrease": 2, "n_hold": 3, "total_ep_uplift": 5.0,
    }

    run_all.archive_run(
        "test-run-1", started, finished, "data/raw/uci_credit.csv",
        pd_metrics={"val_roc_auc": 0.7, "val_pr_auc": 0.5},
        brier_raw=0.2, brier_cal=0.1,
        ead_metrics={"val_mae": 100.0},
        portfolio_summary=portfolio_summary,
        stress_df=make_stress_df(),
    )

    run_dir = isolated_paths["runs_dir"] / "test-run-1"
    assert run_dir.is_dir()
    for fname in ["recommendations_raw.csv", "recommendations_final.csv",
                  "stress_test_results.csv", "audit_log.csv",
                  "pipeline_metrics.txt", "shap_global_importance.csv"]:
        assert (run_dir / fname).exists()

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["run_id"] == "test-run-1"
    assert manifest["wall_time_seconds"] == pytest.approx(5.0)
    assert manifest["portfolio_summary"]["n_increase_applied"] == 1
    assert manifest["stress_test"][0]["pd_shock"] == "+0%"

    index_df = pd.read_csv(isolated_paths["runs_index"])
    assert len(index_df) == 1
    assert index_df.iloc[0]["run_id"] == "test-run-1"
    assert index_df.iloc[0]["n_increase_applied"] == 1


def test_archive_run_appends_a_second_row_without_rewriting_the_header(isolated_paths):
    started = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    finished = started + datetime.timedelta(seconds=1)
    portfolio_summary = {
        "used_el": 0.0, "used_ead": 0.0, "el_budget": 1.0, "ead_budget": 1.0,
        "n_increase_applied": 0, "n_decrease": 0, "n_hold": 0, "total_ep_uplift": 0.0,
    }
    common_kwargs = dict(
        raw_path="data/raw/uci_credit.csv",
        pd_metrics={"val_roc_auc": 0.7, "val_pr_auc": 0.5},
        brier_raw=0.2, brier_cal=0.1,
        ead_metrics={"val_mae": 100.0},
        portfolio_summary=portfolio_summary,
        stress_df=make_stress_df(),
    )

    run_all.archive_run("run-a", started, finished, **common_kwargs)
    run_all.archive_run("run-b", started, finished, **common_kwargs)

    index_df = pd.read_csv(isolated_paths["runs_index"])
    assert list(index_df["run_id"]) == ["run-a", "run-b"]
    # header must appear exactly once
    raw_text = isolated_paths["runs_index"].read_text()
    assert raw_text.count("run_id,started_at_utc") == 1
