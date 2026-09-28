# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Adaptive credit line manager built on the UCI "Default of Credit Card Clients" dataset (`data/raw/uci_credit.csv`). It predicts default probability (PD) and exposure (EAD), simulates alternative credit limits per customer, picks the limit that maximizes risk-adjusted expected profit, then applies portfolio-level budget constraints and stress tests.

## Commands

```bash
pip install -r requirements-dev.txt      # includes requirements.txt + pytest; venv/ is gitignored
python run_all.py                        # full pipeline (run from repo root)
python run_all.py --data path/to/file.csv
python run_analytics.py                  # segments + backtest + policy comparison; needs run_all.py's models/*.pkl first
streamlit run src/dashboard_app.py       # dashboard; needs run_all.py output first
pytest                                   # tests/*.py, pure-function unit tests only (no model training)
pytest tests/test_portfolio_opt.py -q    # single file
```

There is no linter or build step. Run everything from the repo root — `run_all.py` imports `src.*` as a package and all paths (`data/processed`, `models`, `reports`) are relative to the cwd.

The `tests/` suite covers the pure-function modules (`economics.py`, `calibrate.py`, `counterfactual.py`, `portfolio_opt.py`, `decision_engine.py` with fake PD/EAD models) — it does not train real models or run the pipeline end-to-end. `pyproject.toml` sets `pythonpath = ["."]` so `from src... import ...` resolves without installing the package.

The full pipeline (fresh model training included) runs in ~10s. `decision_engine.recommend_limits` scores every customer's baseline and every (customer, candidate-limit) pair via `counterfactual.build_counterfactual_batch` in a small constant number of batched `pd_model`/`ead_model` calls (2 and 1 respectively, regardless of `n`), not one call per row/candidate — the per-call overhead of ~210,000 individual 1-row model calls, not the actual model compute, was what made this slow before (~10 min for one run, ~27 min for a 4-shock stress test). See `git log` on `decision_engine.py`/`counterfactual.py`/`economics.py` for the change and `tests/test_counterfactual.py`'s equivalence tests against the old per-row path. SHAP is capped at `sample_n=5000` in `step_explainability`.

## Pipeline architecture (`run_all.py`)

Steps run in this order (note explainability runs before portfolio, despite its "6b" label):

1. `data_prep` — load CSV, rename target (`default.payment.next.month` → `TARGET`), clip negatives.
2. `features.build_features` — utilization, payment-ratio, delinquency, bill-trend and interaction features; renames `ID` → `customer_id`. Writes `data/processed/features.parquet`.
3. `pd_model` (XGBClassifier) + `calibrate` (isotonic on the validation split). A calibrator is a `(kind, obj)` tuple — always apply it via `apply_calibrator`, never call `.predict` on it directly.
4. `ead_model` (XGBRegressor) — target is a balance proxy `0.7*BILL_AMT1 + 0.3*BILL_AMT2`.
5. `decision_engine.recommend_limits` — for every customer and every `LIMIT_MULTIPLIERS` candidate at once (see the "Vectorized" note above): rebuild limit-dependent features (`counterfactual.build_counterfactual_batch`), re-score PD, scale balance by a log-elasticity (`economics.balance_under_limit`), compute EP. Increases are blocked by PD guardrails (`PD_INCREASE_MAX`, `PD_DECREASE_MIN`). Takes optional `pd_shock`/`ead_shock` (proportional, applied before the guardrails/EP comparison, clipped to valid ranges) so stress scenarios can change which action is chosen, not just its reported numbers. Writes `recommendations_raw.csv`.
6. `explainability.run_explainability` — SHAP top features + human-readable reason codes (`_REASON_MAP`, matched by feature-name substring); overwrites `recommendations_raw.csv` with the annotated version and writes `audit_log.csv` and `reports/shap_global_importance.csv`. `build_explainer` always uses `feature_perturbation="tree_path_dependent"` — `"interventional"` unconditionally raises on any XGBoost model with `enable_categorical=True` (the default since XGBoost 2.x, unrelated to whether any column is actually categorical).
7. `portfolio_opt.portfolio_select` — greedy selection of increases by ROI (`ep_uplift / el_uplift_proxy`) under `EL_BUDGET` / `EAD_BUDGET`; unapproved increases revert to hold. Decreases/holds always pass. The running `used_el`/`used_ead` accumulate the *signed* `el_uplift_proxy`/`ead_uplift` — do not clamp to `max(..., 0.0)` before accumulating, or negative-delta rows silently stop contributing to the budget check (this was a real bug; see `git log` on this file). When an increase is rejected, **every** "recommended" column (`recommended_limit`, `pd_recommended`, `ead_recommended`, `ep_recommended`, `ep_uplift`, `el_uplift_proxy`, `ead_uplift`) must revert to its "current"/zero value, not just `action`/`recommended_limit` — otherwise a row labeled `"hold"` still reports the profit/risk of the increase that was never approved, silently inflating `total_ep_uplift` (this was also a real bug, found via the policy-comparison analytics in `run_analytics.py`; see `reports/analytics_findings.md` §1 and `git log` on this file). Writes `recommendations_final.csv`.
8. `stress_test.py` — see its module docstring for the two stress-testing paths and when each is used: `run_stress_scenario` (genuine re-simulation via `recommend_limits(pd_shock=...)`, used by `run_all.py` step 7/`step_stress_test`) vs. `apply_pd_shock`/`apply_ead_shock` (fast O(1) rescaling of an already-decided recommendation table, used by the dashboard's live Policy Simulator slider; cannot change which action was chosen). `step_write_report` writes `reports/pipeline_metrics.txt`.
9. `archive_run` — copies this run's output files into `data/processed/runs/<run_id>/`, writes `run_manifest.json` (config snapshot + every metric for that run — self-contained, doesn't depend on whatever the flat "latest" files happen to contain later), and appends one row to `data/processed/runs_index.csv`. See "Run history" below.

### Run history

Every `run_all.py` run gets a `run_id` (`_new_run_id`: UTC timestamp + a short random suffix, so runs sort chronologically and never collide). The top-level flat files (`data/processed/recommendations_raw.csv`, `reports/pipeline_metrics.txt`, etc.) are still overwritten each run — that's deliberate, it's what the dashboard reads as "the current plan" — but `archive_run` (the last step of `main()`) additionally:
- copies that run's output files into `data/processed/runs/<run_id>/`
- writes `data/processed/runs/<run_id>/run_manifest.json`: `run_id`, start/finish timestamps, wall time, git commit, the raw data path, a full config snapshot (budgets, guardrails, multipliers, APR/LGD grid, elasticity, seed — the policy that produced this run's recommendations, not just the recommendations themselves), PD/EAD model metrics, the portfolio summary, and the stress-test results
- appends one flattened summary row to `data/processed/runs_index.csv` (creating it with a header on the first run) — this is the file to read for any "how has X trended across runs" analysis; don't recompute that by scanning `runs/*/run_manifest.json`

Both `runs/` and `runs_index.csv` live under `data/processed/`, so they're gitignored like everything else there — this is deliberately plain per-run files (`archive_run` in `run_all.py`), not a database. Before this, every run overwrote its own history (including the file literally named `audit_log.csv`), so there was no way to compare runs or see how a config change affected the portfolio over time; this is the fix, and the natural precursor to a future `model_runs`/`portfolio_runs` schema if/when this becomes a real database.

### Key cross-file invariants

- **Customer ID handling**: `customer_id` stays in `X` but every model call drops it first (the `id_col` / `trainable()` pattern repeated in `pd_model`, `ead_model`, `decision_engine`, `run_all`). Models are trained on the column set without it, so column order/names must match exactly.
- **Counterfactual features must mirror `build_features`**: any new feature derived from `LIMIT_BAL` (utilization and interactions) must be added to `counterfactual._recompute_util_features` — the single core implementation shared by `apply_new_limit_features` (single-row Series in/out, e.g. for a future single-customer what-if endpoint) and `build_counterfactual_batch` (the vectorized `(customer x candidate)` grid decision_engine.py actually uses). `util_cols` there is built explicitly from `util_1..util_6` — do not switch it to `c.startswith("util_")`/`.startswith("util_")`-style matching against existing columns, since the frame still holds the *old* aggregate columns (`util_mean`, `util_max`, `util_std`, `util_last`, `util_trend`, `util_x_delinq`) at that point, which also match that prefix and would contaminate the recompute (this was a real bug; see `git log` on this file and `tests/test_counterfactual.py`, including its `apply_new_limit_features`-vs-`build_counterfactual_batch` equivalence test).
- **Economics** (`economics.py`): EP per scenario = `(APR/12)·EAD − PD·EAD·LGD` over the APR × LGD grid in `config.py`; `ROBUST_MODE="worst_case"` takes the min across scenarios. Given the default grid, worst-case EP-per-dollar-of-EAD is only positive for PD below ~0.019 — above that, minimizing EAD (i.e. decreasing the limit) is the economically rational choice even before any PD guardrail kicks in.
- **All tunables live in `src/config.py`** (limit multipliers, APR/LGD scenarios, elasticity, budgets, guardrails, seed).
- **The counterfactual PD is a correlation, not a causal estimate**: raising a customer's limit lowers their utilization ratio, and the PD model reads low utilization as materially lower risk (since that's the correlation in the training data) — so counterfactual PD tends to drop sharply whenever a limit is raised. In practice this means `EL_BUDGET` essentially never binds (aggregate `el_uplift_proxy` across approved increases comes out net negative). This is a modeling limitation to be aware of, not something `portfolio_opt.py`'s accounting can fix.

## Dashboard (`src/dashboard_app.py`)

Reads only `data/processed/recommendations_raw.csv` (and optionally the pickled models in `models/`), then re-runs `portfolio_select` live using sidebar EL/EAD budget inputs. So changes to columns in the decision engine / explainability output affect the dashboard. Pages: Portfolio Overview, Action Queue, Customer Drilldown, Policy Simulator, Model Diagnostics. The Policy Simulator's PD/EAD shock sliders use the fast-approximation `apply_pd_shock`/`apply_ead_shock` (not `run_stress_scenario`) so slider drags stay instant — its output only approximates the effect of stress, since it can't change which action a customer was recommended.

## Analytics (`run_analytics.py`, `src/analytics/`)

Reads `models/*.pkl` (from `run_all.py`) and produces, to `data/processed/analytics/` (gitignored) plus stdout:
- **`segments.py`** — buckets customers into behavioral segments (`util_tier`, `delinquency_tier`) and demographic segments (`sex_segment`, `education_segment`, `marriage_segment`, `age_segment`) via `add_segment_columns`, then `segment_summary` aggregates a recommendation table per segment value. `SEX`/`EDUCATION`/`MARRIAGE` are fed to `pd_model` as raw numeric features (`build_features` passes them through unchanged) — a real fair-lending concern (ECOA, in the US) independent of whether they're "intentional" — so `disparate_approval_ratio` (a four-fifths-rule-style check) exists specifically to make that checkable, not just to slice data for its own sake. See `reports/analytics_findings.md` §2.
- **`backtest.py`** — see its module docstring for why a classic time-based train/test split doesn't apply to this dataset (one cross-sectional snapshot per customer, not repeated observations over time). `restricted_feature_backtest` instead compares PD model performance trained on only the N most recent months of behavior (via a small, deliberately consistent feature set, `build_restricted_features` — not `features.build_features`, so absolute metrics aren't comparable to the production model). See `reports/analytics_findings.md` §3 for the "recency matters more than history length" finding.
- **`policy_compare.py`** — re-runs the decision engine + portfolio selection under several named guardrail/budget overrides via `decision_engine.recommend_limits`'s `pd_increase_max`/`pd_decrease_min` parameters (added for this; default `None` falls back to `src.config`) and `portfolio_select`'s existing `el_budget`/`ead_budget` params. Only practical because of the vectorization in the perf commit — each policy is a full ~30k-customer run, ~0.3s. See `reports/analytics_findings.md` §4 for the "more approvals ≠ more profit" finding.

`reports/analytics_findings.md` is the actual write-up (git-tracked); `data/processed/analytics/*.csv` is the regeneratable evidence behind it.

## Testing

`tests/` (pytest) covers the pure-function modules — `economics.py`, `calibrate.py`, `counterfactual.py`, `portfolio_opt.py`, `decision_engine.py` (via fake PD/EAD models + an identity calibrator), `stress_test.py`, `src/analytics/*` — plus `run_all.py`'s run-archival helpers (`test_run_archival.py`, which monkeypatches the module's path constants to a `tmp_path` rather than touching real `data/processed/`). It does not train real models or exercise the full pipeline end-to-end (`test_backtest.py` is the one exception — it does train small real XGBoost models on synthetic data, since that's what it's testing). Regression guards (verified by temporarily reverting the corresponding fix and confirming the suite catches it): `test_counterfactual.py::test_util_mean_uses_only_the_six_period_ratios`, `test_portfolio_opt.py::test_negative_el_uplift_is_not_floored_to_zero`, `test_portfolio_opt.py::test_summary_total_ep_uplift_excludes_rejected_increases`.

## Repo notes

- `data/processed/*`, `models/*.pkl`, `__pycache__/`, and `.DS_Store` are gitignored (generated by `run_all.py`; regenerate locally rather than expecting them in a fresh clone). `reports/*` stays tracked since it's small and human-readable — it reflects whatever the last `run_all.py` run on this branch produced.
- Git history is organized by "Milestones" (1: PD prediction, 2: counterfactual decision engine, 3: portfolio optimization, stress test, explainability, dashboard).
