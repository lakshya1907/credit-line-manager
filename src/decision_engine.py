import numpy as np
import pandas as pd
from .config import LIMIT_MULTIPLIERS, PD_INCREASE_MAX, PD_DECREASE_MIN
from .counterfactual import build_counterfactual_batch
from .economics import balance_under_limit, robust_ep
from .calibrate import apply_calibrator

def recommend_limits(X_feat, pd_model, pd_calibrator, ead_model, pd_shock=0.0, ead_shock=0.0):
    """
    pd_shock / ead_shock: proportional macro stress applied *inside* the
    simulation (e.g. pd_shock=0.2 -> every calibrated PD, baseline and
    every counterfactual, is scaled by 1.2 and clipped to [0, 1] before the
    guardrails and EP comparison run; ead_shock scales the predicted base
    balance the same way before elasticity is applied). This lets the
    guardrails and the best-candidate choice itself respond to stress,
    unlike src.stress_test.apply_pd_shock/apply_ead_shock, which only
    rescale an already-decided recommendation table after the fact. See
    src/stress_test.py for which one to use where.

    Vectorized: every customer's baseline is scored in one batched
    pd_model/ead_model call, and every (customer, candidate-limit) pair is
    built via counterfactual.build_counterfactual_batch and scored in one
    further batched pd_model call, instead of the ~n*(k+1) individual
    single-row model calls this used to make (the dominant cost of a full
    30k-customer run: XGBoost is fast on large batches, but per-call
    overhead on 1-row DataFrames, called ~210,000 times, was not). See
    tests/test_decision_engine.py for the guardrail/tie-break semantics
    this preserves exactly (first-candidate-wins ties, strict Improvement
    over baseline required, etc.).
    """
    df = X_feat.reset_index(drop=True).copy()
    n = len(df)
    id_col = "customer_id" if "customer_id" in df.columns else None
    multipliers = np.array(LIMIT_MULTIPLIERS, dtype=float)
    k = len(multipliers)

    def trainable(frame):
        return frame.drop(columns=[id_col]) if id_col else frame

    def shocked_pd(raw_pd):
        return np.clip(np.asarray(raw_pd, dtype=float) * (1.0 + pd_shock), 0.0, 1.0)

    X_train = trainable(df)
    L0 = df["LIMIT_BAL"].to_numpy(dtype=float)

    # ── Baseline: one batched call for all n customers ──
    base_balance = np.asarray(ead_model.predict(X_train), dtype=float)
    base_balance = np.maximum(base_balance * (1.0 + ead_shock), 0.0)

    s0 = np.asarray(pd_model.predict_proba(X_train))[:, 1]
    pd0 = shocked_pd(apply_calibrator(pd_calibrator, s0))

    ead0 = balance_under_limit(base_balance, L0, L0)
    ep0, _ = robust_ep(pd0, ead0)

    # ── Candidate grid: one batched call for all n*k (customer, candidate) pairs ──
    cf_batch = build_counterfactual_batch(df, multipliers)
    cf_train = trainable(cf_batch)

    s1 = np.asarray(pd_model.predict_proba(cf_train))[:, 1]
    pd1 = shocked_pd(apply_calibrator(pd_calibrator, s1))

    L0_rep = np.repeat(L0, k)
    base_balance_rep = np.repeat(base_balance, k)
    L1 = np.tile(multipliers, n) * L0_rep

    ead1 = balance_under_limit(base_balance_rep, L0_rep, L1)
    ep1, _ = robust_ep(pd1, ead1)

    # Guardrails: an "increase" candidate (L1 > L0) is dropped if it would
    # push PD above either threshold -- matches the two `continue`s in the
    # original per-row loop exactly.
    is_increase = L1 > L0_rep
    blocked = is_increase & ((pd1 > PD_INCREASE_MAX) | (pd1 > PD_DECREASE_MIN))
    ep1_masked = np.where(blocked, -np.inf, ep1)

    ep1_grid = ep1_masked.reshape(n, k)
    L1_grid = L1.reshape(n, k)
    pd1_grid = pd1.reshape(n, k)
    ead1_grid = ead1.reshape(n, k)

    # np.argmax returns the *first* index achieving the max, matching the
    # original loop's "if ep1 > best_ep" (strict >, so the first candidate
    # to strictly beat the running best wins any tie against later ones).
    rows = np.arange(n)
    best_idx = np.argmax(ep1_grid, axis=1)
    best_ep_candidate = ep1_grid[rows, best_idx]

    # A candidate only replaces the baseline if it strictly beats it --
    # matches best_ep starting at ep0 in the original loop, so an exact tie
    # with the baseline (or every candidate being guardrail-blocked, all
    # -inf) leaves the customer at baseline (action "hold").
    candidate_better = best_ep_candidate > ep0

    best_L = np.where(candidate_better, L1_grid[rows, best_idx], L0)
    best_pd = np.where(candidate_better, pd1_grid[rows, best_idx], pd0)
    best_ead = np.where(candidate_better, ead1_grid[rows, best_idx], ead0)
    best_ep = np.where(candidate_better, best_ep_candidate, ep0)

    action = np.full(n, "hold", dtype=object)
    action[best_L > L0 * 1.001] = "increase"
    action[best_L < L0 * 0.999] = "decrease"

    rec_df = pd.DataFrame({
        "current_limit": L0,
        "recommended_limit": best_L,
        "action": action,
        "pd_current": pd0,
        "pd_recommended": best_pd,
        "ead_current": ead0,
        "ead_recommended": best_ead,
        "ep_current": ep0,
        "ep_recommended": best_ep,
        "ep_uplift": best_ep - ep0,
        "el_uplift_proxy": (best_pd * best_ead) - (pd0 * ead0),
        "ead_uplift": best_ead - ead0,
    })

    if id_col:
        rec_df.insert(0, "customer_id", df[id_col].to_numpy())

    return rec_df
