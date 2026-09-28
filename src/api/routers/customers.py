from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.api.schemas import CustomerHistoryEntry, ScoreRequest, ScoreResponse
from src.db.models import ModelRun, Recommendation
from src.decision_engine import recommend_limits
from src.counterfactual import apply_new_limit_features
from src.economics import balance_under_limit, robust_ep
from src.calibrate import apply_calibrator
from src.config import PD_INCREASE_MAX, PD_DECREASE_MIN
import pandas as pd

router = APIRouter(prefix="/customers", tags=["customers"])


@router.get("/{customer_id}/history", response_model=list[CustomerHistoryEntry])
def customer_history(customer_id: int, db: Session = Depends(get_db)):
    """A customer's recommendation across every synced run, oldest first --
    the thing the pre-#2/#4 file-based pipeline structurally couldn't
    answer, since every run overwrote the last one's output."""
    stmt = (
        select(Recommendation, ModelRun.started_at)
        .join(ModelRun, Recommendation.model_run_id == ModelRun.run_id)
        .where(Recommendation.customer_id == customer_id)
        .order_by(ModelRun.started_at)
    )
    rows = db.execute(stmt).all()
    if not rows:
        raise HTTPException(404, f"No recommendation history found for customer_id {customer_id}")
    return [
        CustomerHistoryEntry(
            run_id=rec.model_run_id, started_at=started_at, action=rec.action,
            current_limit=rec.current_limit, recommended_limit=rec.recommended_limit,
            pd_current=rec.pd_current, pd_recommended=rec.pd_recommended, ep_uplift=rec.ep_uplift,
        )
        for rec, started_at in rows
    ]


def _trainable_row(row: pd.Series) -> pd.DataFrame:
    frame = row.drop(labels=["customer_id"]).to_frame().T
    return frame.astype(float)


@router.post("/{customer_id}/score", response_model=ScoreResponse)
def score_customer(customer_id: int, body: ScoreRequest, request: Request):
    """Live single-customer what-if scoring against the currently loaded
    models (models/*.pkl, loaded once at startup -- see the app lifespan),
    not the DB. With no body, runs the same best-candidate search the
    decision engine does for this one customer (cheap: ~7 model calls, not
    a 30k-customer batch). With {"new_limit": X}, scores exactly that
    hypothetical limit and reports whether the PD guardrails would block
    it as an increase -- useful for "what if I set this customer's limit
    to $X" questions a batch run doesn't answer directly."""
    ml = request.app.state.ml
    if not ml.get("models_loaded"):
        raise HTTPException(503, "Models not loaded -- run `python run_all.py` first.")

    X = ml["X"]
    if customer_id not in X.index:
        raise HTTPException(404, f"customer_id {customer_id} not found in the current feature set")
    row = X.loc[customer_id]

    pd_model, pd_calibrator, ead_model = ml["pd_model"], ml["pd_calibrator"], ml["ead_model"]
    row_train = _trainable_row(row)

    L0 = float(row["LIMIT_BAL"])
    base_balance = max(float(ead_model.predict(row_train)[0]), 0.0)
    s0 = float(pd_model.predict_proba(row_train)[0, 1])
    pd0 = float(apply_calibrator(pd_calibrator, [s0])[0])
    ead0 = balance_under_limit(base_balance, L0, L0)
    ep0, _ = robust_ep(pd0, ead0)

    if body.new_limit is None:
        single = recommend_limits(pd.DataFrame([row]), pd_model, pd_calibrator, ead_model)
        r = single.iloc[0]
        return ScoreResponse(
            customer_id=customer_id,
            current_limit=L0, evaluated_limit=float(r["recommended_limit"]), action=str(r["action"]),
            guardrail_blocked=False,
            pd_current=pd0, pd_evaluated=float(r["pd_recommended"]),
            ead_current=ead0, ead_evaluated=float(r["ead_recommended"]),
            ep_current=ep0, ep_evaluated=float(r["ep_recommended"]), ep_uplift=float(r["ep_uplift"]),
        )

    L1 = float(body.new_limit)
    if L1 <= 0:
        raise HTTPException(422, "new_limit must be positive")

    cf = apply_new_limit_features(row, L1)
    cf_train = _trainable_row(cf)
    s1 = float(pd_model.predict_proba(cf_train)[0, 1])
    pd1 = float(apply_calibrator(pd_calibrator, [s1])[0])

    is_increase = L1 > L0
    guardrail_blocked = is_increase and (pd1 > PD_INCREASE_MAX or pd1 > PD_DECREASE_MIN)

    ead1 = balance_under_limit(base_balance, L0, L1)
    ep1, _ = robust_ep(pd1, ead1)

    action = "hold"
    if L1 > L0 * 1.001:
        action = "increase"
    elif L1 < L0 * 0.999:
        action = "decrease"

    return ScoreResponse(
        customer_id=customer_id,
        current_limit=L0, evaluated_limit=L1, action=action, guardrail_blocked=guardrail_blocked,
        pd_current=pd0, pd_evaluated=pd1, ead_current=ead0, ead_evaluated=ead1,
        ep_current=ep0, ep_evaluated=ep1, ep_uplift=ep1 - ep0,
    )
