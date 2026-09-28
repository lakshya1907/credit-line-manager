"""
src/api/app.py
─────────────────
FastAPI app. Run with:
    uvicorn src.api.app:app --reload

Loads models/*.pkl and rebuilds the feature matrix once at startup (into
app.state.ml), not per-request -- see the lifespan below. If models/*.pkl
don't exist yet (fresh clone, run_all.py never run), the app still starts
(readiness reports models_loaded=False, /customers/*/score returns 503)
rather than crashing at import time.
"""

import os

import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.data_prep import load_uci, basic_clean
from src.features import build_features

MODEL_DIR = "models"
RAW_PATH = "data/raw/uci_credit.csv"

# The frontend dev server goes through Vite's proxy (see frontend/vite.config.ts),
# which needs no CORS config -- this is for calling the API directly
# (production build, or any other client). Comma-separated in CORS_ORIGINS;
# defaults cover Vite's own dev-server origin as a convenience if the proxy
# is ever bypassed.
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", _default_origins).split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    state: dict = {"models_loaded": False}
    try:
        state["pd_model"] = joblib.load(os.path.join(MODEL_DIR, "pd_xgb.pkl"))
        state["pd_calibrator"] = joblib.load(os.path.join(MODEL_DIR, "pd_calibrator.pkl"))
        state["ead_model"] = joblib.load(os.path.join(MODEL_DIR, "ead_xgb.pkl"))
        df = load_uci(RAW_PATH)
        df = basic_clean(df)
        X, _y = build_features(df)
        state["X"] = X.set_index("customer_id", drop=False)
        state["models_loaded"] = True
    except FileNotFoundError:
        pass  # models/*.pkl or the raw CSV don't exist yet -- degraded, not fatal
    app.state.ml = state
    yield
    app.state.ml = {}


app = FastAPI(
    title="Credit Line Manager API",
    description="Read access to run history + live single-customer scoring. "
                 "See CLAUDE.md's Database/API sections for the endpoint design rationale.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.routers import health, runs, customers  # noqa: E402  (after `app` to avoid circular import surprises)

app.include_router(health.router)
app.include_router(runs.router)
app.include_router(customers.router)
