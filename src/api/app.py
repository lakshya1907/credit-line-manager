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

import logging
import os

import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.logging_config import Timer, configure_logging, log_request
from src.data_prep import load_uci, basic_clean
from src.features import build_features

access_logger = logging.getLogger("api.access")

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
    # Must run here, not at module import time: uvicorn applies its own
    # logging.config.dictConfig() (disable_existing_loggers=True by
    # default) as part of starting the server, *after* this module is
    # imported -- calling configure_logging() at import time got silently
    # undone by that, and every one of our JSON log lines was lost (found
    # by actually reading `docker logs` and seeing plain uvicorn-format
    # access lines instead of JSON -- looked fine in the code, wasn't).
    # lifespan startup runs after uvicorn's own logging setup, so this is
    # the reliable place.
    configure_logging()
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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One JSON log line per request (method, path, status, duration) --
    see src/api/logging_config.py. Not request tracing/metrics, just
    enough to read container logs and answer "what got hit and how long
    did it take" without a separate observability stack."""
    with Timer() as t:
        response = await call_next(request)
    log_request(access_logger, request.method, request.url.path, response.status_code, t.duration_ms)
    return response


from src.api.routers import health, runs, customers  # noqa: E402  (after `app` to avoid circular import surprises)

app.include_router(health.router)
app.include_router(runs.router)
app.include_router(customers.router)
