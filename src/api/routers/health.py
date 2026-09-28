from fastapi import APIRouter, Request
from sqlalchemy import text

from src.api.schemas import HealthResponse, ReadinessResponse
from src.db.session import get_engine

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    """Liveness: the process is up. Does not touch the DB or models --
    that's readiness's job."""
    return HealthResponse(status="ok")


@router.get("/readiness", response_model=ReadinessResponse)
def readiness(request: Request):
    """Readiness: can this instance actually serve traffic -- DB reachable
    and models loaded. A load balancer should use this, not /health, to
    decide whether to route requests here."""
    db_status = "ok"
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        db_status = "unreachable"

    models_loaded = bool(getattr(request.app.state, "ml", {}).get("models_loaded", False))
    overall = "ok" if db_status == "ok" and models_loaded else "degraded"
    return ReadinessResponse(status=overall, database=db_status, models_loaded=models_loaded)
