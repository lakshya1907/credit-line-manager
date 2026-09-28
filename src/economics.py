import numpy as np
from .config import EPS, EAD_ELASTICITY, APR_ANNUAL_SCENARIOS, LGD_SCENARIOS, ROBUST_MODE

def balance_under_limit(base_balance, L0, L1):
    """
    Accepts scalars or numpy arrays/pandas Series of equal (broadcastable)
    shape. decision_engine.py relies on the array path to score every
    (customer, candidate-limit) pair in one vectorized pass instead of one
    Python call per candidate. Returns a plain float when every input was
    scalar, otherwise an ndarray of the broadcast shape.
    """
    base_balance = np.maximum(np.asarray(base_balance, dtype=float), 0.0)
    L0 = np.maximum(np.asarray(L0, dtype=float), 1.0)
    L1 = np.maximum(np.asarray(L1, dtype=float), 1.0)
    scale = 1.0 + EAD_ELASTICITY * np.log((L1 + EPS) / (L0 + EPS))
    scale = np.maximum(scale, 0.2)
    result = base_balance * scale
    return result.item() if result.ndim == 0 else result

def scenario_eps(pd_cal, ead):
    """Accepts scalars or equally-shaped arrays; see balance_under_limit."""
    ead = np.maximum(np.asarray(ead, dtype=float), 0.0)
    eps = []
    for apr_a in APR_ANNUAL_SCENARIOS:
        apr_m = apr_a / 12.0
        for lgd in LGD_SCENARIOS:
            er = apr_m * ead
            el = pd_cal * ead * lgd
            eps.append(er - el)
    return eps

def robust_ep(pd_cal, ead):
    """Accepts scalars or equally-shaped arrays; see balance_under_limit."""
    eps = scenario_eps(pd_cal, ead)
    stacked = np.stack([np.asarray(e, dtype=float) for e in eps], axis=0)
    if ROBUST_MODE == "worst_case":
        result = np.min(stacked, axis=0)
    else:
        result = np.mean(stacked, axis=0)
    return (result.item() if result.ndim == 0 else result), eps
