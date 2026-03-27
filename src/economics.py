import numpy as np
from .config import EPS, EAD_ELASTICITY, APR_ANNUAL_SCENARIOS, LGD_SCENARIOS, ROBUST_MODE

def balance_under_limit(base_balance, L0, L1):
    base_balance = float(max(base_balance, 0.0))
    L0 = float(max(L0, 1.0))
    L1 = float(max(L1, 1.0))
    scale = 1.0 + EAD_ELASTICITY * np.log((L1 + EPS) / (L0 + EPS))
    scale = max(scale, 0.2)
    return base_balance * scale

def scenario_eps(pd_cal, ead):
    pd_cal = float(pd_cal)
    ead = float(max(ead, 0.0))
    eps = []
    for apr_a in APR_ANNUAL_SCENARIOS:
        apr_m = apr_a / 12.0
        for lgd in LGD_SCENARIOS:
            er = apr_m * ead
            el = pd_cal * ead * lgd
            eps.append(er - el)
    return eps

def robust_ep(pd_cal, ead):
    eps = scenario_eps(pd_cal, ead)
    if ROBUST_MODE == "worst_case":
        return float(min(eps)), eps
    return float(np.mean(eps)), eps

