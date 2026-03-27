# src/config.py

RANDOM_SEED = 42
EPS = 1e-6

# Finance
APR_ANNUAL = 0.24
APR_MONTHLY = APR_ANNUAL / 12
LGD = 0.6

# Credit limit candidates
LIMIT_MULTIPLIERS = [0.8, 1.0, 1.2, 1.5]

# EAD constraint
MAX_BALANCE_FRAC_OF_LIMIT = 1.0

# --- Scenario engine (Milestone 2) ---
APR_ANNUAL_SCENARIOS = [0.18, 0.24, 0.30]  # conservative/base/aggressive
LGD_SCENARIOS = [0.40, 0.60, 0.80]         # optimistic/base/stress

# Elasticity factor: how predicted balance responds to limit change
# balance(L') = base_balance * (1 + ELAST * log(L'/L0))
EAD_ELASTICITY = 0.35

# robust objective: "worst_case" or "expected"
ROBUST_MODE = "worst_case"
