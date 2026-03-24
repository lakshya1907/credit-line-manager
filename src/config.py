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