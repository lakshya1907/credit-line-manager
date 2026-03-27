RANDOM_SEED = 42
EPS = 1e-9

# Candidate credit limits to simulate
LIMIT_MULTIPLIERS = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5]

# Economic scenarios
APR_ANNUAL_SCENARIOS = [0.18, 0.24, 0.30]   # 18/24/30% APR annual
LGD_SCENARIOS = [0.40, 0.60, 0.80]          # optimistic/base/stress LGD

# Robustness mode: "worst_case" or "expected"
ROBUST_MODE = "worst_case"

# Limit-response elasticity in EAD
EAD_ELASTICITY = 0.35

# Portfolio constraints (tunable in dashboard)
EL_BUDGET = 2.0e7          # example expected loss budget (units depend on currency scale)
EAD_BUDGET = 5.0e7         # exposure cap

# Simple policy guardrails
PD_INCREASE_MAX = 0.08     # if PD above this, do not increase (can still decrease)
PD_DECREASE_MIN = 0.20     # if PD above this, strongly consider decrease
