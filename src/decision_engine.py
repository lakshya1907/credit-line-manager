from .config import LIMIT_MULTIPLIERS
from .economics import ead_proxy, expected_profit

def recommend_limits(X, pd):

    results = []

    for i, row in X.iterrows():
        L0 = row["LIMIT_BAL"]

        best_L = L0
        best_ep = -1e9

        for m in LIMIT_MULTIPLIERS:
            L = L0 * m
            ead = ead_proxy(row, L)
            ep, er, el = expected_profit(pd[i], ead)

            if ep > best_ep:
                best_ep = ep
                best_L = L

        action = "hold"
        if best_L > L0:
            action = "increase"
        elif best_L < L0:
            action = "decrease"

        results.append({
            "current_limit": L0,
            "recommended_limit": best_L,
            "action": action,
            "pd": pd[i],
            "ep": best_ep
        })

    return results