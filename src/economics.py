from .config import APR_MONTHLY, LGD, MAX_BALANCE_FRAC_OF_LIMIT

def ead_proxy(row, limit):
    bill = row["bill_last"]
    return min(bill, limit * MAX_BALANCE_FRAC_OF_LIMIT)

def expected_profit(pd, ead):
    er = APR_MONTHLY * ead
    el = pd * ead * LGD
    return er - el, er, el