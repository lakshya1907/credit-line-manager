# src/data_prep.py

import pandas as pd

def load_uci(path: str):
    import pandas as pd

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower()

    print("Columns:", df.columns.tolist())

    rename_map = {
        "limit_bal": "LIMIT_BAL",
        "bill_amt1": "BILL_AMT1",
        "bill_amt2": "BILL_AMT2",
        "bill_amt3": "BILL_AMT3",
        "bill_amt4": "BILL_AMT4",
        "bill_amt5": "BILL_AMT5",
        "bill_amt6": "BILL_AMT6",
        "pay_amt1": "PAY_AMT1",
        "pay_amt2": "PAY_AMT2",
        "pay_amt3": "PAY_AMT3",
        "pay_amt4": "PAY_AMT4",
        "pay_amt5": "PAY_AMT5",
        "pay_amt6": "PAY_AMT6",
        "pay_0": "PAY_0",
        "pay_2": "PAY_2",
        "pay_3": "PAY_3",
        "pay_4": "PAY_4",
        "pay_5": "PAY_5",
        "pay_6": "PAY_6",
        "default.payment.next.month": "TARGET",
        "id": "customer_id"
    }

    df = df.rename(columns=rename_map)

    # 🔥 FIX HERE
    if "PAY_1" not in df.columns and "PAY_0" in df.columns:
        df["PAY_1"] = df["PAY_0"]

    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates()
    df = df.fillna(0)

    return df