import pandas as pd

def load_uci(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "default.payment.next.month" in df.columns:
        df = df.rename(columns={"default.payment.next.month": "TARGET"})
    elif "default" in df.columns:
        df = df.rename(columns={"default": "TARGET"})
    else:
        raise ValueError("Target not found. Expected 'default.payment.next.month'.")

    df = df.dropna(axis=1, how="all")
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bill_cols = [c for c in df.columns if c.startswith("BILL_AMT")]
    pay_cols  = [c for c in df.columns if c.startswith("PAY_AMT")]

    df[bill_cols] = df[bill_cols].clip(lower=0)
    df[pay_cols]  = df[pay_cols].clip(lower=0)

    df["LIMIT_BAL"] = df["LIMIT_BAL"].clip(lower=1)
    df["TARGET"] = df["TARGET"].astype(int)
    return df
