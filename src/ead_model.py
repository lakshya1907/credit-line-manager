import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from .config import RANDOM_SEED

def make_balance_target(df, alpha=0.7):
    # Proxy for next-period balance using last two bills
    b1 = df["BILL_AMT1"].astype(float)
    b2 = df["BILL_AMT2"].astype(float) if "BILL_AMT2" in df.columns else b1
    y = alpha * b1 + (1 - alpha) * b2
    return y.clip(lower=0)

def train_ead_model(X, y_balance):
    id_col = "customer_id" if "customer_id" in X.columns else None
    X_trainable = X.drop(columns=[id_col]) if id_col else X

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainable, y_balance, test_size=0.2, random_state=RANDOM_SEED
    )

    model = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)

    return model, {"val_mae": float(mae)}
