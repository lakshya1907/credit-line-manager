import pandas as pd
import joblib
import os

from src.data_prep import load_uci, basic_clean
from src.features import build_features
from src.pd_model import train_pd_model
from src.calibrate import calibrate_pd, apply_calibrator
from decision_engine_v2 import recommend_limits

DATA_PATH = "data/raw/uci_credit.csv"
OUT_PATH = "data/processed/output.csv"

def main():
    df = load_uci(DATA_PATH)
    df = basic_clean(df)

    X, y = build_features(df)

    model, metrics, splits = train_pd_model(X, y)
    X_train, X_val, y_train, y_val = splits

    val_scores = model.predict_proba(X_val)[:, 1]
    calibrator = calibrate_pd(val_scores, y_val)

    all_scores = model.predict_proba(X.drop(columns=["customer_id"], errors="ignore"))[:, 1]
    pd_cal = apply_calibrator(calibrator, all_scores)

    recs = recommend_limits(X, pd_cal)

    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame(recs).to_csv(OUT_PATH, index=False)

    joblib.dump(model, "models/pd.pkl")
    joblib.dump(calibrator, "models/calibrator.pkl")

    print("DONE:", metrics)

if __name__ == "__main__":
    main()