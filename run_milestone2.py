import os
import joblib
import pandas as pd

from src.data_prep import load_uci, basic_clean
from src.features import build_features
from src.pd_model import train_pd_model
from src.calibrate import calibrate_pd
from src.ead_model import make_balance_target, train_ead_model
from src.decision_engine_v2 import recommend_limits_v2

RAW_PATH = "data/raw/uci_credit.csv"
OUT_PATH = "data/processed/milestone2_recommendations.csv"

def main():
    print("Loading data...")
    df = load_uci(RAW_PATH)

    print("Cleaning...")
    df = basic_clean(df)

    print("Building features...")
    X, y = build_features(df)

    print("Training PD model...")
    pd_model, pd_metrics, splits = train_pd_model(X, y)

    print("Splitting...")
    X_train, X_val, y_train, y_val = splits

    print("Calibrating...")
    val_scores = pd_model.predict_proba(X_val)[:, 1]
    pd_calibrator = calibrate_pd(val_scores, y_val, method="isotonic")

    print("Training EAD...")
    y_balance = make_balance_target(df)
    ead_model, ead_metrics = train_ead_model(X, y_balance)

    print("Recommending...")
    rec = recommend_limits_v2(X, pd_model, pd_calibrator, ead_model)

    print("Saving...")

if __name__ == "__main__":
    main()

