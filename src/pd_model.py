from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from .config import RANDOM_SEED

def train_pd_model(X, y):

    id_col = "customer_id" if "customer_id" in X.columns else None
    X_trainable = X.drop(columns=[id_col]) if id_col else X

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainable, y, test_size=0.2,
        random_state=RANDOM_SEED, stratify=y
    )

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    p_val = model.predict_proba(X_val)[:, 1]

    return model, {
        "auc": roc_auc_score(y_val, p_val),
        "pr_auc": average_precision_score(y_val, p_val)
    }, (X_train, X_val, y_train, y_val)