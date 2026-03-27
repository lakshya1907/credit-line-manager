import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

def calibrate_pd(scores_val, y_val, method="isotonic"):
    scores_val = np.asarray(scores_val).ravel()
    y_val = np.asarray(y_val).ravel()
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(scores_val, y_val)
        return ("isotonic", iso)
    lr = LogisticRegression()
    lr.fit(scores_val.reshape(-1, 1), y_val)
    return ("platt", lr)

def apply_calibrator(calibrator, scores):
    kind, obj = calibrator
    scores = np.asarray(scores).ravel()
    if kind == "isotonic":
        return obj.predict(scores)
    return obj.predict_proba(scores.reshape(-1, 1))[:, 1]



