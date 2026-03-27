from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

def calibrate_pd(scores, y, method="isotonic"):
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(scores, y)
        return model
    
    elif method == "platt":
        model = LogisticRegression()
        model.fit(scores.reshape(-1,1), y)
        return model
    
    else:
        raise ValueError("method must be 'isotonic' or 'platt'")

def apply_calibrator(model, scores):
    try:
        return model.predict(scores)
    except:
        return model.predict_proba(scores.reshape(-1,1))[:,1]


