from sklearn.isotonic import IsotonicRegression

def calibrate_pd(scores, y):
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(scores, y)
    return model

def apply_calibrator(model, scores):
    return model.predict(scores)