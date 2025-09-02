# model.py
from joblib import load
from utils import basic_clean

_model = None

def load_model(path="model.joblib"):
    global _model
    if _model is None:
        _model = load(path)
    return _model

def predict_proba(text: str):
    m = load_model()
    cleaned = basic_clean(text)
    # returns probability of class 1 = phishing
    p = m.predict_proba([cleaned])[0][1]
    return float(p)
