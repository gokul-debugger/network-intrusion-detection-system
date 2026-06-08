from __future__ import annotations

import joblib
import pandas as pd


def load_model(path):
    return joblib.load(path)


def predict_connection(model, row: pd.DataFrame) -> dict:
    prediction = model.predict(row)[0]
    result = {"prediction": prediction}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        classes = model.classes_
        result["probabilities"] = {str(label): float(prob) for label, prob in zip(classes, probabilities)}

    return result

