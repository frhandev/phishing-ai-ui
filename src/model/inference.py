from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path
from typing import Union

from src.model.feature_extractor import extract_url_features


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "phishing_url_model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "url_features.pkl"

RAW_URL_MODEL_PATH = ARTIFACTS_DIR / "raw_url_model.pkl"
RAW_URL_FEATURES_PATH = ARTIFACTS_DIR / "raw_url_features.pkl"


def load_model_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature list file not found: {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_list = joblib.load(FEATURES_PATH)

    return model, feature_list

def load_raw_url_model_artifacts():
    if not RAW_URL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Raw URL model file not found: {RAW_URL_MODEL_PATH}")

    if not RAW_URL_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Raw URL feature list file not found: {RAW_URL_FEATURES_PATH}")

    model = joblib.load(RAW_URL_MODEL_PATH)
    feature_list = joblib.load(RAW_URL_FEATURES_PATH)

    return model, feature_list


def prepare_input(
    data: Union[pd.DataFrame, dict],
    feature_list: list[str]
) -> pd.DataFrame:
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Input data must be a pandas DataFrame or a dictionary.")

    missing_features = [feature for feature in feature_list if feature not in df.columns]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    df = df[feature_list]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def predict_from_dataframe(data: Union[pd.DataFrame, dict]) -> pd.DataFrame:
    model, feature_list = load_model_artifacts()
    X = prepare_input(data, feature_list)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    results = pd.DataFrame({
        "prediction": predictions,
        "legitimate_probability": probabilities[:, 0],
        "phishing_probability": probabilities[:, 1]
    })

    return results

def predict_from_url(url: str) -> pd.DataFrame:
    model, feature_list = load_raw_url_model_artifacts()

    extracted_features = extract_url_features(url)

    X = prepare_input(extracted_features, feature_list)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    results = pd.DataFrame({
        "prediction": predictions,
        "legitimate_probability": probabilities[:, 0],
        "phishing_probability": probabilities[:, 1]
    })

    return results


if __name__ == "__main__":
    test_url = "https://example.com/login?user=test&id=123"

    output = predict_from_url(test_url)

    print("Input URL:")
    print(test_url)

    print("\nPrediction output:")
    print(output)