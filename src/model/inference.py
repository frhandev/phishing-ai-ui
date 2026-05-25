from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import pandas as pd

from src.model.feature_extractor import extract_url_features


# ============================================================
# Paths
# ============================================================

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

# Original feature-based model artifacts
MODEL_PATH = ARTIFACTS_DIR / "phishing_url_model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "url_features.pkl"

# Deployment-ready raw URL model artifacts
RAW_URL_MODEL_PATH = ARTIFACTS_DIR / "raw_url_model.pkl"
RAW_URL_FEATURES_PATH = ARTIFACTS_DIR / "raw_url_features.pkl"


# ============================================================
# Label meaning
# ============================================================

# Important:
# In this dataset:
#   0 = Phishing
#   1 = Legitimate
LABEL_MAPPING = {
    0: "Phishing",
    1: "Legitimate"
}


# ============================================================
# Artifact loading
# ============================================================

def load_model_artifacts():
    """
    Load the original feature-based model and its feature list.

    This model expects precomputed URL/page feature columns.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature list file not found: {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_list = joblib.load(FEATURES_PATH)

    return model, feature_list


def load_raw_url_model_artifacts():
    """
    Load the deployment-ready raw URL model and its feature list.

    This model was trained on features extracted by feature_extractor.py.
    Therefore, it is suitable for direct URL input.
    """
    if not RAW_URL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Raw URL model file not found: {RAW_URL_MODEL_PATH}")

    if not RAW_URL_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Raw URL feature list file not found: {RAW_URL_FEATURES_PATH}")

    model = joblib.load(RAW_URL_MODEL_PATH)
    feature_list = joblib.load(RAW_URL_FEATURES_PATH)

    return model, feature_list


# ============================================================
# Input preparation
# ============================================================

def prepare_input(
    data: Union[pd.DataFrame, dict],
    feature_list: list[str]
) -> pd.DataFrame:
    """
    Validate and prepare input data for model prediction.

    Args:
        data:
            A pandas DataFrame or a single dictionary containing feature values.
        feature_list:
            Ordered list of features expected by the model.

    Returns:
        A DataFrame containing only the required features in the correct order.

    Raises:
        TypeError:
            If input is not a DataFrame or dict.
        ValueError:
            If required features are missing.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Input data must be a pandas DataFrame or a dictionary.")

    missing_features = [
        feature for feature in feature_list
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Keep only required features and enforce the exact training order
    df = df[feature_list]

    # Ensure all feature values are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


# ============================================================
# Prediction formatting
# ============================================================

def build_prediction_output(predictions, probabilities) -> pd.DataFrame:
    """
    Build a clean prediction output DataFrame.

    Important:
        In this dataset:
            class 0 = Phishing
            class 1 = Legitimate

    Therefore:
        probabilities[:, 0] = phishing probability
        probabilities[:, 1] = legitimate probability
    """
    results = pd.DataFrame({
        "prediction": predictions,
        "prediction_label": [LABEL_MAPPING[int(pred)] for pred in predictions],
        "phishing_probability": probabilities[:, 0],
        "legitimate_probability": probabilities[:, 1],
    })

    return results


# ============================================================
# Prediction functions
# ============================================================

def predict_from_dataframe(data: Union[pd.DataFrame, dict]) -> pd.DataFrame:
    """
    Run prediction using the original feature-based model.

    Args:
        data:
            DataFrame or dictionary containing all required model features.

    Returns:
        DataFrame containing:
            - prediction
            - prediction_label
            - phishing_probability
            - legitimate_probability
    """
    model, feature_list = load_model_artifacts()
    X = prepare_input(data, feature_list)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    return build_prediction_output(predictions, probabilities)


def predict_from_url(url: str) -> pd.DataFrame:
    """
    Run phishing prediction directly from a raw URL string.

    Processing pipeline:
        Raw URL
        -> feature_extractor.py
        -> raw_url_model.pkl
        -> prediction output

    Args:
        url:
            Raw URL string entered by the user.

    Returns:
        DataFrame containing:
            - prediction
            - prediction_label
            - phishing_probability
            - legitimate_probability
    """
    model, feature_list = load_raw_url_model_artifacts()

    extracted_features = extract_url_features(url)
    X = prepare_input(extracted_features, feature_list)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    return build_prediction_output(predictions, probabilities)


# ============================================================
# Manual test
# ============================================================

if __name__ == "__main__":
    test_urls = [
        "https://example.com/login?user=test&id=123",
        "http://192.168.1.1/login",
        "https://www.google.com",
        "paypal-security-check.com/login",
        "http://free-gift-card-login.ru/verify?account=123",
    ]

    for test_url in test_urls:
        print("=" * 80)
        print("Input URL:")
        print(test_url)

        print("\nPrediction output:")
        print(predict_from_url(test_url))
        print()