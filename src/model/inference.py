from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path
from typing import Union


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "phishing_url_model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "url_features.pkl"


def load_model_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature list file not found: {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_list = joblib.load(FEATURES_PATH)

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


if __name__ == "__main__":
    sample_input = {
        "URLLength": 54,
        "DomainLength": 18,
        "IsDomainIP": 0,
        "TLDLegitimateProb": 0.4,
        "URLCharProb": 0.03,
        "TLDLength": 3,
        "NoOfSubDomain": 1,
        "HasObfuscation": 0,
        "NoOfObfuscatedChar": 0,
        "ObfuscationRatio": 0.0,
        "NoOfLettersInURL": 35,
        "LetterRatioInURL": 0.64,
        "NoOfDegitsInURL": 4,
        "DegitRatioInURL": 0.07,
        "NoOfEqualsInURL": 0,
        "NoOfQMarkInURL": 1,
        "NoOfAmpersandInURL": 0,
        "NoOfOtherSpecialCharsInURL": 3,
        "SpacialCharRatioInURL": 0.05,
        "IsHTTPS": 1,
        "DomainTitleMatchScore": 0.0,
        "URLTitleMatchScore": 0.0,
        "CharContinuationRate": 0.8
    }

    output = predict_from_dataframe(sample_input)
    print(output)