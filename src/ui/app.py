from __future__ import annotations

import sys
from pathlib import Path
from io import StringIO

import pandas as pd
import streamlit as st


# ============================================================
# Project path setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.model.inference import predict_from_url


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Phishing URL Detection",
    page_icon="🔐",
    layout="wide"
)


# ============================================================
# Helper functions
# ============================================================

def classify_probability(phishing_probability: float) -> str:
    """
    Convert phishing probability into a simple risk level.
    """
    if phishing_probability >= 0.80:
        return "High Risk"
    elif phishing_probability >= 0.50:
        return "Medium Risk"
    return "Low Risk"


def run_single_url_prediction(url: str) -> pd.DataFrame:
    """
    Run prediction for a single raw URL.
    """
    return predict_from_url(url)


def run_batch_url_prediction(df: pd.DataFrame, url_column: str = "URL") -> pd.DataFrame:
    """
    Run prediction for every URL in a DataFrame.
    """
    if url_column not in df.columns:
        raise ValueError(
            f"The uploaded CSV must contain a '{url_column}' column."
        )

    prediction_rows = []

    for url in df[url_column]:
        result = predict_from_url(str(url)).iloc[0].to_dict()
        prediction_rows.append(result)

    predictions_df = pd.DataFrame(prediction_rows)

    final_df = pd.concat(
        [df.reset_index(drop=True), predictions_df.reset_index(drop=True)],
        axis=1
    )

    final_df["risk_level"] = final_df["phishing_probability"].apply(classify_probability)

    return final_df


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convert DataFrame to CSV bytes for Streamlit download button.
    """
    return df.to_csv(index=False).encode("utf-8")


# ============================================================
# UI Layout
# ============================================================

st.title("🔐 AI-based Phishing URL Detection")
st.write(
    "This application classifies URLs as **Phishing** or **Legitimate** "
    "using a machine learning model trained on URL-based features."
)

st.info(
    "Label meaning: 0 = Phishing, 1 = Legitimate. "
    "The final model uses features extracted directly from the raw URL."
)

tab_single, tab_batch, tab_about = st.tabs([
    "Single URL Check",
    "Batch CSV Check",
    "Model Info"
])


# ============================================================
# Single URL tab
# ============================================================

with tab_single:
    st.header("Single URL Check")

    url_input = st.text_input(
        "Enter a URL:",
        placeholder="https://example.com/login"
    )

    analyze_button = st.button("Analyze URL", type="primary")

    if analyze_button:
        if not url_input.strip():
            st.error("Please enter a URL before analysis.")
        else:
            try:
                result = run_single_url_prediction(url_input)
                row = result.iloc[0]

                prediction_label = row["prediction_label"]
                phishing_probability = row["phishing_probability"]
                legitimate_probability = row["legitimate_probability"]
                risk_level = classify_probability(phishing_probability)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Prediction", prediction_label)

                with col2:
                    st.metric(
                        "Phishing Probability",
                        f"{phishing_probability:.4f}"
                    )

                with col3:
                    st.metric(
                        "Legitimate Probability",
                        f"{legitimate_probability:.4f}"
                    )

                if prediction_label == "Phishing":
                    st.error(f"Result: This URL is classified as Phishing. Risk level: {risk_level}")
                else:
                    st.success(f"Result: This URL is classified as Legitimate. Risk level: {risk_level}")

                st.subheader("Raw Prediction Output")
                st.dataframe(result, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ============================================================
# Batch CSV tab
# ============================================================

with tab_batch:
    st.header("Batch CSV Check")

    st.write(
        "Upload a CSV file containing a column named **URL**. "
        "The application will classify each URL and return a downloadable result file."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")
            st.dataframe(input_df.head(), use_container_width=True)

            if "URL" not in input_df.columns:
                st.error("The uploaded CSV must contain a column named 'URL'.")
            else:
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner("Running predictions..."):
                        output_df = run_batch_url_prediction(input_df)

                    st.success("Batch prediction completed successfully.")

                    st.subheader("Prediction Results")
                    st.dataframe(output_df, use_container_width=True)

                    csv_data = convert_df_to_csv(output_df)

                    st.download_button(
                        label="Download Prediction CSV",
                        data=csv_data,
                        file_name="phishing_url_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")


# ============================================================
# Model Info tab
# ============================================================

with tab_about:
    st.header("Model Information")

    st.markdown(
        """
        ### Project Summary

        This project detects phishing URLs using machine learning.

        ### Final Model

        - Model: `HistGradientBoostingClassifier`
        - Input type: Raw URL string
        - Feature source: `feature_extractor.py`
        - Output:
          - prediction
          - prediction_label
          - phishing_probability
          - legitimate_probability

        ### Label Meaning

        - `0 = Phishing`
        - `1 = Legitimate`

        ### Final Extractor-Based Model Performance

        - Accuracy: `0.9951`
        - Precision: `0.9929`
        - Recall: `0.9986`
        - F1 Score: `0.9957`
        - ROC AUC: `0.9979`
        - PR AUC: `0.9973`

        ### Important Note

        The model uses structural URL features. It does not use live blacklist lookup,
        WHOIS data, DNS reputation, or page content analysis.
        """
    )