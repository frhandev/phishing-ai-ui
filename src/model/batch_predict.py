import pandas as pd
from pathlib import Path
from src.model.inference import predict_from_dataframe

def predict_from_csv(input_csv_path: str, output_csv_path: str) -> None:
    input_path = Path(input_csv_path)
    output_path = Path(output_csv_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_path}")

    df = pd.read_csv(input_path)

    prediction_results = predict_from_dataframe(df)

    final_df = df.copy()
    final_df["prediction"] = prediction_results["prediction"]
    final_df["legitimate_probability"] = prediction_results["legitimate_probability"]
    final_df["phishing_probability"] = prediction_results["phishing_probability"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"Batch prediction completed successfully.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    sample_input_path = "data/processed/sample_batch_input.csv"
    sample_output_path = "reports/sample_batch_predictions.csv"

    predict_from_csv(sample_input_path, sample_output_path)