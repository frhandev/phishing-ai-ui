from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.model.inference import predict_from_url


def predict_urls_from_csv(
    input_csv_path: str,
    output_csv_path: str,
    url_column: str = "URL"
) -> None:
    input_path = Path(input_csv_path)
    output_path = Path(output_csv_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_path}")

    df = pd.read_csv(input_path)

    if url_column not in df.columns:
        raise ValueError(
            f"Input CSV must contain a '{url_column}' column. "
            f"Available columns: {df.columns.tolist()}"
        )

    prediction_rows = []

    for url in df[url_column]:
        prediction_result = predict_from_url(str(url)).iloc[0].to_dict()
        prediction_rows.append(prediction_result)

    predictions_df = pd.DataFrame(prediction_rows)

    final_df = pd.concat(
        [df.reset_index(drop=True), predictions_df.reset_index(drop=True)],
        axis=1
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print("Raw URL batch prediction completed successfully.")
    print(f"Input file : {input_path}")
    print(f"Output file: {output_path}")
    print(f"Rows processed: {len(final_df)}")


if __name__ == "__main__":
    sample_input_path = "data/processed/sample_raw_urls.csv"
    sample_output_path = "reports/sample_raw_url_predictions.csv"

    predict_urls_from_csv(
        input_csv_path=sample_input_path,
        output_csv_path=sample_output_path,
        url_column="URL"
    )