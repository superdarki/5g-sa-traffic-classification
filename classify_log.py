import os
from pathlib import Path
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
from typing import Any

from traffic_utils import engineer_contextual_packet_features

warnings.filterwarnings("ignore")


def file_path(string: str | Path) -> str:
    path = (
        str(string)
        if os.path.isabs(string)
        else os.path.join(os.path.abspath(os.path.curdir), string)
    )
    if os.path.isfile(path):
        return path
    else:
        raise FileNotFoundError(path)


def main():
    """Main function to load model and classify a log file on a per-packet basis."""
    parser = argparse.ArgumentParser(
        description="Classify packets in a gNB log file as eMBB or URLLC."
    )
    parser.add_argument(
        "log_file",
        type=file_path,
        help="Path to the normalized CSV log file to classify.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="traffic_mixed_classifier.joblib",
        help="Path to the saved model bundle.",
    )
    args = parser.parse_args()

    # --- 1. Load the Model and Artifacts ---
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at '{args.model}'")
        print("Please run train_model.py first to create the model file.")
        sys.exit(1)

    print(f"Loading model from '{args.model}'...")
    model_bundle: dict[str, Any] = joblib.load(args.model)
    model = model_bundle["model"]
    encoder = model_bundle["encoder"]
    training_columns = model_bundle["columns"]

    window_size = model_bundle["window_size"]
    print(f"Using contextual window size from trained model: {window_size}")

    # --- 2. Process the Input Log File ---
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found at '{args.log_file}'")
        sys.exit(1)

    normalized_df = pd.read_csv(args.log_file)  # type: ignore
    if "timestamp" in normalized_df.columns:
        normalized_df["timestamp"] = pd.to_datetime(
            normalized_df["timestamp"], errors="coerce"
        )
    if normalized_df.empty:
        print("No parsable data found in the log file.")
        sys.exit(0)
    print(f"\nParsed {len(normalized_df)} total packets from log file.")

    required_columns = {"timestamp", "type", "direction"}
    missing_columns = required_columns.difference(normalized_df.columns)
    if missing_columns:
        print(f"Error: Normalized logs are missing columns: {sorted(missing_columns)}")
        sys.exit(1)

    # Engineer features only on the relevant (non-system) packets
    if "traffic_type" in normalized_df.columns:
        normalized_df = normalized_df.drop(columns=["traffic_type"])
    original_timestamps = normalized_df["timestamp"]
    features_df = engineer_contextual_packet_features(
        normalized_df, window_size=window_size
    )

    # --- 3. Align Columns with Training Data ---
    print("Aligning features with model's training columns...")
    # Ensure column names are sanitized exactly as they were for training
    features_df.columns = [
        "".join(c if c.isalnum() else "_" for c in str(x)) for x in features_df.columns
    ]

    # Reindex to match the training columns, filling missing ones with 0
    # This handles cases where a log might be missing certain packet types (e.g., no PUSCH)
    features_df_aligned = features_df.reindex(columns=training_columns, fill_value=0)

    # --- 4. Make Predictions ---
    print("Generating model predictions for each packet...")
    predictions_encoded = model.predict(features_df_aligned)
    predictions_proba = model.predict_proba(features_df_aligned)

    predictions_text = encoder.inverse_transform(predictions_encoded)

    # --- 5. Display Results ---
    results = pd.DataFrame(
        {
            "timestamp": original_timestamps,
            "prediction": predictions_text,
            "confidence": np.max(predictions_proba, axis=1) * 100,
        }
    )

    print("\n--- Per-Packet Classification Results (excluding SYSTEM packets) ---")
    for _, row in results.iterrows():
        ts_str = row["timestamp"].strftime("%H:%M:%S.%f")[:-3]
        print(
            f"Packet @ {ts_str} | Prediction: {row['prediction']:<5} | Confidence: {row['confidence']:.1f}%"
        )

    print("\n--- Summary (of classified packets) ---")
    if not results.empty:
        summary = results["prediction"].value_counts(normalize=True) * 100
        dominant_class = summary.idxmax()
        print(
            f"The log is predominantly classified as: {dominant_class} ({summary.max():.1f}% of packets)"
        )
        for traffic_type, percentage in summary.items():
            print(f"- {traffic_type}: {percentage:.1f}%")
    else:
        print("No predictions were made.")


if __name__ == "__main__":
    main()
