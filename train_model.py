# File: train_model.py

import argparse
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import warnings
import joblib

# Import parsing and feature-engineering helpers from the local module.
from traffic_utils import parse_log_file, engineer_contextual_packet_features

warnings.filterwarnings("ignore")


def main():
    """Main function to handle training or retraining a classifier from one or more log files."""
    parser = argparse.ArgumentParser(
        description="Train or retrain a LightGBM model to classify traffic packets from one or more log files."
    )

    # --- Command-line Arguments ---
    parser.add_argument(
        "logfiles",
        type=str,
        nargs="+",
        help="Path to one or more mixed-traffic log files for training (e.g., log1.txt log2.txt).",
    )
    parser.add_argument(
        "--embb-ue",
        type=int,
        nargs="+",
        required=True,
        help="One or more UE IDs to be labeled as eMBB traffic (e.g., --embb-ue 9 10).",
    )
    parser.add_argument(
        "--urllc-ue",
        type=int,
        nargs="+",
        required=True,
        help="One or more UE IDs to be labeled as URLLC traffic (e.g., --urllc-ue 1 2).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="traffic_classifier.joblib",
        help="Path for the saved model bundle (default: traffic_classifier.joblib).",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="The rolling window size for contextual features (default: 5).",
    )
    parser.add_argument(
        "--retrain-from",
        type=str,
        default=None,
        help="Path to an existing model bundle to continue training from (enables incremental training).",
    )
    args = parser.parse_args()

    # --- 1. Argument Validation ---
    embb_ues = set(args.embb_ue)
    urllc_ues = set(args.urllc_ue)
    if embb_ues.intersection(urllc_ues):
        print(f"Error: The same UE ID cannot be used for both eMBB and URLLC traffic.")
        print(f"Overlap found: {embb_ues.intersection(urllc_ues)}")
        return

    # --- 2. Data Loading and Combining ---
    all_raw_packets = []
    print(f"--- Parsing {len(args.logfiles)} log file(s) ---")
    for logfile in args.logfiles:
        if not os.path.exists(logfile):
            print(f"Warning: Log file not found at '{logfile}'. Skipping.")
            continue
        print(f"Reading: {logfile}")
        df = parse_log_file(logfile)
        if not df.empty:
            all_raw_packets.append(df)

    if not all_raw_packets:
        print("Error: No valid log files were parsed. Exiting.")
        return

    print("\n--- Combining and sorting all logs into a single timeline ---")
    master_raw_df = pd.concat(all_raw_packets, ignore_index=True)
    master_raw_df["timestamp"] = pd.to_datetime(master_raw_df["timestamp"])
    master_raw_df = master_raw_df.sort_values("timestamp").reset_index(drop=True)
    print(f"Total raw packets combined from all files: {len(master_raw_df)}")

    # --- 3. Labeling and Pre-processing ---
    print("\n--- Assigning traffic labels based on UE ID lists ---")
    ue_id_to_traffic_map = {}
    for ue_id in args.embb_ue:
        ue_id_to_traffic_map[ue_id] = "eMBB"
    for ue_id in args.urllc_ue:
        ue_id_to_traffic_map[ue_id] = "URLLC"

    master_raw_df["traffic_type"] = master_raw_df["ue_id"].map(ue_id_to_traffic_map)

    initial_packet_count = len(master_raw_df)
    master_raw_df.dropna(subset=["traffic_type"], inplace=True)
    kept_ues = list(ue_id_to_traffic_map.keys())
    print(f"Using {len(master_raw_df)} packets from specified UEs: {sorted(kept_ues)}.")
    print(
        f"Filtered out {initial_packet_count - len(master_raw_df)} packets from other UEs."
    )

    initial_count = len(master_raw_df)
    if "harq" in master_raw_df.columns:
        master_raw_df = master_raw_df.query("harq != -1").reset_index(drop=True)
    filtered_count = len(master_raw_df)
    print(
        f"\n--- Filtering out {initial_count - filtered_count} system information packets. ---"
    )
    print(f"Remaining packets for training: {filtered_count}")

    if master_raw_df.empty:
        print("No training data remains after filtering. Exiting.")
        return

    # --- 4. Feature Engineering ---
    print(f"\n--- Engineering contextual features (window_size={args.window_size}) ---")
    labels = master_raw_df["traffic_type"]
    features_to_engineer = master_raw_df.drop(columns=["traffic_type"])

    master_features_df = engineer_contextual_packet_features(
        features_to_engineer, window_size=args.window_size
    )
    master_features_df["traffic_type"] = labels.values

    print("\n--- Feature Engineering Complete ---")
    print(
        f"Total samples: {len(master_features_df)}, Total features: {len(master_features_df.columns) - 1}"
    )

    # --- 5. Label Encoding and Model Training ---
    le = LabelEncoder()
    initial_model = None
    model_columns = None

    if args.retrain_from:
        try:
            print(
                f"--- Loading existing model bundle from '{args.retrain_from}' for incremental training ---"
            )
            old_model_bundle = joblib.load(args.retrain_from)
            initial_model = old_model_bundle["model"]
            le = old_model_bundle["encoder"]  # IMPORTANT: Reuse the old encoder
            model_columns = old_model_bundle["columns"]
            print("Successfully loaded model and encoder for retraining.")
        except Exception as e:
            print(
                f"Warning: Could not load model from '{args.retrain_from}'. Training a new model from scratch. Error: {e}"
            )
            initial_model = None

    if initial_model:
        # Reuse the existing encoder so label indices stay consistent during retraining.
        master_features_df["traffic_type_encoded"] = le.transform(
            master_features_df["traffic_type"]
        )
    else:
        # Fit a fresh encoder when starting from scratch to derive the label mapping.
        print("--- Training a new model from scratch ---")
        master_features_df["traffic_type_encoded"] = le.fit_transform(
            master_features_df["traffic_type"]
        )

    class_names = le.classes_
    print(
        f"\nClass mapping: {list(zip(class_names, np.array(le.transform(class_names)).tolist()))}"
    )

    X = master_features_df.drop(columns=["traffic_type", "traffic_type_encoded"])
    y = master_features_df["traffic_type_encoded"]

    # Replace non-alphanumeric characters so LightGBM receives valid column names.
    X.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X.columns]

    # Align to the feature ordering expected by the previously trained model, filling missing features with 0.
    if model_columns:
        X = X.reindex(columns=model_columns, fill_value=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\nTraining on {len(X_train)} packets, testing on {len(X_test)} packets.")

    print("\n--- Training LightGBM Model ---")
    lgbm = lgb.LGBMClassifier(objective="binary", random_state=42, learning_rate=0.01)

    # Pass the previous model (if available) so LightGBM can warm start training.
    lgbm.fit(X_train, y_train, init_model=initial_model)

    # --- 6. Evaluation ---
    print("\n--- Model Evaluation ---")
    y_pred = np.array(lgbm.predict(X_test))
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- 7. Save the Model ---
    model_bundle = {
        "model": lgbm,
        "encoder": le,
        "columns": X.columns.tolist(),
        "window_size": args.window_size,
    }
    joblib.dump(model_bundle, args.output)
    print(f"\n--- Model and artifacts saved successfully to '{args.output}' ---")


if __name__ == "__main__":
    main()
