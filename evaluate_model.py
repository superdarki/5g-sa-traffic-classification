import argparse
import os
import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
import matplotlib.pyplot as plt
import warnings

# traffic_utils.py is unchanged and works as is
from traffic_utils import engineer_contextual_packet_features, parse_log_file

warnings.filterwarnings("ignore")


def main():
    """Main function to evaluate a trained classifier on one or more mixed-traffic logs."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained traffic classifier on one or more log files containing multiple UEs."
    )
    parser.add_argument(
        "model_file",
        type=str,
        help="Path to the saved .joblib model bundle (e.g., 'traffic_classifier.joblib').",
    )
    # --- MODIFIED ARGUMENT TO ACCEPT MULTIPLE FILES ---
    parser.add_argument(
        "logfiles",
        type=str,
        nargs="+",  # Accept one or more file paths
        help="Path to one or more mixed-traffic log files for evaluation.",
    )
    parser.add_argument(
        "--embb-ue",
        type=int,
        nargs="+",
        required=True,
        help="One or more UE IDs for eMBB traffic (e.g., --embb-ue 9 10).",
    )
    parser.add_argument(
        "--urllc-ue",
        type=int,
        nargs="+",
        required=True,
        help="One or more UE IDs for URLLC traffic (e.g., --urllc-ue 1 2).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the classification result for every single packet.",
    )
    args = parser.parse_args()

    # --- VALIDATION ---
    embb_ues = set(args.embb_ue)
    urllc_ues = set(args.urllc_ue)

    if embb_ues.intersection(urllc_ues):
        print("Error: The same UE ID cannot be used for both eMBB and URLLC traffic.")
        print(f"Overlap found: {embb_ues.intersection(urllc_ues)}")
        return

    # --- 1. Load the Model and Artifacts ---
    try:
        print(f"--- Loading model bundle from: {args.model_file} ---")
        model_bundle = joblib.load(args.model_file)
        model = model_bundle["model"]
        le = model_bundle["encoder"]
        model_columns = model_bundle["columns"]
        window_size = model_bundle["window_size"]
    except FileNotFoundError:
        print(f"Error: Model file not found at '{args.model_file}'")
        return
    except KeyError as e:
        print(f"Error: Model bundle is missing a required key: {e}")
        return

    # --- 2. Parse Log Files, Combine, and Assign True Labels ---
    all_raw_packets = []
    print(f"\n--- Parsing {len(args.logfiles)} evaluation log file(s) ---")
    for logfile in args.logfiles:
        if not os.path.exists(logfile):
            print(f"Warning: Log file not found at '{logfile}'. Skipping.")
            continue
        print(f"Reading: {logfile}")
        df = parse_log_file(logfile)
        if not df.empty:
            all_raw_packets.append(df)

    if not all_raw_packets:
        print("Error: No valid log files were parsed or files were empty. Exiting.")
        return

    print("\n--- Combining and sorting all logs into a single timeline ---")
    eval_df_raw = pd.concat(all_raw_packets, ignore_index=True)
    eval_df_raw = eval_df_raw.sort_values("timestamp").reset_index(drop=True)
    print(f"Total raw packets combined for evaluation: {len(eval_df_raw)}")

    print("\n--- Assigning traffic labels based on UE ID lists ---")
    ue_id_to_traffic_map = {}
    for ue_id in args.embb_ue:
        ue_id_to_traffic_map[ue_id] = "eMBB"
    for ue_id in args.urllc_ue:
        ue_id_to_traffic_map[ue_id] = "URLLC"

    eval_df_raw["traffic_type"] = eval_df_raw["ue_id"].map(ue_id_to_traffic_map)

    initial_packet_count = len(eval_df_raw)
    eval_df_raw.dropna(subset=["traffic_type"], inplace=True)
    kept_ues = list(ue_id_to_traffic_map.keys())
    print(f"Kept {len(eval_df_raw)} packets from specified UEs: {sorted(kept_ues)}.")
    print(
        f"Filtered out {initial_packet_count - len(eval_df_raw)} packets from other UEs."
    )

    # --- 3. Pre-process the Data ---
    initial_count = len(eval_df_raw)
    if "harq" in eval_df_raw.columns:
        eval_df_filtered = eval_df_raw.query("harq != -1").reset_index(drop=True)
    else:
        eval_df_filtered = eval_df_raw.reset_index(drop=True)

    filtered_count = len(eval_df_filtered)
    print(f"Filtered out {initial_count - filtered_count} system info packets.")
    print(f"Remaining packets for evaluation: {filtered_count}")

    if eval_df_filtered.empty:
        print("No data remains after filtering. Cannot evaluate. Exiting.")
        return

    true_labels = eval_df_filtered["traffic_type"]
    features_to_engineer = eval_df_filtered.drop(columns=["traffic_type"])

    # --- 4. Engineer Features ---
    print("\n--- Engineering contextual features ---")
    engineered_df = engineer_contextual_packet_features(
        features_to_engineer, window_size=window_size
    )

    # --- 5. Prepare Data for Prediction ---
    X_eval = engineered_df.reindex(columns=model_columns, fill_value=0)
    X_eval.columns = [
        "".join(c if c.isalnum() else "_" for c in str(x)) for x in X_eval.columns
    ]
    y_true_encoded = le.transform(true_labels)

    # --- 6. Make Predictions ---
    print("\n--- Making predictions on the evaluation data ---")
    y_pred_encoded = model.predict(X_eval)

    # --- 7. Report Results ---
    print("\n--- Evaluation Results ---")
    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(
        classification_report(y_true_encoded, y_pred_encoded, target_names=le.classes_)
    )

    # Create a detailed results DataFrame
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    results_df = eval_df_filtered[
        ["timestamp", "ue_id", "type", "direction", "tb_len", "prbs", "snr"]
    ].copy()
    results_df["true_type"] = true_labels.values
    results_df["predicted_type"] = y_pred_labels

    # Print misclassified packets
    misclassified_df = results_df[
        results_df["true_type"] != results_df["predicted_type"]
    ]
    print(
        f"\n--- {len(misclassified_df)}/{len(results_df)} ({len(misclassified_df) / len(results_df) * 100:.2f}%) Misclassified Packets ---"
    )
    if misclassified_df.empty:
        print("No packets were misclassified. Excellent!")
    elif args.verbose:
        print(misclassified_df.to_string())

    # Print all packets if --verbose is used
    if args.verbose:
        print("\n--- Detailed Packet-by-Packet Classification Results (--verbose) ---")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(results_df)

    # Display Confusion Matrix
    print("\nSaving Confusion Matrix to 'confusion_matrix_evaluation.png'...")
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix for Evaluation Data")
    plt.savefig("confusion_matrix_evaluation.png")
    print("Done.")
    plt.show()


if __name__ == "__main__":
    main()
