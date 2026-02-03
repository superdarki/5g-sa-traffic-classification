import argparse
import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    classification_report,  # type: ignore
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_curve,
    auc,
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
    all_raw_packets: list[pd.DataFrame] = []
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
    eval_df_raw: pd.DataFrame = pd.concat(all_raw_packets, ignore_index=True)
    eval_df_raw = eval_df_raw.sort_values("timestamp").reset_index(drop=True)
    print(f"Total raw packets combined for evaluation: {len(eval_df_raw)}")

    print("\n--- Assigning traffic labels based on UE ID lists ---")
    ue_id_to_traffic_map: dict[int, str] = {}
    for ue_id in args.embb_ue:
        ue_id_to_traffic_map[ue_id] = "eMBB"
    for ue_id in args.urllc_ue:
        ue_id_to_traffic_map[ue_id] = "URLLC"

    eval_df_raw["traffic_type"] = eval_df_raw["ue_id"].map(ue_id_to_traffic_map)

    initial_packet_count = len(eval_df_raw)
    eval_df_raw.dropna(subset=["traffic_type"], inplace=True)  # type: ignore
    kept_ues = list(ue_id_to_traffic_map.keys())
    print(f"Kept {len(eval_df_raw)} packets from specified UEs: {sorted(kept_ues)}.")
    print(
        f"Filtered out {initial_packet_count - len(eval_df_raw)} packets from other UEs."
    )

    # --- 3. Pre-process the Data ---
    initial_count = len(eval_df_raw)
    if "harq" in eval_df_raw.columns:
        eval_df_filtered = eval_df_raw.query("harq != -1").reset_index(drop=True)  # type: ignore
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
    engineered_df.columns = [
        "".join(c if c.isalnum() else "_" for c in str(col))
        for col in engineered_df.columns
    ]

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
        classification_report(y_true_encoded, y_pred_encoded, target_names=le.classes_)  # type: ignore
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
        print(misclassified_df.to_string())  # type: ignore

    # Print all packets if --verbose is used
    if args.verbose:
        print("\n--- Detailed Packet-by-Packet Classification Results (--verbose) ---")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(results_df)

    # Diagnostics: Confusion Matrix, Feature Importance, ROC curves (if binary)
    print("\n--- Visual Diagnostics ---")
    class_names = le.classes_
    ncols = 3 if len(class_names) == 2 else 2
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(9 * ncols, 7))  # type: ignore
    axes = np.atleast_1d(axes).ravel()

    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[0], cmap="Blues")  # type: ignore
    axes[0].set_title("Confusion Matrix")

    lgb.plot_importance(model, ax=axes[1], max_num_features=20, height=0.7)  # type: ignore
    axes[1].set_title("Top 20 Feature Importances")

    if len(class_names) == 2:
        roc_ax = axes[2]
        class_probabilities = model.predict_proba(X_eval)
        for class_name in class_names:
            encoded_label = le.transform([class_name])[0]
            class_index = list(model.classes_).index(encoded_label)
            y_scores = class_probabilities[:, class_index]
            fpr, tpr, _ = roc_curve(y_true_encoded, y_scores, pos_label=encoded_label)
            roc_auc = auc(fpr, tpr)
            roc_ax.plot(fpr, tpr, label=f"{class_name} AUC = {roc_auc:.3f}")
        roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        roc_ax.set_xlim([0.0, 1.0])
        roc_ax.set_ylim([0.0, 1.05])
        roc_ax.set_xlabel("False Positive Rate")
        roc_ax.set_ylabel("True Positive Rate")
        roc_ax.set_title("ROC Curve")
        roc_ax.legend(loc="lower right")
    else:
        print("Skipping ROC curve plot because there are more than two classes.")

    plt.tight_layout()
    plt.savefig("evaluation_diagnostics.png", dpi=300)  # type: ignore
    print("Saved diagnostic plots to 'evaluation_diagnostics.png'.")
    plt.show()  # type: ignore


if __name__ == "__main__":
    main()
