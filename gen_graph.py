import argparse
import os

import pandas as pd
from traffic_utils import parse_log_file


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
    args = parser.parse_args()

    # --- 1. Argument Validation ---
    embb_ues = set(args.embb_ue)
    urllc_ues = set(args.urllc_ue)
    if embb_ues.intersection(urllc_ues):
        print(f"Error: The same UE ID cannot be used for both eMBB and URLLC traffic.")
        print(f"Overlap found: {embb_ues.intersection(urllc_ues)}")
        return

    # --- 2. Data Loading and Combining ---
    all_raw_packets: list[pd.DataFrame] = []
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
    ue_id_to_traffic_map: dict[int, str] = {}
    for ue_id in args.embb_ue:
        ue_id_to_traffic_map[ue_id] = "eMBB"
    for ue_id in args.urllc_ue:
        ue_id_to_traffic_map[ue_id] = "URLLC"

    master_raw_df["traffic_type"] = master_raw_df["ue_id"].map(ue_id_to_traffic_map)

    initial_packet_count = len(master_raw_df)
    master_raw_df.dropna(subset=["traffic_type"], inplace=True)  # type: ignore
    kept_ues = list(ue_id_to_traffic_map.keys())
    print(f"Using {len(master_raw_df)} packets from specified UEs: {sorted(kept_ues)}.")
    print(
        f"Filtered out {initial_packet_count - len(master_raw_df)} packets from other UEs."
    )

    initial_count = len(master_raw_df)
    if "harq" in master_raw_df.columns:
        master_raw_df = master_raw_df.query("harq != -1").reset_index(drop=True)  # type: ignore
    filtered_count = len(master_raw_df)
    print(
        f"\n--- Filtering out {initial_count - filtered_count} system information packets. ---"
    )
    print(f"Remaining packets for training: {filtered_count}")

    if master_raw_df.empty:
        print("No data remains after filtering. Exiting.")
        return

    master_raw_df[
        (master_raw_df["traffic_type"] == "URLLC")
        & (master_raw_df["tb_len"].notna())
        & (master_raw_df["tb_len"] != "")
    ][["timestamp", "tb_len"]].to_csv("urllc.csv", index=False)
    master_raw_df[
        (master_raw_df["traffic_type"] == "eMBB")
        & (master_raw_df["tb_len"].notna())
        & (master_raw_df["tb_len"] != "")
    ][["timestamp", "tb_len"]].to_csv("embb.csv", index=False)


if __name__ == "__main__":
    main()
