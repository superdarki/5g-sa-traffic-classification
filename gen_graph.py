import argparse
import os

import pandas as pd


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
        help="Path to one or more normalized CSV log files for plotting.",
    )
    args = parser.parse_args()

    # --- 1. Data Loading and Combining ---
    all_raw_packets: list[pd.DataFrame] = []
    print(f"--- Parsing {len(args.logfiles)} log file(s) ---")
    for logfile in args.logfiles:
        if not os.path.exists(logfile):
            print(f"Warning: Log file not found at '{logfile}'. Skipping.")
            continue
        print(f"Reading: {logfile}")
        df = pd.read_csv(logfile)  # type: ignore
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
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

    # --- 2. Validate Normalized Data ---
    if "traffic_type" not in master_raw_df.columns:
        print("Error: Normalized logs must include a 'traffic_type' column.")
        return

    initial_packet_count = len(master_raw_df)
    master_raw_df.dropna(subset=["traffic_type"], inplace=True)  # type: ignore
    print(f"Using {len(master_raw_df)} labeled packets for plotting.")
    print(
        f"Filtered out {initial_packet_count - len(master_raw_df)} unlabeled packets."
    )

    print(
        "\n--- Normalization complete (system information packets with harq=-1 already filtered). ---"
    )
    print(f"Remaining packets for plotting: {len(master_raw_df)}")

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
