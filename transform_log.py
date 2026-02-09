import argparse
import os
import sys

from traffic_utils import normalize_log_file, STANDARD_COLUMNS


def default_output_path(input_path: str, output_dir: str | None) -> str:
    base = os.path.basename(input_path)
    name, _ = os.path.splitext(base)
    directory = output_dir or os.path.dirname(input_path)
    return os.path.join(directory, f"{name}.normalized.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize logs into a common CSV format for downstream tools."
    )
    parser.add_argument(
        "logfiles",
        nargs="+",
        help="One or more raw log files (Amarisoft log or RS ROMES CSV).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (only valid when a single input file is provided).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write normalized files (default: alongside inputs).",
    )
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=["amarisoft", "rs_romes", "romes", "normalized", "norm"],
        help="Input log format (required).",
    )
    parser.add_argument(
        "--embb-ue",
        type=int,
        nargs="+",
        default=None,
        help="UE IDs to label as eMBB (used for Amarisoft logs).",
    )
    parser.add_argument(
        "--urllc-ue",
        type=int,
        nargs="+",
        default=None,
        help="UE IDs to label as URLLC (used for Amarisoft logs).",
    )
    parser.add_argument(
        "--embb-time",
        type=str,
        nargs="+",
        default=None,
        help="One or more inclusive time ranges (HH:MM:SS.mmm-HH:MM:SS.mmm) for eMBB.",
    )
    parser.add_argument(
        "--urllc-time",
        type=str,
        nargs="+",
        default=None,
        help="One or more inclusive time ranges (HH:MM:SS.mmm-HH:MM:SS.mmm) for URLLC.",
    )

    args = parser.parse_args()

    if args.output and len(args.logfiles) > 1:
        print("Error: --output can only be used with a single input file.")
        sys.exit(1)

    if args.output and args.output_dir:
        print("Error: Use either --output or --output-dir, not both.")
        sys.exit(1)

    format_map = {
        "amarisoft": "amarisoft",
        "rs_romes": "rs_romes",
        "romes": "rs_romes",
        "normalized": "normalized",
        "norm": "normalized",
    }
    log_format = format_map[args.format]

    if log_format == "rs_romes" and (args.embb_ue or args.urllc_ue):
        print("Error: RS ROMES logs must be labeled with time ranges, not UE IDs.")
        sys.exit(1)
    if log_format == "amarisoft" and (args.embb_time or args.urllc_time):
        print("Error: Amarisoft logs must be labeled with UE IDs, not time ranges.")
        sys.exit(1)
    if log_format == "normalized" and (
        args.embb_ue or args.urllc_ue or args.embb_time or args.urllc_time
    ):
        print("Error: Normalized logs should already include traffic_type labels.")
        sys.exit(1)

    for logfile in args.logfiles:
        if not os.path.exists(logfile):
            print(f"Warning: Log file not found at '{logfile}'. Skipping.")
            continue

        normalized_df = normalize_log_file(
            logfile,
            log_format=log_format,
            embb_ues=set(args.embb_ue) if args.embb_ue else None,
            urllc_ues=set(args.urllc_ue) if args.urllc_ue else None,
            embb_time_ranges=args.embb_time,
            urllc_time_ranges=args.urllc_time,
        )
        if normalized_df.empty:
            print(f"Warning: No parsable data found in '{logfile}'. Skipping.")
            continue

        if (
            "traffic_type" not in normalized_df.columns
            or normalized_df["traffic_type"].isna().all()
        ):
            print(
                f"Error: No traffic labels were assigned for '{logfile}'. Provide --embb-ue/--urllc-ue or --embb-time/--urllc-time."
            )
            sys.exit(1)

        normalized_df = normalized_df.dropna(subset=["traffic_type"]).reset_index(  # type: ignore
            drop=True
        )

        output_path = args.output or default_output_path(logfile, args.output_dir)
        normalized_df = normalized_df[STANDARD_COLUMNS]
        normalized_df.to_csv(output_path, index=False)
        print(f"Wrote normalized log to '{output_path}' ({len(normalized_df)} rows).")


if __name__ == "__main__":
    main()
