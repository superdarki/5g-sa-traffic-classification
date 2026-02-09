import csv
import os
import re
from datetime import date, datetime
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm


STANDARD_COLUMNS = [
    "timestamp",
    "traffic_type",
    "type",
    "direction",
    "harq",
    "prbs",
    "tb_len",
    "mod",
    "rv_idx",
    "cr",
    "retx",
    "crc_ok",
    "snr",
    "mcs",
    "format",
    "total_len",
]

NUMERIC_COLUMNS = [
    "harq",
    "prbs",
    "tb_len",
    "mod",
    "rv_idx",
    "cr",
    "retx",
    "crc_ok",
    "snr",
    "mcs",
    "format",
    "total_len",
]


def parse_phy_mac_log_line(line: str) -> Optional[dict[str, int | float | str]]:
    """
    Parses a single PHY or MAC log line and extracts relevant features.
    This version is compatible with both gNB and UE log formats by handling
    optional fields (crc, snr) and field name variations (mcs/mcs1).
    """
    # --- PUSCH (Uplink Data) ---
    pusch_match = re.search(
        r"\[PHY\].*?UL.*?PUSCH:?\s+.*?harq=([\w\d]+).*?prb=([-\d:;,]+).*?tb_len=(\d+).*?mod=(\d+).*?rv_idx=(\d+).*?cr=([\d\.]+).*?retx=(\d+)"
        r"(?:.*?crc=(\w+))?(?:.*?snr=([\d\.-]+))?",
        line,
    )
    if pusch_match:
        harq_val = pusch_match.group(1)
        harq_id = -1 if harq_val == "si" else int(harq_val)

        prbs = 0
        prb_groups = pusch_match.group(2).split(",")
        for group in prb_groups:
            prb_field = group.split(":")
            prbs += (
                int(prb_field[-1]) - int(prb_field[0]) + 1 if len(prb_field) > 1 else 1
            )

        crc_val = pusch_match.group(8)
        snr_val = pusch_match.group(9)

        return {
            "type": "PUSCH",
            "direction": "UL",
            "harq": harq_id,
            "prbs": prbs,
            "tb_len": int(pusch_match.group(3)),
            "mod": int(pusch_match.group(4)),
            "rv_idx": int(pusch_match.group(5)),
            "cr": float(pusch_match.group(6)),
            "retx": int(pusch_match.group(7)),
            "crc_ok": 1 if crc_val == "OK" else 0 if crc_val else 1,
            "snr": float(snr_val) if snr_val else np.nan,
        }

    # --- PDSCH (Downlink Data) ---
    pdsch_match = re.search(
        r"\[PHY\].*?DL.*?PDSCH:?\s+.*?harq=([\w\d]+).*?prb=([-\d:;,]+).*?tb_len=(\d+).*?mod=(\d+).*?rv_idx=(\d+).*?cr=([\d\.]+).*?retx=(\d+)"
        r"(?:.*?crc=(\w+))?(?:.*?snr=([\d\.-]+))?",
        line,
    )
    if pdsch_match:
        harq_val = pdsch_match.group(1)
        harq_id = -1 if harq_val == "si" else int(harq_val)

        prbs = 0
        prb_groups = pdsch_match.group(2).split(",")
        for group in prb_groups:
            prb_field = group.split(":")
            prbs += (
                int(prb_field[-1]) - int(prb_field[0]) + 1 if len(prb_field) > 1 else 1
            )

        crc_val = pdsch_match.group(8)
        snr_val = pdsch_match.group(9)

        return {
            "type": "PDSCH",
            "direction": "DL",
            "harq": harq_id,
            "prbs": prbs,
            "tb_len": int(pdsch_match.group(3)),
            "mod": int(pdsch_match.group(4)),
            "rv_idx": int(pdsch_match.group(5)),
            "cr": float(pdsch_match.group(6)),
            "retx": int(pdsch_match.group(7)),
            "crc_ok": 1 if crc_val == "OK" else 0 if crc_val else 1,
            "snr": float(snr_val) if snr_val else np.nan,
        }

    # --- PDCCH (Downlink Control) ---
    pdcch_match = re.search(r"\[PHY\].*?DL.*?PDCCH:?\s+.*?mcs1?=(\d+)", line)
    if pdcch_match:
        return {
            "type": "PDCCH",
            "direction": "DL",
            "mcs": int(pdcch_match.group(1)),
        }

    # --- PUCCH (Uplink Control) ---
    pucch_match = re.search(
        r"\[PHY\].*?UL.*?PUCCH:?\s+.*?format=(\d+)(?:.*?snr=([\d\.-]+))?", line
    )
    if pucch_match:
        snr_val = pucch_match.group(2)
        return {
            "type": "PUCCH",
            "direction": "UL",
            "format": int(pucch_match.group(1)),
            "snr": float(snr_val) if snr_val else np.nan,
        }

    # --- MAC UL ---
    mac_ul_match = re.search(r"\[MAC\].*?UL", line)
    if mac_ul_match:
        lengths = [int(l) for l in re.findall(r"len=(\d+)", line)]
        if lengths:
            return {
                "type": "MAC",
                "direction": "UL",
                "total_len": sum(lengths),
            }

    # --- MAC DL ---
    mac_dl_match = re.search(r"\[MAC\].*?DL", line)
    if mac_dl_match:
        lengths = [int(l) for l in re.findall(r"len=(\d+)", line)]
        if lengths:
            return {
                "type": "MAC",
                "direction": "DL",
                "total_len": sum(lengths),
            }

    return None


def _extract_date_from_filename(file_path: str) -> Optional[date]:
    basename = os.path.basename(file_path)
    match = re.match(r"(\d{8})-\d{6}", basename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _parse_amarisoft_log_file(file_path: str) -> pd.DataFrame:
    """
    Reads an Amarisoft log file, extracts UE ID, and returns a DataFrame of parsed data.
    """
    parsed_data: list[dict[str, int | float | str]] = []
    # This regex captures timestamp, layer, direction, and UE ID from the log line prefix
    line_prefix_re = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d{3})\s+\[(\w+)\]\s+(\S+)\s+(\d{4})"
    )

    with open(file_path, "r") as f, tqdm(
        total=os.path.getsize(file_path),
        unit="B",
        unit_scale=True,
        desc=f"Parsing {os.path.basename(file_path)}",
    ) as pbar:
        for line in f:
            pbar.update(len(line.encode("utf-8")))

            prefix_match = line_prefix_re.match(line)
            if not prefix_match:
                continue

            timestamp = prefix_match.group(1)
            ue_id = int(prefix_match.group(4))

            data = parse_phy_mac_log_line(line)
            if data:
                data["timestamp"] = timestamp
                data["ue_id"] = ue_id  # Add the extracted UE ID
                data["source_format"] = "amarisoft"
                parsed_data.append(data)

    if not parsed_data:
        return pd.DataFrame()

    df = pd.DataFrame(parsed_data)
    date_from_name = _extract_date_from_filename(file_path)
    if date_from_name is not None:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"].apply(lambda t: f"{date_from_name} {t}"),
            format="%Y-%m-%d %H:%M:%S.%f",
            errors="coerce",
        )
    else:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%H:%M:%S.%f", errors="coerce"
        )

    df.sort_values("timestamp", inplace=True)
    return df.reset_index(drop=True)


def _modulation_to_order(value: str) -> Optional[int]:
    value = str(value).strip().upper()
    if not value:
        return None
    mapping = {
        "QPSK": 2,
        "16Q": 4,
        "64Q": 6,
        "256Q": 8,
        "1024Q": 10,
    }
    if value in mapping:
        return mapping[value]
    try:
        return int(value)
    except ValueError:
        return None


def _parse_rs_rome_csv(file_path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(
        file_path,
        sep=";",
        dtype=str,
        keep_default_na=False,
        na_values=["", "NA", "N/A"],
        engine="python",
        on_bad_lines="skip",
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )
    if df_raw.empty:
        return pd.DataFrame()

    df_raw.columns = [c.strip() for c in df_raw.columns]

    if "Source" not in df_raw.columns:
        return pd.DataFrame()

    df_raw["Source"] = df_raw["Source"].astype(str).str.strip()

    def map_type_direction(source: str) -> Optional[tuple[str, str]]:
        src_upper = source.upper()
        if src_upper.startswith("PUSCH"):
            return ("PUSCH", "UL")
        if src_upper.startswith("PDSCH"):
            return ("PDSCH", "DL")
        if src_upper.startswith("PUCCH"):
            return ("PUCCH", "UL")
        if src_upper.startswith("PDCCH"):
            return ("PDCCH", "DL")
        if src_upper.startswith("MAC DCI"):
            return ("MAC", "DL")
        if src_upper.startswith("MAC"):
            direction = "UL" if "UL" in src_upper else "DL" if "DL" in src_upper else ""
            if direction:
                return ("MAC", direction)
        return None

    type_direction = df_raw["Source"].map(map_type_direction)
    df_raw = df_raw[type_direction.notna()].copy()
    if df_raw.empty:
        return pd.DataFrame()

    df_raw[["type", "direction"]] = pd.DataFrame(
        type_direction.dropna().tolist(), index=df_raw.index
    )

    df_raw["timestamp"] = pd.to_datetime(
        df_raw["Time"], format="%H:%M:%S.%f", errors="coerce"
    )

    df_raw["harq"] = pd.to_numeric(df_raw.get("HARQ"), errors="coerce")
    df_raw["mcs"] = pd.to_numeric(df_raw.get("MCS"), errors="coerce")
    df_raw["mod"] = df_raw.get("Mod", "").map(_modulation_to_order)
    df_raw["tb_len"] = pd.to_numeric(df_raw.get("TBS"), errors="coerce")
    df_raw["prbs"] = pd.to_numeric(df_raw.get("RB"), errors="coerce")
    df_raw["rv_idx"] = pd.to_numeric(df_raw.get("RV"), errors="coerce")

    retx_raw = df_raw.get("ReTx", "")
    retx_numeric = pd.to_numeric(retx_raw, errors="coerce")
    df_raw["retx"] = retx_numeric.fillna(
        retx_raw.astype(str).str.upper().eq("NEW").astype(int)
    )

    crc_raw = df_raw.get("CRC", "").astype(str).str.upper()
    df_raw["crc_ok"] = crc_raw.map({"OK": 1, "PASS": 1, "KO": 0, "FAIL": 0})

    df_raw["snr"] = pd.to_numeric(df_raw.get("RSRP/TxP"), errors="coerce")
    df_raw["format"] = pd.to_numeric(df_raw.get("Format"), errors="coerce")

    df_raw["total_len"] = pd.to_numeric(df_raw.get("TBS"), errors="coerce")
    df_raw["source_format"] = "rs_rome"

    df_raw = df_raw.dropna(subset=["timestamp", "type", "direction"])
    return df_raw.reset_index(drop=True)


def _ensure_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[STANDARD_COLUMNS]


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _parse_time_ranges(
    range_specs: list[str],
) -> list[tuple[datetime.time, datetime.time]]:
    ranges: list[tuple[datetime.time, datetime.time]] = []
    for spec in range_specs:
        if "-" not in spec:
            raise ValueError(
                f"Invalid time range '{spec}'. Expected format: HH:MM:SS.mmm-HH:MM:SS.mmm"
            )
        start_str, end_str = spec.split("-", 1)
        start_time = datetime.strptime(start_str.strip(), "%H:%M:%S.%f").time()
        end_time = datetime.strptime(end_str.strip(), "%H:%M:%S.%f").time()
        ranges.append((start_time, end_time))
    return ranges


def _label_by_time_ranges(
    df: pd.DataFrame,
    embb_ranges: list[tuple[datetime.time, datetime.time]],
    urllc_ranges: list[tuple[datetime.time, datetime.time]],
) -> pd.DataFrame:
    if df.empty:
        return df

    def in_ranges(
        ts: pd.Timestamp, ranges: list[tuple[datetime.time, datetime.time]]
    ) -> bool:
        t = ts.time()
        for start, end in ranges:
            if start <= t <= end:
                return True
        return False

    df["traffic_type"] = pd.NA
    if urllc_ranges:
        urllc_mask = df["timestamp"].apply(lambda ts: in_ranges(ts, urllc_ranges))
        df.loc[urllc_mask, "traffic_type"] = "URLLC"
    if embb_ranges:
        embb_mask = df["timestamp"].apply(lambda ts: in_ranges(ts, embb_ranges))
        df.loc[embb_mask, "traffic_type"] = "eMBB"
    return df


def normalize_log_file(
    file_path: str,
    log_format: str,
    embb_ues: Optional[set[int]] = None,
    urllc_ues: Optional[set[int]] = None,
    embb_time_ranges: Optional[list[str]] = None,
    urllc_time_ranges: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Normalize a log file (Amarisoft, RS ROME CSV, or normalized CSV) into a common schema.
    Filters out system-information packets (harq == -1) during normalization.
    """
    if log_format == "normalized":
        df = pd.read_csv(file_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif log_format == "rs_rome":
        df = _parse_rs_rome_csv(file_path)
        if df.empty:
            return df
    elif log_format == "amarisoft":
        df = _parse_amarisoft_log_file(file_path)
        if df.empty:
            return df
    else:
        raise ValueError(f"Unknown log format '{log_format}'.")

    if "traffic_type" not in df.columns or df["traffic_type"].isna().all():
        if embb_time_ranges or urllc_time_ranges:
            embb_ranges = _parse_time_ranges(embb_time_ranges or [])
            urllc_ranges = _parse_time_ranges(urllc_time_ranges or [])
            df = _label_by_time_ranges(df, embb_ranges, urllc_ranges)
        elif embb_ues or urllc_ues:
            embb_ues = embb_ues or set()
            urllc_ues = urllc_ues or set()
            if "ue_id" in df.columns:
                df["traffic_type"] = pd.NA
                df.loc[df["ue_id"].isin(embb_ues), "traffic_type"] = "eMBB"
                df.loc[df["ue_id"].isin(urllc_ues), "traffic_type"] = "URLLC"

    df = _ensure_standard_columns(df)
    df = _coerce_numeric_columns(df)

    if "harq" in df.columns:
        df = df[(df["harq"].isna()) | (df["harq"] != -1)].reset_index(drop=True)

    df = df.dropna(subset=["timestamp", "type", "direction"])
    return df.reset_index(drop=True)


# Backwards compatibility for existing imports.
def parse_log_file(file_path: str) -> pd.DataFrame:
    raise ValueError(
        "parse_log_file() is disabled. Use normalize_log_file(file_path, log_format=...) instead."
    )


def engineer_contextual_packet_features(df: pd.DataFrame, window_size: int = 5):
    """
    Engineers features for each packet including context from surrounding packets.
    """
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()

    # Drop identifiers or metadata before engineering as they are not features
    drop_cols = [col for col in ["ue_id", "traffic_type"] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 1. Handle Categorical Features (One-Hot Encoding)
    df = pd.get_dummies(
        df, columns=["type", "direction"], prefix=["type", "dir"], dummy_na=False
    )

    # 2. Calculate Inter-Packet Time
    df["inter_packet_time_ns"] = df["timestamp"].diff().dt.total_seconds() * 1e9
    df["time_to_next_packet_ns"] = df["timestamp"].diff(-1).dt.total_seconds() * -1e9

    # 3. Create Rolling Window Features for ALL Numeric Columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        if "time" in col:
            continue
        rolling_window = df[col].rolling(window=window_size, min_periods=1)
        df[f"{col}_rolling_mean"] = rolling_window.mean()
        df[f"{col}_rolling_std"] = rolling_window.std()

    # 4. Clean Up
    df = df.bfill().ffill()
    df = df.drop(columns=["timestamp"])

    return df
