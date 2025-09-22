import re, os
import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_phy_mac_log_line(line):
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


def parse_log_file(file_path):
    """
    Reads an entire log file, extracts UE ID, and returns a DataFrame of parsed data.
    """
    parsed_data = []
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
                parsed_data.append(data)

    if not parsed_data:
        return pd.DataFrame()

    df = pd.DataFrame(parsed_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S.%f")
    df.sort_values("timestamp", inplace=True)
    return df.reset_index(drop=True)


def engineer_contextual_packet_features(df, window_size=5):
    """
    Engineers features for each packet including context from surrounding packets.
    """
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()

    # Drop ue_id before engineering as it's an identifier, not a feature
    if "ue_id" in df.columns:
        df = df.drop(columns=["ue_id"])

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
