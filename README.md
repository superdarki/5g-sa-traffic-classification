# 5G SA Traffic Classification

Classify enhanced Mobile Broadband (eMBB) and Ultra-Reliable Low-Latency Communication (URLLC) traffic from standalone (SA) 5G traces using LightGBM and contextual PHY/MAC features.

## Highlights
- `transform_log.py` - normalizes raw logs into a single CSV schema with `traffic_type` labels.
- `traffic_utils.py` - normalization helpers and contextual feature engineering.
- `train_model.py` - trains or warm starts a LightGBM classifier from normalized CSV logs.
- `evaluate_model.py` - scores a saved model on normalized datasets and writes evaluation plots.
- `classify_log.py` - performs per-packet predictions on a normalized capture.
- `list_feature_importances.py` - optional helper to inspect the most important features.
- `data/` - sample logs you can use for experiments.

## Requirements
- Python 3.11 or newer.
- A virtual environment is recommended so project dependencies do not leak into the system Python.
- Raw logs from Amarisoft gNB or RS ROME exports (normalized before training/evaluation).

## Installation (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks script execution, run `Set-ExecutionPolicy -Scope Process RemoteSigned` before activating the environment.

## Training a Model
Train from normalized logs (CSV) that already include `traffic_type` labels:

```powershell
python train_model.py data/normalized/embb_and_urllc.normalized.csv --output traffic_classifier.joblib
```

The script:
- Loads and merges normalized logs into a single timeline (system-information packets with `harq == -1` are already filtered during normalization).
- Engineers rolling-window statistics and categorical encodings via `traffic_utils.engineer_contextual_packet_features`.
- Trains a LightGBM classifier (or continues training if `--retrain-from` is supplied).
- Displays a confusion matrix and top feature importances.
- Persists a bundle (`model`, `LabelEncoder`, feature columns, and window size) to disk.

Key optional arguments:
- `--window_size`: window length for contextual rolling statistics (default `5`).
- `--retrain-from`: path to an existing `.joblib` bundle to warm-start training.

## Evaluating a Saved Model
```powershell
python evaluate_model.py traffic_classifier.joblib data/normalized/mixed_1.normalized.csv data/normalized/mixed_2.normalized.csv
```

Outputs include overall accuracy, a full `classification_report`, misclassification counts, and `confusion_matrix_evaluation.png`. Use `--verbose` to print packet-level predictions.

## Classifying a New Capture
```powershell
python classify_log.py data/normalized/new_capture.normalized.csv --model traffic_classifier.joblib
```

The script engineers the same contextual features, aligns them to the training schema, and prints each packet's predicted slice with confidence.

## Normalizing Log Formats
If you have logs from multiple sources, normalize them once and use the normalized CSVs everywhere:

```powershell
python transform_log.py data/amarisoft_uesim/20250829-133752_urllc_ping_long.log --format amarisoft --embb-ue 9 --urllc-ue 1
python transform_log.py data/rs_romes/40MHz.csv --format rome --urllc-time 11:52:59.000-11:53:10.000 --embb-time 11:53:20.000-11:54:00.000 --output-dir data/normalized
```

Notes:
- `--format` is required. Aliases: `rome` → `rs_rome`, `norm` → `normalized`.
- RS ROME logs are labeled using inclusive time ranges on the `Time` column.
- Amarisoft logs are labeled using UE IDs.
- Normalized inputs should already contain `traffic_type` (no labeling arguments allowed).

The normalized format is a CSV with the columns in `traffic_utils.STANDARD_COLUMNS`. It includes `traffic_type` and does not include UE IDs. System-information packets (`harq == -1`) are removed during normalization.

## Understanding the Features
`traffic_utils.engineer_contextual_packet_features` builds:
- One-hot encoded indicators for packet `type` (PUSCH, PDSCH, PDCCH, PUCCH, MAC) and `direction`.
- Inter-packet timing features (`inter_packet_time_ns`, `time_to_next_packet_ns`).
- Rolling mean and standard deviation metrics for numeric columns such as transport block size, PRB counts, modulation order, code rate, retransmissions, SNR, and MAC payload sizes.
- Forward/backward filling to remove gaps before dropping the timestamp column.

Run the helper to inspect the most informative inputs:
```powershell
python list_feature_importances.py traffic_classifier.joblib --top-k 25
```

## Sample Data
The `data/` directory ships with assorted captures that mix ping, throughput, and UDP traffic. Use them to bootstrap training or to validate the pipeline.

## Troubleshooting
- `transform_log.py` will reject mismatched labeling args (UE IDs for RS ROME, or time ranges for Amarisoft).
- If a capture lacks certain packet types, missing features are zero-filled when aligning with the model columns.
- Retraining reuses the original `LabelEncoder` so class indices remain stable across sessions.
- In headless setups, comment out or remove the plotting calls in `train_model.py` and `evaluate_model.py`.

## License
This project is distributed under the terms described in the `LICENSE` file.
