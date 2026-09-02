"""
TON_IoT Data Preprocessing Pipeline
=====================================

Transforms raw TON_IoT network traffic CSV files into normalised numpy
arrays suitable for federated learning simulation.

Pipeline:
  1. Load and concatenate TON_IoT network traffic CSV(s)
  2. Binary label encoding: 0=Normal, 1=Attack
  3. Drop identifier/non-feature columns
  4. Remove duplicate flows
  5. Min-max normalise all features per column
  6. Train / val / test split (70% / 10% / 20%, stratified)
  7. Extract 500-sample server validation buffer from val split
  8. Save processed arrays to data/processed/

Dataset URL: https://research.unsw.edu.au/projects/toniot-datasets
Expected file: NF-TON-IoT.csv (Network Flow variant)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SERVER_DIR = Path("data/server")

# Expected TON_IoT columns to drop (identifiers, not features)
COLUMNS_TO_DROP = [
    "Label", "label", "Attack", "attack", "type", "Type",
    "src_ip", "dst_ip", "src_port", "dst_port",
    "Timestamp", "timestamp", "ts", "date", "time",
]

VAL_BUFFER_SIZE = 500
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_raw_data(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and concatenate all CSV or Parquet files from the raw data directory.

    Looks for NF-TON-IoT.csv first, then falls back to any *.parquet or *.csv in raw_dir.

    Returns
    -------
    pd.DataFrame  Raw concatenated dataset.
    """
    raw_dir = Path(raw_dir)
    candidate_csv = raw_dir / "NF-TON-IoT.csv"
    candidate_parquet = raw_dir / "NF-TON-IoT.parquet"

    if candidate_csv.exists():
        logger.info("Loading %s ...", candidate_csv)
        return pd.read_csv(candidate_csv, low_memory=False)

    if candidate_parquet.exists():
        logger.info("Loading %s ...", candidate_parquet)
        return pd.read_parquet(candidate_parquet)

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if parquet_files:
        logger.info("Loading %s ...", parquet_files[0])
        return pd.read_parquet(parquet_files[0])

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV or Parquet files found in {raw_dir}. "
            "Download the TON_IoT Network Traffic dataset from "
            "https://research.unsw.edu.au/projects/toniot-datasets "
            "and place it in data/raw/"
        )

    logger.info("Concatenating %d CSV files from %s ...", len(csv_files), raw_dir)
    frames = [pd.read_csv(f, low_memory=False) for f in csv_files]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def encode_labels(df: pd.DataFrame) -> pd.Series:
    """Return binary labels: 0=Normal, 1=Attack."""
    for col in ["label", "Label", "Attack", "attack", "type", "Type"]:
        if col in df.columns:
            if df[col].dtype == object:
                # String column: 'Normal'=0, everything else=1
                normal_variants = {"normal", "benign", "0", "none", "-"}
                labels = (~df[col].str.lower().isin(normal_variants)).astype(int)
            else:
                # Numeric: assume 0=normal
                labels = (df[col] != 0).astype(int)
            logger.info(
                "Label column '%s': %d normal (%.1f%%), %d attack (%.1f%%)",
                col, (labels == 0).sum(), 100 * (labels == 0).mean(),
                (labels == 1).sum(), 100 * (labels == 1).mean(),
            )
            return labels

    raise ValueError("No label column found. Expected one of: label, Label, Attack, type.")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Drop identifier/label columns and return numeric feature matrix."""
    drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    feature_df = df.drop(columns=drop_cols, errors="ignore")

    # Keep only numeric columns
    numeric_df = feature_df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric feature columns found after dropping labels.")

    logger.info("Feature matrix shape: %s  (columns: %d)", numeric_df.shape, numeric_df.shape[1])

    # Fill NaN with column mean
    X = numeric_df.fillna(numeric_df.mean()).values

    return X, list(numeric_df.columns)


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DIR,
    server_dir: Path = SERVER_DIR,
    test_size: float = 0.20,
    val_size: float = 0.10,
    val_buffer_size: int = VAL_BUFFER_SIZE,
    random_state: int = RANDOM_STATE,
    deduplicate: bool = True,
    save: bool = True,
) -> dict:
    """End-to-end preprocessing pipeline.

    Parameters
    ----------
    raw_dir         : Path  Directory containing raw CSV files.
    processed_dir   : Path  Output directory for processed arrays.
    server_dir      : Path  Output directory for server-side buffers.
    test_size       : float Fraction of data for held-out test set.
    val_size        : float Fraction of data for validation set (from remaining).
    val_buffer_size : int   Number of samples for server quality-eval buffer.
    random_state    : int   Random seed for reproducibility.
    deduplicate     : bool  Remove duplicate rows before splitting.
    save            : bool  Write processed files to disk.

    Returns
    -------
    dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test,
        X_val_buffer, y_val_buffer, feature_names, scaler
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(server_dir, exist_ok=True)

    # ── 1. Load ────────────────────────────────────────────────────────────
    df = load_raw_data(raw_dir)
    logger.info("Loaded %d rows × %d columns", *df.shape)

    # ── 2. Labels ─────────────────────────────────────────────────────────
    y = encode_labels(df).values

    # ── 3. Features ───────────────────────────────────────────────────────
    X, feature_names = extract_features(df)

    # ── 4. Deduplication ──────────────────────────────────────────────────
    if deduplicate:
        n_before = len(X)
        combined = np.column_stack([X, y.reshape(-1, 1)])
        _, unique_idx = np.unique(combined, axis=0, return_index=True)
        unique_idx.sort()
        X, y = X[unique_idx], y[unique_idx]
        logger.info(
            "Deduplication: %d → %d rows (removed %d duplicates)",
            n_before, len(X), n_before - len(X),
        )

    # ── 5. Min-Max Normalisation ───────────────────────────────────────────
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Features normalised to [0, 1].")

    # ── 6. Train / Val / Test Split ────────────────────────────────────────
    # val_fraction from total = val_size = 0.10
    # val_split_from_temp: val_size / (1 - test_size) = 0.10/0.80 = 0.125
    val_from_temp = val_size / (1.0 - test_size)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_from_temp, stratify=y_temp, random_state=random_state
    )

    logger.info(
        "Split — train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # ── 7. Server Validation Buffer ────────────────────────────────────────
    rng = np.random.RandomState(random_state)
    buf_idx = rng.choice(len(X_val), size=min(val_buffer_size, len(X_val)), replace=False)
    X_val_buffer, y_val_buffer = X_val[buf_idx], y_val[buf_idx]
    logger.info("Server validation buffer: %d samples", len(X_val_buffer))

    # ── 8. Save ────────────────────────────────────────────────────────────
    if save:
        np.savez_compressed(
            processed_dir / "train.npz", X=X_train, y=y_train
        )
        np.savez_compressed(
            processed_dir / "val.npz", X=X_val, y=y_val
        )
        np.savez_compressed(
            processed_dir / "test.npz", X=X_test, y=y_test
        )
        np.savez_compressed(
            server_dir / "val_buffer.npz", X=X_val_buffer, y=y_val_buffer
        )
        # Save scaler parameters for reproducibility
        np.savez_compressed(
            processed_dir / "scaler.npz",
            scale_=scaler.scale_,
            min_=scaler.min_,
            data_min_=scaler.data_min_,
            data_max_=scaler.data_max_,
        )
        # Save feature names
        with open(processed_dir / "feature_names.txt", "w") as f:
            f.write("\n".join(feature_names))

        logger.info("Processed data saved to %s", processed_dir)
        logger.info("Server buffer saved to %s", server_dir)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val,   "y_val": y_val,
        "X_test": X_test,  "y_test": y_test,
        "X_val_buffer": X_val_buffer, "y_val_buffer": y_val_buffer,
        "feature_names": feature_names,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Loader for already-processed data
# ---------------------------------------------------------------------------

def load_processed(processed_dir: Path = PROCESSED_DIR, server_dir: Path = SERVER_DIR) -> dict:
    """Load previously processed data arrays from disk."""
    processed_dir = Path(processed_dir)
    server_dir = Path(server_dir)

    def _load(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(
                f"Processed data not found at {path}. "
                "Run: python -m amfta.data.preprocessing"
            )
        return np.load(path)

    train  = _load(processed_dir / "train.npz")
    val    = _load(processed_dir / "val.npz")
    test   = _load(processed_dir / "test.npz")
    buffer = _load(server_dir / "val_buffer.npz")

    return {
        "X_train": train["X"], "y_train": train["y"],
        "X_val":   val["X"],   "y_val":   val["y"],
        "X_test":  test["X"],  "y_test":  test["y"],
        "X_val_buffer": buffer["X"], "y_val_buffer": buffer["y"],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(description="Preprocess TON_IoT dataset.")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--server_dir", default="data/server")
    parser.add_argument("--val_buffer_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preprocess(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        server_dir=Path(args.server_dir),
        val_buffer_size=args.val_buffer_size,
        random_state=args.seed,
    )
