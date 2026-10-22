"""Load and preprocess the shared survival datasets (Fair Rules 1-2).

All three datasets (METABRIC, SUPPORT, GBSG) are loaded once through pycox,
exactly as specified in the project plan, so both the classic and deep side
use the identical copy of the data.

Note: pycox's versions of these datasets are already fully numeric (no raw
categorical/string columns), so there is nothing left to one-hot encode.
Scaling is still applied, fit on the training rows only, to avoid leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pycox.datasets import gbsg, metabric, support
from sklearn.preprocessing import StandardScaler

DATASETS = {
    "metabric": metabric,
    "support": support,
    "gbsg": gbsg,
}


def load_dataset(name: str) -> pd.DataFrame:
    """Load one dataset as a DataFrame with feature columns, 'duration', 'event'."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASETS)}.")
    df = DATASETS[name].read_df()
    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except the survival target columns."""
    return [c for c in df.columns if c not in ("duration", "event")]


def _targets(df: pd.DataFrame, idx) -> tuple[np.ndarray, np.ndarray]:
    duration = df.loc[idx, "duration"].to_numpy(dtype="float64")
    event = df.loc[idx, "event"].to_numpy(dtype="float64")
    return duration, event


def preprocess_split(df: pd.DataFrame, train_idx, val_idx, test_idx):
    """Scale features, fitting the scaler on the training rows only (Fair Rule 2).

    Returns (X_train, X_val, X_test, y_train, y_val, y_test). Each X is a
    numpy array of scaled features. Each y is a (duration, event) pair of
    numpy arrays.
    """
    feats = feature_columns(df)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df.loc[train_idx, feats].to_numpy(dtype="float64"))
    X_val = scaler.transform(df.loc[val_idx, feats].to_numpy(dtype="float64"))
    X_test = scaler.transform(df.loc[test_idx, feats].to_numpy(dtype="float64"))

    y_train = _targets(df, train_idx)
    y_val = _targets(df, val_idx)
    y_test = _targets(df, test_idx)
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    for name in DATASETS:
        d = load_dataset(name)
        print(f"{name}: {d.shape[0]} rows, {len(feature_columns(d))} features, "
              f"{d['event'].mean():.1%} events")
