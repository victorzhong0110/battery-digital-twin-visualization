"""Centralized data loader for the dashboard.

Loads all processed data once at startup and provides accessor functions.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.constants import PROCESSED_DIR


@lru_cache(maxsize=1)
def load_unified() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "unified.parquet")


@lru_cache(maxsize=1)
def load_features() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "features.parquet")


@lru_cache(maxsize=1)
def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "predictions.parquet")


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    with open(PROCESSED_DIR / "metrics.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_ecm_params() -> dict:
    path = PROCESSED_DIR / "ecm_params.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@lru_cache(maxsize=1)
def load_explanations() -> dict:
    path = PROCESSED_DIR / "explanations.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_curves(battery_id: str) -> dict[str, list[np.ndarray]]:
    """Load voltage/current/temp curve data for a battery."""
    npz_path = PROCESSED_DIR / f"curves_{battery_id}.npz"
    if not npz_path.exists():
        return {}
    data = np.load(npz_path, allow_pickle=True)
    result = {}
    for key in data.files:
        result[key] = list(data[key])
    return result


def load_ensemble_meta(battery_id: str) -> dict:
    path = PROCESSED_DIR / "models" / f"ensemble_meta_{battery_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_pinn_loss_history(battery_id: str) -> list[dict]:
    path = PROCESSED_DIR / "models" / f"pinn_loss_history_{battery_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def load_transformer_attention(battery_id: str) -> np.ndarray | None:
    path = PROCESSED_DIR / "models" / f"transformer_attn_{battery_id}.npy"
    if path.exists():
        return np.load(path)
    return None


def load_ensemble_weights(strategy: str, battery_id: str) -> np.ndarray | None:
    path = PROCESSED_DIR / "models" / f"ens_weights_{strategy}_{battery_id}.npy"
    if path.exists():
        return np.load(path)
    return None


def get_battery_ids() -> list[str]:
    """Get all battery IDs from predictions."""
    df = load_predictions()
    return sorted(df["battery_id"].unique().tolist())


def get_battery_summary() -> pd.DataFrame:
    """Build a summary table with one row per battery."""
    feat = load_features()
    rows = []
    for bid in feat["battery_id"].unique():
        bdf = feat[feat["battery_id"] == bid]
        source = bdf["source"].iloc[0]
        rows.append({
            "battery_id": bid,
            "source": source,
            "total_cycles": len(bdf),
            "initial_capacity": float(bdf["capacity_ah"].iloc[0]),
            "final_capacity": float(bdf["capacity_ah"].iloc[-1]),
            "final_soh": float(bdf["soh"].iloc[-1]),
            "rated_capacity": float(bdf["rated_capacity_ah"].iloc[0]),
            "capacity_fade_pct": float(
                (1 - bdf["capacity_ah"].iloc[-1] / bdf["capacity_ah"].iloc[0]) * 100
            ),
        })
    return pd.DataFrame(rows)
