"""Extract health indicator features from unified battery data.

Features per cycle:
  - soh: State of Health (capacity / rated_capacity)
  - capacity_fade_rate: capacity change from previous cycle
  - resistance_increase_rate: resistance change from previous cycle
  - voltage_slope: slope of voltage during constant-current discharge
  - temp_rise: max_temp - ambient_temp
  - discharge_duration_s: directly from unified data
  - cycle_index: aging proxy
  - internal_resistance_ohm: from unified data
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.constants import PROCESSED_DIR

logger = logging.getLogger(__name__)


def _compute_voltage_slope(voltage_curve: np.ndarray | None) -> float | None:
    """Compute the linear regression slope of voltage during discharge.

    A steeper (more negative) slope indicates faster voltage drop.
    """
    if voltage_curve is None or not isinstance(voltage_curve, np.ndarray):
        return None
    if len(voltage_curve) < 10:
        return None

    # Use middle 60% of the curve to avoid edge effects
    n = len(voltage_curve)
    start = int(n * 0.2)
    end = int(n * 0.8)
    segment = voltage_curve[start:end]

    if len(segment) < 5:
        return None

    x = np.arange(len(segment), dtype=np.float64)
    try:
        slope = np.polyfit(x, segment, deg=1)[0]
        return float(slope)
    except (np.linalg.LinAlgError, ValueError):
        return None


def extract_features(
    unified_df: pd.DataFrame,
    curves_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Compute per-cycle features from the unified scalar DataFrame.

    Args:
        unified_df: DataFrame from unified_schema.build_unified or load_unified.
        curves_dir: Path to curves/ directory with .npz files for voltage slopes.

    Returns:
        DataFrame with one row per (battery_id, cycle_index) and feature columns.
    """
    curves_dir = Path(curves_dir) if curves_dir else PROCESSED_DIR / "curves"

    features_list: list[pd.DataFrame] = []

    for battery_id, group in unified_df.groupby("battery_id"):
        df = group.sort_values("cycle_index").reset_index(drop=True).copy()

        # --- Scalar features from unified data ---
        df["capacity_fade_rate"] = df["capacity_ah"].diff()
        df["resistance_increase_rate"] = df["internal_resistance_ohm"].diff()
        df["temp_rise"] = df["max_temp_c"] - df["ambient_temp_c"]

        # --- Voltage slope from curves ---
        voltage_slopes: list[float | None] = []
        npz_path = curves_dir / f"{battery_id}.npz"
        if npz_path.exists():
            npz_data = np.load(npz_path, allow_pickle=True)
            for i in range(len(df)):
                key = f"voltage_curve_{i}"
                if key in npz_data:
                    voltage_slopes.append(_compute_voltage_slope(npz_data[key]))
                else:
                    voltage_slopes.append(None)
        else:
            voltage_slopes = [None] * len(df)

        df["voltage_slope"] = voltage_slopes

        # --- Rolling statistics (window=5) ---
        df["capacity_rolling_std"] = df["capacity_ah"].rolling(window=5, min_periods=2).std()
        df["resistance_rolling_mean"] = df["internal_resistance_ohm"].rolling(window=5, min_periods=1).mean()

        # --- Normalized features ---
        rated = df["rated_capacity_ah"].iloc[0] if "rated_capacity_ah" in df.columns else 1.0
        df["capacity_normalized"] = df["capacity_ah"] / rated
        initial_resistance = df["internal_resistance_ohm"].dropna().iloc[0] if df["internal_resistance_ohm"].dropna().any() else 1.0
        df["resistance_normalized"] = df["internal_resistance_ohm"] / initial_resistance

        features_list.append(df)

    result = pd.concat(features_list, ignore_index=True)

    # Select final feature columns
    feature_cols = [
        "battery_id", "source", "cycle_index",
        "capacity_ah", "soh", "rated_capacity_ah",
        "internal_resistance_ohm", "charge_transfer_resistance_ohm",
        "max_temp_c", "mean_temp_c", "ambient_temp_c",
        "discharge_duration_s", "temp_rise",
        "capacity_fade_rate", "resistance_increase_rate",
        "voltage_slope",
        "capacity_rolling_std", "resistance_rolling_mean",
        "capacity_normalized", "resistance_normalized",
    ]
    # Only keep columns that exist
    existing_cols = [c for c in feature_cols if c in result.columns]
    result = result[existing_cols]

    logger.info("Extracted features: %d rows, %d columns", len(result), len(result.columns))
    return result


def build_features(processed_dir: str | Path | None = None) -> pd.DataFrame:
    """Load unified data and extract features, save to parquet."""
    from src.data.unified_schema import load_unified

    processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    unified = load_unified(processed_dir)
    features = extract_features(unified, curves_dir=processed_dir / "curves")

    output_path = processed_dir / "features.parquet"
    features.to_parquet(output_path, index=False)
    logger.info("Saved features to %s", output_path)
    return features


def load_features(processed_dir: str | Path | None = None) -> pd.DataFrame:
    """Load pre-built features from parquet."""
    processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    return pd.read_parquet(processed_dir / "features.parquet")
