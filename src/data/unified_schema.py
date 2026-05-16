"""Merge NASA and CALCE data into a unified schema and persist as Parquet.

The unified DataFrame drops raw curves (stored separately as numpy files)
and keeps scalar per-cycle metrics for fast dashboard loading.
Curves are saved as .npz for on-demand loading in detail views.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.nasa_parser import parse_all_nasa
from src.data.calce_parser import parse_all_calce
from src.utils.constants import PROCESSED_DIR, NASA_NOMINAL_CAPACITY_AH, CALCE_NOMINAL_CAPACITY_AH

logger = logging.getLogger(__name__)

# Scalar columns to keep in the main parquet
SCALAR_COLS = [
    "battery_id",
    "source",
    "cycle_index",
    "capacity_ah",
    "internal_resistance_ohm",
    "charge_transfer_resistance_ohm",
    "max_temp_c",
    "mean_temp_c",
    "ambient_temp_c",
    "discharge_duration_s",
]

CURVE_COLS = ["voltage_curve", "current_curve", "temp_curve"]


def _compute_rated_capacity(row: pd.Series) -> float:
    """Return the nominal rated capacity based on data source."""
    if row["source"] == "NASA":
        return NASA_NOMINAL_CAPACITY_AH
    return CALCE_NOMINAL_CAPACITY_AH


def build_unified(
    nasa_dir: str | Path | None = None,
    calce_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Parse all batteries, merge, compute SOH, and save.

    Returns the unified scalar DataFrame.
    """
    output_dir = Path(output_dir) if output_dir else PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse both datasets
    logger.info("Parsing NASA batteries...")
    nasa_df = parse_all_nasa(nasa_dir)
    logger.info("Parsing CALCE batteries...")
    calce_df = parse_all_calce(calce_dir)

    # Concatenate
    all_df = pd.concat([nasa_df, calce_df], ignore_index=True)
    logger.info("Total records: %d (%d NASA + %d CALCE)", len(all_df), len(nasa_df), len(calce_df))

    # Compute SOH per battery
    all_df["rated_capacity_ah"] = all_df.apply(_compute_rated_capacity, axis=1)
    all_df["soh"] = all_df["capacity_ah"] / all_df["rated_capacity_ah"]

    # Save curve data as compressed numpy archives (per battery)
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(exist_ok=True)

    for battery_id, group in all_df.groupby("battery_id"):
        curves = {}
        for col in CURVE_COLS:
            # Store as a list of arrays (ragged, so use object arrays)
            arrays = group[col].tolist()
            for i, arr in enumerate(arrays):
                if arr is not None:
                    curves[f"{col}_{i}"] = np.asarray(arr)
        if curves:
            np.savez_compressed(curves_dir / f"{battery_id}.npz", **curves)

    # Save scalar data
    scalar_df = all_df[SCALAR_COLS + ["rated_capacity_ah", "soh"]].copy()
    parquet_path = output_dir / "unified.parquet"
    scalar_df.to_parquet(parquet_path, index=False)
    logger.info("Saved unified data to %s (%d rows)", parquet_path, len(scalar_df))

    return scalar_df


def load_unified(processed_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the pre-built unified parquet."""
    processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    return pd.read_parquet(processed_dir / "unified.parquet")


def load_curves(battery_id: str, processed_dir: str | Path | None = None) -> dict[str, list[np.ndarray]]:
    """Load voltage/current/temp curves for a single battery.

    Returns dict like {"voltage_curve": [arr0, arr1, ...], ...}
    """
    processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    npz_path = processed_dir / "curves" / f"{battery_id}.npz"
    if not npz_path.exists():
        return {}

    data = np.load(npz_path, allow_pickle=True)
    result: dict[str, list[np.ndarray]] = {}
    for key in data.files:
        # Keys are like "voltage_curve_0", "voltage_curve_1", ...
        parts = key.rsplit("_", 1)
        col_name = parts[0]
        if col_name not in result:
            result[col_name] = []
        result[col_name].append(data[key])

    # Sort by index to maintain cycle order
    for col_name in result:
        # Re-sort based on original indices
        pass  # Already sorted by groupby order

    return result
