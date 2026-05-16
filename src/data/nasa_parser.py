"""Parse NASA PCoE lithium-ion battery .mat files into structured DataFrames.

Each .mat file contains a top-level struct with a 'cycle' array.
Each cycle has: type (charge/discharge/impedance), ambient_temperature, time, data.
Discharge data includes: Voltage_measured, Current_measured, Temperature_measured,
                         Current_load, Voltage_load, Time, Capacity.
Impedance data includes: Re, Rct, Battery_impedance, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio

logger = logging.getLogger(__name__)


def _safe_scalar(arr: Any) -> float | None:
    """Extract a scalar from a potentially nested MATLAB array."""
    if arr is None:
        return None
    try:
        val = float(np.squeeze(arr))
        return val if np.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def _safe_array(arr: Any) -> np.ndarray | None:
    """Extract a 1-D numpy array from a MATLAB matrix."""
    if arr is None:
        return None
    try:
        flat = np.squeeze(np.asarray(arr, dtype=np.float64))
        return flat if flat.size > 1 else None
    except (TypeError, ValueError):
        return None


def _parse_impedance_for_cycle(cycle_struct: Any) -> dict[str, float | None]:
    """Extract Re and Rct from an impedance cycle."""
    data = cycle_struct["data"][0, 0]
    return {
        "re": _safe_scalar(data.get("Re", None) if hasattr(data, "get") else data["Re"]),
        "rct": _safe_scalar(data.get("Rct", None) if hasattr(data, "get") else data["Rct"]),
    }


def parse_nasa_battery(mat_path: str | Path) -> pd.DataFrame:
    """Parse a single NASA battery .mat file.

    Returns a DataFrame with one row per discharge cycle, containing:
        - cycle_index: sequential discharge number
        - capacity_ah: discharge capacity
        - voltage_curve, current_curve, temp_curve: raw time-series as numpy arrays
        - max_temp_c, mean_temp_c, ambient_temp_c: temperature stats
        - internal_resistance_ohm (Re), charge_transfer_resistance_ohm (Rct)
        - discharge_duration_s: duration of discharge in seconds
    """
    mat_path = Path(mat_path)
    battery_id = mat_path.stem  # e.g. "B0005"

    logger.info("Parsing NASA battery: %s", battery_id)
    mat = sio.loadmat(str(mat_path))
    battery_struct = mat[battery_id]
    cycles = battery_struct["cycle"][0, 0]
    n_cycles = cycles.shape[1]

    # First pass: collect impedance measurements (Re, Rct) indexed by position
    # We'll map the most recent impedance values to each discharge cycle.
    impedance_log: list[dict[str, float | None]] = []
    for i in range(n_cycles):
        cycle = cycles[0, i]
        cycle_type = str(cycle["type"][0])
        if cycle_type == "impedance":
            try:
                data = cycle["data"][0, 0]
                imp = {
                    "idx": i,
                    "re": _safe_scalar(data["Re"]),
                    "rct": _safe_scalar(data["Rct"]),
                }
                impedance_log.append(imp)
            except (KeyError, IndexError):
                pass

    # Second pass: extract discharge cycles
    rows: list[dict] = []
    discharge_num = 0
    last_re: float | None = None
    last_rct: float | None = None
    imp_ptr = 0

    for i in range(n_cycles):
        cycle = cycles[0, i]
        cycle_type = str(cycle["type"][0])

        # Update impedance pointer
        while imp_ptr < len(impedance_log) and impedance_log[imp_ptr]["idx"] <= i:
            last_re = impedance_log[imp_ptr]["re"]
            last_rct = impedance_log[imp_ptr]["rct"]
            imp_ptr += 1

        if cycle_type != "discharge":
            continue

        data = cycle["data"][0, 0]
        voltage = _safe_array(data["Voltage_measured"])
        current = _safe_array(data["Current_measured"])
        temperature = _safe_array(data["Temperature_measured"])
        capacity = _safe_scalar(data["Capacity"])
        time_arr = _safe_array(data["Time"])
        ambient = _safe_scalar(cycle["ambient_temperature"])

        # Compute stats
        max_temp = float(np.nanmax(temperature)) if temperature is not None else None
        mean_temp = float(np.nanmean(temperature)) if temperature is not None else None
        duration = float(time_arr[-1] - time_arr[0]) if time_arr is not None and len(time_arr) > 1 else None

        rows.append({
            "battery_id": f"NASA_{battery_id}",
            "source": "NASA",
            "cycle_index": discharge_num,
            "raw_cycle_index": i,
            "capacity_ah": capacity,
            "internal_resistance_ohm": last_re,
            "charge_transfer_resistance_ohm": last_rct,
            "max_temp_c": max_temp,
            "mean_temp_c": mean_temp,
            "ambient_temp_c": ambient,
            "discharge_duration_s": duration,
            "voltage_curve": voltage,
            "current_curve": current,
            "temp_curve": temperature,
        })
        discharge_num += 1

    df = pd.DataFrame(rows)
    logger.info(
        "Parsed %s: %d discharge cycles, capacity %.3f -> %.3f Ah",
        battery_id,
        len(df),
        df["capacity_ah"].iloc[0] if len(df) > 0 else 0,
        df["capacity_ah"].iloc[-1] if len(df) > 0 else 0,
    )
    return df


def parse_all_nasa(nasa_dir: str | Path | None = None) -> pd.DataFrame:
    """Parse all NASA battery .mat files and concatenate."""
    from src.utils.constants import NASA_BATTERY_IDS, NASA_DIR

    nasa_dir = Path(nasa_dir) if nasa_dir else NASA_DIR
    frames = []
    for bid in NASA_BATTERY_IDS:
        mat_path = nasa_dir / f"{bid}.mat"
        if mat_path.exists():
            frames.append(parse_nasa_battery(mat_path))
        else:
            logger.warning("NASA file not found: %s", mat_path)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
