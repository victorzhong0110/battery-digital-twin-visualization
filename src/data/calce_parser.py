"""Parse CALCE CS2 series battery Excel files into structured DataFrames.

Each battery has a directory of date-stamped .xlsx files from Arbin tester.
Columns: Data_Point, Test_Time(s), Date_Time, Step_Time(s), Step_Index,
         Cycle_Index, Current(A), Voltage(V), Charge_Capacity(Ah),
         Discharge_Capacity(Ah), Charge_Energy(Wh), Discharge_Energy(Wh),
         dV/dt(V/s), Internal_Resistance(Ohm), Is_FC_Data,
         AC_Impedance(Ohm), ACI_Phase_Angle(Deg)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical column mapping (handle potential naming variations)
_COL_MAP = {
    "Data_Point": "data_point",
    "Test_Time(s)": "test_time_s",
    "Date_Time": "date_time",
    "Step_Time(s)": "step_time_s",
    "Step_Index": "step_index",
    "Cycle_Index": "cycle_index",
    "Current(A)": "current_a",
    "Voltage(V)": "voltage_v",
    "Charge_Capacity(Ah)": "charge_capacity_ah",
    "Discharge_Capacity(Ah)": "discharge_capacity_ah",
    "Charge_Energy(Wh)": "charge_energy_wh",
    "Discharge_Energy(Wh)": "discharge_energy_wh",
    "dV/dt(V/s)": "dv_dt",
    "Internal_Resistance(Ohm)": "internal_resistance_ohm",
    "Is_FC_Data": "is_fc_data",
    "AC_Impedance(Ohm)": "ac_impedance_ohm",
    "ACI_Phase_Angle(Deg)": "aci_phase_angle_deg",
}


def _find_channel_sheet(xlsx_path: Path) -> str | None:
    """Find the data sheet name (Channel_*) in an Arbin xlsx file."""
    import openpyxl

    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)
    sheets = wb.sheetnames
    wb.close()
    for s in sheets:
        if s.lower().startswith("channel"):
            return s
    # Fallback: if only one sheet or first sheet has Data_Point
    return sheets[0] if sheets else None


def _load_and_concat_xlsx(battery_dir: Path) -> pd.DataFrame:
    """Load all xlsx files for one battery, concatenate, clean, and sort."""
    xlsx_files = sorted(battery_dir.glob("*.xlsx"))
    if not xlsx_files:
        logger.warning("No xlsx files in %s", battery_dir)
        return pd.DataFrame()

    frames = []
    for f in xlsx_files:
        try:
            # Arbin xlsx files have data in a "Channel_*" sheet, not the first sheet
            sheet_name = _find_channel_sheet(f)
            df = pd.read_excel(f, engine="openpyxl", sheet_name=sheet_name)

            # Normalize column names
            rename = {}
            for orig, canon in _COL_MAP.items():
                if orig in df.columns:
                    rename[orig] = canon
            df = df.rename(columns=rename)
            frames.append(df)
        except Exception as e:
            logger.warning("Failed to read %s: %s", f.name, e)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # De-duplicate by data_point (global counter) and sort by test time
    if "data_point" in combined.columns:
        combined = combined.drop_duplicates(subset=["data_point"], keep="first")
    if "test_time_s" in combined.columns:
        combined = combined.sort_values("test_time_s").reset_index(drop=True)

    return combined


def _extract_discharge_cycles(raw_df: pd.DataFrame, battery_id: str) -> pd.DataFrame:
    """Extract per-cycle summary from raw time-series data.

    CALCE Arbin data uses Cycle_Index that resets per xlsx file. After
    concatenation, multiple files may share the same Cycle_Index.  We detect
    true cycle boundaries via large time-gaps in the discharge current signal
    and assign a global sequential index.

    A discharge segment is identified by negative current (current_a < -0.01).
    """
    if raw_df.empty:
        return pd.DataFrame()

    # Identify discharge rows (current is negative during discharge)
    discharge_mask = raw_df["current_a"] < -0.01
    discharge_df = raw_df.loc[discharge_mask].copy()

    if discharge_df.empty:
        logger.warning("No discharge data found for %s", battery_id)
        return pd.DataFrame()

    # Detect true cycle boundaries: a gap > 300s in test_time between
    # consecutive discharge rows indicates a new cycle.
    discharge_df = discharge_df.sort_values("test_time_s").reset_index(drop=True)
    time_diff = discharge_df["test_time_s"].diff()
    cycle_boundary = (time_diff > 300) | (time_diff.isna())
    discharge_df["global_cycle"] = cycle_boundary.cumsum()

    rows: list[dict] = []
    for cycle_idx, group in discharge_df.groupby("global_cycle"):
        if len(group) < 5:  # Skip very short segments
            continue

        voltage = group["voltage_v"].values
        current = group["current_a"].values
        test_times = group["test_time_s"].values

        # Per-cycle discharge capacity via Coulomb counting: integrate |current| * dt
        dt = np.diff(test_times)
        avg_current = np.abs(current[:-1] + current[1:]) / 2
        capacity = float(np.sum(avg_current * dt) / 3600)  # Ah

        # Sanity filter: single-cycle capacity should be 0.3~2.0 Ah for CS2 cells
        if capacity < 0.3 or capacity > 2.0:
            continue

        # Internal resistance: use non-zero median
        ir_col = "internal_resistance_ohm"
        ir_values = group[ir_col] if ir_col in group.columns else pd.Series(dtype=float)
        ir_nonzero = ir_values[ir_values > 0]
        resistance = float(ir_nonzero.median()) if len(ir_nonzero) > 0 else None

        duration = float(test_times[-1] - test_times[0])

        rows.append({
            "battery_id": f"CALCE_{battery_id}",
            "source": "CALCE",
            "cycle_index": int(cycle_idx),
            "capacity_ah": capacity,
            "internal_resistance_ohm": resistance,
            "charge_transfer_resistance_ohm": None,
            "max_temp_c": None,  # Not in CALCE CS2 Arbin data
            "mean_temp_c": None,
            "ambient_temp_c": 25.0,  # Room temperature assumption
            "discharge_duration_s": duration,
            "voltage_curve": voltage,
            "current_curve": current,
            "temp_curve": None,
        })

    df = pd.DataFrame(rows)

    # Re-index cycles sequentially from 0
    df = df.sort_values("cycle_index").reset_index(drop=True)
    df["cycle_index"] = range(len(df))

    if len(df) > 0:
        logger.info(
            "Parsed %s: %d discharge cycles, capacity %.3f -> %.3f Ah",
            battery_id,
            len(df),
            df["capacity_ah"].iloc[0],
            df["capacity_ah"].iloc[-1],
        )
    return df


def parse_calce_battery(battery_dir: str | Path) -> pd.DataFrame:
    """Parse a single CALCE battery directory.

    Expects structure: battery_dir/CS2_XX/*.xlsx
    """
    battery_dir = Path(battery_dir)
    battery_id = battery_dir.name  # e.g. "CS2_35"

    # Handle double-nested directory: CS2_35/CS2_35/*.xlsx
    inner_dir = battery_dir / battery_id
    if inner_dir.exists():
        battery_dir = inner_dir

    logger.info("Parsing CALCE battery: %s from %s", battery_id, battery_dir)
    raw_df = _load_and_concat_xlsx(battery_dir)
    if raw_df.empty:
        return pd.DataFrame()

    return _extract_discharge_cycles(raw_df, battery_id)


def parse_all_calce(calce_dir: str | Path | None = None) -> pd.DataFrame:
    """Parse all CALCE batteries and concatenate."""
    from src.utils.constants import CALCE_BATTERY_IDS, CALCE_DIR

    calce_dir = Path(calce_dir) if calce_dir else CALCE_DIR
    frames = []
    for bid in CALCE_BATTERY_IDS:
        bat_dir = calce_dir / bid
        if bat_dir.exists():
            frames.append(parse_calce_battery(bat_dir))
        else:
            logger.warning("CALCE dir not found: %s", bat_dir)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
