"""Equivalent Circuit Model (1-RC Thevenin) for lithium-ion batteries.

Model equations:
    V_terminal = OCV(SOC) - I * R0 - V_rc
    dV_rc/dt = I/C1 - V_rc/(R1*C1)

where:
    R0: ohmic resistance (from EIS Re)
    R1: charge transfer resistance (from EIS Rct)
    C1: double-layer capacitance (estimated from time constant)
    OCV(SOC): open-circuit voltage as function of state of charge
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass
class ECMParams:
    """Parameters for the 1-RC Thevenin equivalent circuit model."""

    battery_id: str
    r0_ohm: float          # Ohmic resistance (Re from EIS)
    r1_ohm: float          # Charge transfer resistance (Rct from EIS)
    c1_farad: float        # Double-layer capacitance
    tau_s: float           # Time constant = R1 * C1
    rated_capacity_ah: float
    ocv_coeffs: list[float]  # Polynomial coefficients for OCV(SOC), highest degree first
    ocv_degree: int

    # Aging parameters (linear model: param = initial + slope * cycle)
    r0_initial: float
    r0_slope: float        # R0 increase per cycle
    r1_initial: float
    r1_slope: float        # R1 increase per cycle
    capacity_initial: float
    capacity_slope: float  # Capacity fade per cycle (negative)

    # Temperature coefficients (Arrhenius-like scaling)
    temp_ref_c: float = 24.0   # Reference temperature
    temp_coeff: float = 0.02   # Resistance increase per degree above ref

    def ocv(self, soc: float | np.ndarray) -> float | np.ndarray:
        """Compute open-circuit voltage from SOC using polynomial fit."""
        return np.polyval(self.ocv_coeffs, soc)

    def r0_at_cycle(self, cycle: int) -> float:
        """R0 value at a given cycle number."""
        return self.r0_initial + self.r0_slope * cycle

    def r1_at_cycle(self, cycle: int) -> float:
        """R1 value at a given cycle number."""
        return self.r1_initial + self.r1_slope * cycle

    def capacity_at_cycle(self, cycle: int) -> float:
        """Remaining capacity at a given cycle number."""
        return max(self.capacity_initial + self.capacity_slope * cycle, 0.1)

    def r_scaled_by_temp(self, r: float, temp_c: float) -> float:
        """Scale resistance by temperature (Arrhenius approximation)."""
        delta = temp_c - self.temp_ref_c
        # Higher temp -> lower resistance for Li-ion (but simplify for viz)
        # Actually for aging/visualization: higher temp -> slightly higher internal heat
        return r * (1.0 + self.temp_coeff * abs(delta))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ECMParams:
        return cls(**d)


def _fit_ocv_soc(voltage_curve: np.ndarray, capacity_ah: float, degree: int = 6) -> list[float]:
    """Fit OCV-SOC polynomial from a discharge voltage curve.

    Assumes the curve starts at SOC≈1 and ends at SOC≈0.
    SOC = 1 - (cumulative_discharge / total_capacity)
    """
    n = len(voltage_curve)
    # SOC decreases linearly from ~1 to ~0 during constant-current discharge
    soc = np.linspace(1.0, 0.0, n)
    # Fit polynomial
    coeffs = np.polyfit(soc, voltage_curve, degree)
    return coeffs.tolist()


def _estimate_time_constant(voltage_curve: np.ndarray, dt_s: float = 10.0) -> float:
    """Estimate RC time constant from voltage relaxation after load step.

    Heuristic: use the initial voltage drop rate at the start of discharge.
    The 1-RC response: V_rc(t) = I*R1*(1 - exp(-t/tau))
    We estimate tau from the curvature of the first few points.
    """
    if len(voltage_curve) < 20:
        return 30.0  # Default 30 seconds

    # Look at first 10% of discharge for transient behavior
    n_transient = max(int(len(voltage_curve) * 0.1), 10)
    v_start = voltage_curve[:n_transient]
    t = np.arange(n_transient) * dt_s

    # Voltage drop from initial
    dv = v_start[0] - v_start
    dv_max = dv[-1] if dv[-1] > 0 else 0.01

    # Find time to reach 63.2% of initial drop (1 time constant)
    target = 0.632 * dv_max
    idx = np.searchsorted(dv, target)
    tau = t[min(idx, len(t) - 1)]

    return max(tau, 5.0)  # At least 5 seconds


def calibrate_battery(
    battery_id: str,
    features_df,
    curves: dict[str, list[np.ndarray]],
) -> ECMParams:
    """Calibrate ECM parameters for a single battery from real data.

    Args:
        battery_id: e.g. "NASA_B0005"
        features_df: DataFrame filtered to this battery
        curves: dict from load_curves() with voltage/current arrays
    """
    df = features_df.sort_values("cycle_index").reset_index(drop=True)

    # --- OCV-SOC fit from first discharge (freshest battery) ---
    voltage_curves = curves.get("voltage_curve", [])
    if not voltage_curves:
        raise ValueError(f"No voltage curves available for {battery_id}")

    first_v = voltage_curves[0]
    first_cap = df["capacity_ah"].iloc[0]
    ocv_degree = 6
    ocv_coeffs = _fit_ocv_soc(first_v, first_cap, degree=ocv_degree)

    # --- R0 and R1 from impedance data ---
    re_vals = df["internal_resistance_ohm"].dropna()
    rct_vals = df["charge_transfer_resistance_ohm"].dropna()

    if len(re_vals) > 0:
        r0_initial = float(re_vals.iloc[0])
        # Linear fit for aging: R0 = initial + slope * cycle
        re_cycles = df.loc[re_vals.index, "cycle_index"].values
        if len(re_cycles) > 2:
            r0_fit = np.polyfit(re_cycles, re_vals.values, 1)
            r0_slope = float(r0_fit[0])
        else:
            r0_slope = 0.0
    else:
        # Fallback for CALCE: use internal_resistance_ohm
        ir_vals = df["internal_resistance_ohm"].dropna()
        r0_initial = float(ir_vals.iloc[0]) if len(ir_vals) > 0 else 0.05
        r0_slope = 0.0

    if len(rct_vals) > 0:
        r1_initial = float(rct_vals.iloc[0])
        rct_cycles = df.loc[rct_vals.index, "cycle_index"].values
        if len(rct_cycles) > 2:
            r1_fit = np.polyfit(rct_cycles, rct_vals.values, 1)
            r1_slope = float(r1_fit[0])
        else:
            r1_slope = 0.0
    else:
        # Estimate R1 as ~1.5x R0 when Rct not available
        r1_initial = r0_initial * 1.5
        r1_slope = r0_slope * 1.5

    # --- Capacity aging ---
    cap_vals = df["capacity_ah"].values
    cap_cycles = df["cycle_index"].values
    if len(cap_vals) > 2:
        cap_fit = np.polyfit(cap_cycles, cap_vals, 1)
        capacity_slope = float(cap_fit[0])
    else:
        capacity_slope = 0.0
    capacity_initial = float(cap_vals[0])

    # --- Time constant estimation ---
    # Average sampling interval from discharge duration and point count
    duration = df["discharge_duration_s"].iloc[0]
    n_points = len(first_v)
    dt_approx = duration / n_points if n_points > 0 else 10.0
    tau = _estimate_time_constant(first_v, dt_s=dt_approx)

    c1 = tau / r1_initial if r1_initial > 0 else 100.0

    rated_capacity = float(df["rated_capacity_ah"].iloc[0])

    params = ECMParams(
        battery_id=battery_id,
        r0_ohm=r0_initial,
        r1_ohm=r1_initial,
        c1_farad=c1,
        tau_s=tau,
        rated_capacity_ah=rated_capacity,
        ocv_coeffs=ocv_coeffs,
        ocv_degree=ocv_degree,
        r0_initial=r0_initial,
        r0_slope=r0_slope,
        r1_initial=r1_initial,
        r1_slope=r1_slope,
        capacity_initial=capacity_initial,
        capacity_slope=capacity_slope,
    )

    logger.info(
        "Calibrated %s: R0=%.4f(+%.6f/cyc), R1=%.4f(+%.6f/cyc), "
        "C1=%.1fF, tau=%.1fs, cap=%.3f(%.5f/cyc)",
        battery_id, r0_initial, r0_slope, r1_initial, r1_slope,
        c1, tau, capacity_initial, capacity_slope,
    )
    return params


def calibrate_all(processed_dir: str | Path | None = None) -> dict[str, ECMParams]:
    """Calibrate ECM for all batteries and save parameters."""
    from src.utils.constants import PROCESSED_DIR
    from src.data.unified_schema import load_unified, load_curves
    from src.data.feature_extract import load_features

    processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    features = load_features(processed_dir)
    all_params: dict[str, ECMParams] = {}

    for battery_id in features["battery_id"].unique():
        try:
            bat_features = features[features["battery_id"] == battery_id]
            curves = load_curves(battery_id, processed_dir)
            params = calibrate_battery(battery_id, bat_features, curves)
            all_params[battery_id] = params
        except Exception as e:
            logger.warning("Failed to calibrate %s: %s", battery_id, e)

    # Save to JSON
    params_path = processed_dir / "ecm_params.json"
    serializable = {k: v.to_dict() for k, v in all_params.items()}
    with open(params_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved ECM params for %d batteries to %s", len(all_params), params_path)

    return all_params


def load_ecm_params(processed_dir: str | Path | None = None) -> dict[str, ECMParams]:
    """Load pre-calibrated ECM parameters from JSON."""
    from src.utils.constants import PROCESSED_DIR

    processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    params_path = processed_dir / "ecm_params.json"
    with open(params_path) as f:
        raw = json.load(f)
    return {k: ECMParams.from_dict(v) for k, v in raw.items()}
