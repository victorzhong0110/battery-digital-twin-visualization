"""Real-time discharge and aging simulation using the 1-RC Thevenin ECM.

Single discharge simulation:
    Given ECM params + user conditions (C-rate, temperature, cutoff voltage),
    solve the ODE system to produce voltage, current, SOC, temperature curves.

Aging simulation:
    Simulate multiple charge-discharge cycles, updating R0/R1/capacity per cycle
    according to the linear aging model, producing a capacity trajectory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from src.models.ecm_model import ECMParams

logger = logging.getLogger(__name__)


@dataclass
class DischargeResult:
    """Result of a single discharge simulation."""

    time_s: np.ndarray        # Time array (seconds)
    voltage: np.ndarray       # Terminal voltage
    current_a: np.ndarray     # Discharge current (negative)
    soc: np.ndarray           # State of charge [0, 1]
    temperature_c: np.ndarray # Estimated surface temperature
    v_ocv: np.ndarray         # Open-circuit voltage
    v_rc: np.ndarray          # RC voltage drop
    capacity_ah: float        # Delivered capacity
    duration_s: float         # Discharge duration


@dataclass
class AgingResult:
    """Result of multi-cycle aging simulation."""

    cycles: np.ndarray          # Cycle numbers
    capacities_ah: np.ndarray   # Capacity per cycle
    soh: np.ndarray             # SOH per cycle
    r0_values: np.ndarray       # R0 evolution
    r1_values: np.ndarray       # R1 evolution


def simulate_discharge(
    params: ECMParams,
    c_rate: float = 1.0,
    temp_c: float = 24.0,
    cutoff_v: float = 2.7,
    cycle: int = 0,
    dt: float = 1.0,
    max_time_s: float = 15000.0,
) -> DischargeResult:
    """Simulate a single constant-current discharge.

    Args:
        params: Calibrated ECM parameters.
        c_rate: Discharge rate (e.g., 1.0 = 1C, 0.5 = 0.5C).
        temp_c: Ambient temperature in Celsius.
        cutoff_v: Discharge cutoff voltage.
        cycle: Current cycle number (affects aging parameters).
        dt: Time step for output (seconds).
        max_time_s: Maximum simulation time.

    Returns:
        DischargeResult with all time-series.
    """
    # Get parameters at current cycle
    capacity = params.capacity_at_cycle(cycle)
    r0 = params.r_scaled_by_temp(params.r0_at_cycle(cycle), temp_c)
    r1 = params.r_scaled_by_temp(params.r1_at_cycle(cycle), temp_c)
    c1 = params.c1_farad
    tau = r1 * c1

    # Discharge current (constant)
    i_discharge = capacity * c_rate  # Amps (positive for discharge convention)

    # Thermal model: simple lumped thermal with self-heating
    # dT/dt = (I^2 * R_total) / (m * Cp) - h * (T - T_ambient) / (m * Cp)
    # Simplified: use effective thermal parameters
    thermal_resistance = 3.0   # °C/W (typical for 18650)
    thermal_capacitance = 50.0  # J/°C
    thermal_tau = thermal_resistance * thermal_capacitance

    # ODE system: state = [SOC, V_rc, T_surface]
    def ode_rhs(t, state):
        soc, v_rc, t_surf = state

        # SOC dynamics: dSOC/dt = -I / (capacity * 3600)
        dsoc = -i_discharge / (capacity * 3600)

        # RC dynamics: dV_rc/dt = I/C1 - V_rc/(R1*C1)
        dv_rc = i_discharge / c1 - v_rc / tau

        # Thermal dynamics
        r_total = r0 + r1
        q_gen = i_discharge ** 2 * r_total  # Heat generation (W)
        q_loss = (t_surf - temp_c) / thermal_resistance  # Heat dissipation (W)
        dt_surf = (q_gen - q_loss) / thermal_capacitance

        return [dsoc, dv_rc, dt_surf]

    # Event: stop when voltage reaches cutoff
    def voltage_cutoff(t, state):
        soc, v_rc, _ = state
        v_ocv = params.ocv(max(soc, 0.0))
        v_terminal = v_ocv - i_discharge * r0 - v_rc
        return v_terminal - cutoff_v

    voltage_cutoff.terminal = True
    voltage_cutoff.direction = -1

    # Event: stop when SOC reaches 0
    def soc_floor(t, state):
        return state[0]  # SOC = 0

    soc_floor.terminal = True
    soc_floor.direction = -1

    # Initial conditions
    y0 = [1.0, 0.0, temp_c]  # SOC=1, V_rc=0, T=ambient

    # Solve
    t_span = (0.0, max_time_s)
    t_eval = np.arange(0, max_time_s, dt)

    sol = solve_ivp(
        ode_rhs,
        t_span,
        y0,
        method="RK45",
        t_eval=t_eval,
        events=[voltage_cutoff, soc_floor],
        max_step=dt,
        rtol=1e-6,
        atol=1e-8,
    )

    time_s = sol.t
    soc_arr = np.clip(sol.y[0], 0.0, 1.0)
    v_rc_arr = sol.y[1]
    temp_arr = sol.y[2]

    # Compute voltages
    v_ocv_arr = np.array([params.ocv(s) for s in soc_arr])
    v_terminal = v_ocv_arr - i_discharge * r0 - v_rc_arr

    # Current array (constant)
    current_arr = np.full_like(time_s, -i_discharge)

    # Delivered capacity
    delivered_ah = i_discharge * time_s[-1] / 3600 if len(time_s) > 0 else 0.0

    return DischargeResult(
        time_s=time_s,
        voltage=v_terminal,
        current_a=current_arr,
        soc=soc_arr,
        temperature_c=temp_arr,
        v_ocv=v_ocv_arr,
        v_rc=v_rc_arr,
        capacity_ah=delivered_ah,
        duration_s=float(time_s[-1]) if len(time_s) > 0 else 0.0,
    )


def simulate_aging(
    params: ECMParams,
    n_cycles: int = 200,
    c_rate: float = 1.0,
    temp_c: float = 24.0,
    cutoff_v: float = 2.7,
) -> AgingResult:
    """Simulate battery aging over multiple discharge cycles.

    Uses the linear aging model in ECMParams to update R0, R1, and capacity
    per cycle, then simulates each discharge to get actual delivered capacity.

    Args:
        params: Calibrated ECM parameters.
        n_cycles: Number of cycles to simulate.
        c_rate: Discharge rate for all cycles.
        temp_c: Ambient temperature for all cycles.
        cutoff_v: Cutoff voltage for all cycles.

    Returns:
        AgingResult with per-cycle metrics.
    """
    cycles = np.arange(n_cycles)
    capacities = np.zeros(n_cycles)
    r0_vals = np.zeros(n_cycles)
    r1_vals = np.zeros(n_cycles)

    for i in range(n_cycles):
        # Get aged parameters
        r0_vals[i] = params.r0_at_cycle(i)
        r1_vals[i] = params.r1_at_cycle(i)

        # Simulate discharge at this cycle
        result = simulate_discharge(
            params,
            c_rate=c_rate,
            temp_c=temp_c,
            cutoff_v=cutoff_v,
            cycle=i,
            dt=5.0,  # Coarser time step for speed
        )
        capacities[i] = result.capacity_ah

    soh = capacities / params.rated_capacity_ah

    return AgingResult(
        cycles=cycles,
        capacities_ah=capacities,
        soh=soh,
        r0_values=r0_vals,
        r1_values=r1_vals,
    )
