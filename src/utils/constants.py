"""Project-wide constants."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NASA_DIR = DATA_DIR / "NASA"
CALCE_DIR = DATA_DIR / "CALCE"
PROCESSED_DIR = DATA_DIR / "processed"

# NASA battery IDs
NASA_BATTERY_IDS = ("B0005", "B0006", "B0007", "B0018")

# CALCE battery IDs
CALCE_BATTERY_IDS = ("CS2_35", "CS2_36", "CS2_37", "CS2_38")

# Nominal capacity (Ah) for SOH calculation
NASA_NOMINAL_CAPACITY_AH = 2.0
CALCE_NOMINAL_CAPACITY_AH = 1.1  # CS2 series rated ~1.1 Ah

# Thresholds
SOH_WARNING_THRESHOLD = 0.80
SOH_CRITICAL_THRESHOLD = 0.70
TEMP_WARNING_C = 45.0

# Unified column names
COLS = {
    "battery_id": "battery_id",
    "source": "source",
    "cycle": "cycle_index",
    "capacity": "capacity_ah",
    "resistance": "internal_resistance_ohm",
    "rct": "charge_transfer_resistance_ohm",
    "max_temp": "max_temp_c",
    "mean_temp": "mean_temp_c",
    "ambient_temp": "ambient_temp_c",
    "duration": "discharge_duration_s",
    "voltage_curve": "voltage_curve",
    "current_curve": "current_curve",
    "temp_curve": "temp_curve",
}
