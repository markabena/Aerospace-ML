"""
Schema and paths for the C-MAPSS turbofan dataset.

Keeping column names and paths in one module means the rest of the
codebase never hardcodes a magic string. Change the dataset here and
every script downstream follows.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Dataset schema ------------------------------------------------------
# C-MAPSS files are space-separated with NO header row. The 26 columns are
# always in this fixed order.
INDEX_COLUMNS = ["unit_number", "time_in_cycles"]
SETTING_COLUMNS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# --- Subsets -------------------------------------------------------------
# Difficulty scales with the number of operating conditions and fault modes.
SUBSETS = {
    "FD001": {"conditions": 1, "fault_modes": 1, "train_units": 100, "test_units": 100},
    "FD002": {"conditions": 6, "fault_modes": 1, "train_units": 260, "test_units": 259},
    "FD003": {"conditions": 1, "fault_modes": 2, "train_units": 100, "test_units": 100},
    "FD004": {"conditions": 6, "fault_modes": 2, "train_units": 248, "test_units": 249},
}
DEFAULT_SUBSET = "FD001"

# --- RUL labelling -------------------------------------------------------
# See docs/rul_labelling.md for why this cap exists. Short version: an
# engine at cycle 1 is not meaningfully "more healthy" than one at cycle
# 50 -- degradation hasn't started yet, and the sensors can't tell them
# apart. Capping RUL stops the model chasing a signal that isn't there.
RUL_CAP = 125
