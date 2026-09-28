"""
Project 3: Flight Data Analyser
Aerospace-ML -- Foundations track

Purpose
-------
Read a rocket ascent telemetry CSV, clean it, and pull out the key
ascent events -- max dynamic pressure (max-Q), throttle-down window,
and MECO (Main Engine Cutoff) conditions -- using pandas instead of
hand-written loops.

Input file: Falcon9_Ascent_Telemetry.csv
    A simulated first-stage ascent profile (liftoff to MECO) built
    from a simplified vertical-ascent physics model using public,
    approximate Falcon 9 Block 5 parameters. It is NOT verified flight
    telemetry -- it's shaped like a real ascent (liftoff, throttle
    bucket near max-Q, MECO) with realistic sensor noise and a few
    dropped readings mixed in, specifically so there's something
    real to clean and analyze.

New Python concepts introduced in this build
----------------------------------------------
    - pandas DataFrames  : loading a CSV into a table you can query
    - Boolean masking    : df[df["col"] > x] to filter rows by condition
    - Aggregation        : .max(), .min(), .idxmax(), .describe()
    - Missing data        : .isna(), .interpolate(), .dropna()
    - Rolling windows     : .rolling().mean() to smooth noisy sensor data
    - .diff()             : the change between consecutive readings --
                             used to catch a sudden step, not just a level
    - CSV export          : .to_csv() -- the bridge to MATLAB (see below)

The MATLAB bridge
------------------
df.to_csv() writes a plain CSV. MATLAB reads that same file with
readtable() or readmatrix() -- no special library needed. Going the
other direction, MATLAB's writematrix() produces a CSV that pandas
reads right back with pd.read_csv(). Same file format both ways; the
CSV *is* the interface between the two environments.
"""

import pandas as pd

TELEMETRY_FILE = "Falcon9_Ascent_Telemetry.csv"
CLEANED_OUTPUT_FILE = "Falcon9_Ascent_Telemetry_Cleaned.csv"
SUMMARY_OUTPUT_FILE = "Falcon9_Ascent_Summary.csv"


def load_telemetry(filepath):
    """
    pd.read_csv() turns a CSV file straight into a DataFrame -- a table
    with labelled columns, similar in spirit to a spreadsheet.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Telemetry file not found: {filepath}. "
            f"Make sure it's in the same folder as this script."
        )
    return df


def clean_telemetry(df):
    """
    Real sensor data has gaps. Two common fixes, used in order:

      1. interpolate(): fill a gap by drawing a straight line between
         the readings just before and just after it. Good for smooth
         physical quantities (altitude, velocity, acceleration) that
         can't realistically jump between two adjacent readings.

      2. dropna(): remove any row still missing something afterward
         (e.g. a gap right at the start/end that has nothing to
         interpolate from).
    """
    missing_before = int(df.isna().sum().sum())
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.dropna()
    missing_after = int(df.isna().sum().sum())
    print(f"Cleaned telemetry: {missing_before} missing values -> {missing_after} remaining\n")
    return df


def smooth_acceleration(df, window=5):
    """
    A rolling mean averages each point with its neighbors within
    `window` readings. It smooths out one-off sensor jitter without
    hiding a real, sustained event like a throttle-down -- a genuine
    event shows up across several consecutive points, not just one.
    """
    df["acceleration_g_smoothed"] = (
        df["acceleration_g"].rolling(window=window, center=True, min_periods=1).mean()
    )
    return df


def analyze_ascent(df):
    """
    Boolean masking + aggregation: pull out the ascent events without
    writing a manual loop over rows.
    """
    max_q_row = df.loc[df["dynamic_pressure_kpa"].idxmax()]

    # Altitude and velocity climb the whole way to MECO in this dataset,
    # so their max is also the final telemetry point -- i.e. MECO itself.
    meco_row = df.loc[df["time_s"].idxmax()]

    # Why not just compare against a baseline level? Acceleration rises
    # for the whole burn as propellant mass burns off, so the very start
    # of the burn is naturally *low* too -- comparing against a fixed
    # baseline would flag liftoff itself as a false "throttle-down".
    #
    # What actually marks a throttle-down is a sudden STEP, not a low
    # level. .diff() gives the change between consecutive readings, so a
    # real throttle event shows up as one clear spike in .diff() -- a
    # sharp drop when thrust is cut, a sharp rise when it's restored --
    # while normal engine-only acceleration changes smoothly step to step.
    step = df["acceleration_g_smoothed"].diff()
    step_threshold = 0.05  # g change per second -- well above normal noise

    drops = df.loc[step < -step_threshold, "time_s"]
    throttle_start = drops.min() if not drops.empty else None

    if throttle_start is not None:
        # .min() here, not .max(): we want the FIRST rise after the drop
        # (thrust being restored), not the last one. Late in the burn,
        # mass has dropped so much that acceleration naturally ramps up
        # faster and can cross the same threshold again on its own --
        # that's real physics, not a second throttle event, so taking
        # the max would incorrectly grab that instead.
        rises = df.loc[(step > step_threshold) & (df["time_s"] > throttle_start), "time_s"]
        throttle_end = rises.min() if not rises.empty else None
    else:
        throttle_end = None

    summary = {
        "burn_duration_s": df["time_s"].max() - df["time_s"].min(),
        "max_q_time_s": max_q_row["time_s"],
        "max_q_kpa": round(max_q_row["dynamic_pressure_kpa"], 2),
        "meco_time_s": meco_row["time_s"],
        "meco_altitude_km": round(meco_row["altitude_km"], 2),
        "meco_velocity_ms": round(meco_row["velocity_ms"], 1),
        "throttle_down_start_s": throttle_start,
        "throttle_down_end_s": throttle_end,
    }
    return summary


def print_summary(summary):
    print("ASCENT SUMMARY")
    print("-" * 42)
    for key, value in summary.items():
        print(f"  {key:.<32} {value}")
    print()


def main():
    df = load_telemetry(TELEMETRY_FILE)
    print(f"Loaded {len(df)} telemetry rows from {TELEMETRY_FILE}\n")

    df = clean_telemetry(df)
    df = smooth_acceleration(df)

    summary = analyze_ascent(df)
    print_summary(summary)

    df.to_csv(CLEANED_OUTPUT_FILE, index=False)
    pd.DataFrame([summary]).to_csv(SUMMARY_OUTPUT_FILE, index=False)
    print(f"Cleaned telemetry written to: {CLEANED_OUTPUT_FILE}")
    print(f"Summary written to:           {SUMMARY_OUTPUT_FILE}")
    print("\nBoth are plain CSVs -- MATLAB can load either with readtable()")
    print("or readmatrix() with no conversion step.")


if __name__ == "__main__":
    main()
