"""
Feature engineering for C-MAPSS RUL prediction.

The core problem this module solves: a single sensor reading says almost
nothing about remaining useful life. Engine #7 reading 47.2 on sensor_11
could be healthy or nearly dead -- it depends entirely on where 47.2 sits
relative to where that engine *started* and which direction it has been
moving. Degradation is a trajectory, not a value.

So the features here are all about capturing that trajectory: what the
recent average looks like, how much the signal is fluctuating, how far it
has drifted from the engine's own healthy baseline, and how steeply it is
currently trending.

Every feature is computed WITHIN an engine (groupby unit_number). Rolling
across the engine boundary would leak the end of engine 1 into the start
of engine 2 -- a silent, serious bug that still produces plausible-looking
numbers.
"""

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# SENSOR SELECTION
# ---------------------------------------------------------------------------
def select_sensors(df, min_distinct=10, min_abs_corr=0.10, label="rul", verbose=True):
    """
    Decide which sensors are worth keeping, using three filters in order.

    1. Zero variance -- the sensor never moves. Six of FD001's 21 sensors
       are like this because the subset has a single operating condition.

    2. Too few distinct values -- this catches what filter 1 misses.
       sensor_6 on FD001 has non-zero standard deviation but takes only
       TWO distinct values across all 20,631 rows (21.60 and 21.61). It
       is technically variable and practically useless. A pure std-based
       filter keeps it; a distinct-value filter drops it.

    3. Weak correlation with RUL -- the sensor moves, but not in any way
       that tracks degradation.

    Filters are applied empirically rather than hardcoding a sensor list,
    so the same function works unchanged on FD002/FD004 where different
    sensors carry the signal.

    Returns (kept_sensors, report_dataframe).
    """
    sensors = config.SENSOR_COLUMNS
    stats = pd.DataFrame(index=sensors)
    stats["std"] = df[sensors].std()
    stats["n_distinct"] = df[sensors].nunique()

    stats["drop_zero_variance"] = stats["std"] < 1e-9

    # Correlation is undefined for a constant series (zero denominator),
    # so compute it only on sensors that actually vary. Doing it blindly
    # produces a divide-by-zero warning and a NaN.
    varying = stats.index[~stats["drop_zero_variance"]]
    stats["corr_with_rul"] = df[varying].corrwith(df[label])

    stats["drop_few_distinct"] = (~stats["drop_zero_variance"]) & (
        stats["n_distinct"] < min_distinct
    )
    stats["drop_weak_corr"] = (
        (~stats["drop_zero_variance"])
        & (~stats["drop_few_distinct"])
        & (stats["corr_with_rul"].abs() < min_abs_corr)
    )
    stats["keep"] = ~(
        stats["drop_zero_variance"] | stats["drop_few_distinct"] | stats["drop_weak_corr"]
    )

    kept = stats.index[stats["keep"]].tolist()

    if verbose:
        dropped_zv = stats.index[stats["drop_zero_variance"]].tolist()
        dropped_fd = stats.index[stats["drop_few_distinct"]].tolist()
        dropped_wc = stats.index[stats["drop_weak_corr"]].tolist()
        print(f"Sensor selection: keeping {len(kept)} of {len(sensors)}")
        print(f"  dropped -- zero variance   ({len(dropped_zv)}): {dropped_zv}")
        print(f"  dropped -- few distinct    ({len(dropped_fd)}): {dropped_fd}")
        print(f"  dropped -- weak corr w/RUL ({len(dropped_wc)}): {dropped_wc}")

    return kept, stats


def select_operational_settings(df, min_distinct=10, verbose=True):
    """
    Same idea for the three operational settings. On FD001 (one operating
    condition) these barely move and op_setting_3 is perfectly constant.
    On FD002/FD004 (six conditions) they matter a great deal.
    """
    settings = config.SETTING_COLUMNS
    n_distinct = df[settings].nunique()
    kept = n_distinct[n_distinct >= min_distinct].index.tolist()
    if verbose:
        dropped = [s for s in settings if s not in kept]
        print(f"Operational settings: keeping {len(kept)} of {len(settings)} -- dropped {dropped}")
    return kept


# ---------------------------------------------------------------------------
# ROLLING FEATURES
# ---------------------------------------------------------------------------
def _rolling_slope(series, window):
    """
    Fit a straight line to the last `window` readings and return its
    gradient -- i.e. how fast this sensor is currently moving, and in
    which direction.

    This is the feature that most directly encodes "getting worse".
    A rolling mean tells you the current level; the slope tells you the
    rate of change, which is what separates an engine that has been
    sitting at a mildly elevated reading from one that is climbing fast.

    Implemented via np.polyfit over each window. min_periods=2 because a
    single point has no gradient.
    """
    x = np.arange(window)

    def slope(values):
        if len(values) < 2:
            return 0.0
        return np.polyfit(x[: len(values)], values, 1)[0]

    return series.rolling(window=window, min_periods=2).apply(slope, raw=True)


def add_rolling_features(df, sensors, windows=(5, 20), group_col="unit_number"):
    """
    Build rolling statistics per engine.

    For each sensor and each window size:
      _mean_w   : rolling average      -- smooths sensor noise
      _std_w    : rolling deviation    -- rising variability often precedes failure
      _slope_w  : rolling trend        -- rate and direction of change

    Two window sizes deliberately: a short one (5) reacts quickly to
    recent change, a long one (20) is stable but slow. Giving the model
    both lets it weigh responsiveness against reliability itself instead
    of forcing one choice up front.

    NOTE on causality: every rolling window here looks BACKWARD only.
    A centred window would average in future cycles -- information that
    would not exist at prediction time in a real deployment. That is
    data leakage, and it inflates offline scores while producing a model
    that cannot actually be run in service.
    """
    df = df.sort_values([group_col, "time_in_cycles"]).copy()
    grouped = df.groupby(group_col)
    new_columns = {}

    for sensor in sensors:
        series = grouped[sensor]
        for window in windows:
            new_columns[f"{sensor}_mean_{window}"] = series.transform(
                lambda s, w=window: s.rolling(w, min_periods=1).mean()
            )
            new_columns[f"{sensor}_std_{window}"] = series.transform(
                lambda s, w=window: s.rolling(w, min_periods=1).std()
            ).fillna(0.0)
            new_columns[f"{sensor}_slope_{window}"] = series.transform(
                lambda s, w=window: _rolling_slope(s, w)
            ).fillna(0.0)

    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)


def add_baseline_deviation(df, sensors, baseline_cycles=20, group_col="unit_number"):
    """
    How far has each sensor drifted from THIS engine's own healthy baseline?

    C-MAPSS engines start with different, unknown degrees of initial wear
    and manufacturing variation. So an absolute reading is not comparable
    across engines -- 47.2 might be normal for one engine and elevated for
    another.

    Taking each engine's mean over its first `baseline_cycles` cycles as
    its personal "healthy" reference, and expressing later readings as a
    deviation from that, makes the feature comparable fleet-wide. This is
    normalising per-engine rather than globally.
    """
    df = df.sort_values([group_col, "time_in_cycles"]).copy()
    new_columns = {}

    for sensor in sensors:
        baseline = df.groupby(group_col)[sensor].transform(
            lambda s, n=baseline_cycles: s.iloc[:n].mean()
        )
        new_columns[f"{sensor}_dev_from_baseline"] = df[sensor] - baseline

    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)


def add_cycle_features(df, group_col="unit_number"):
    """
    Age features. time_in_cycles alone is already informative -- an engine
    at cycle 300 is likelier to be near failure than one at cycle 20.

    Deliberately NOT included: anything derived from the engine's total
    lifetime (e.g. fraction of life elapsed). Total lifetime is only
    knowable after the engine has failed, so using it would leak the
    answer directly into the features. It produces near-perfect offline
    scores and a completely useless model.
    """
    df = df.copy()
    df["cycle_normalised"] = df["time_in_cycles"] / 362.0  # FD001 max observed
    return df


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------
def build_features(df, sensors, settings=None, windows=(5, 20), baseline_cycles=20):
    """Run the full feature-engineering pipeline and return (df, feature_names)."""
    df = add_rolling_features(df, sensors, windows=windows)
    df = add_baseline_deviation(df, sensors, baseline_cycles=baseline_cycles)
    df = add_cycle_features(df)

    engineered = [
        c for c in df.columns
        if any(tag in c for tag in ("_mean_", "_std_", "_slope_", "_dev_from_baseline"))
    ]
    feature_names = (
        sensors
        + list(settings or [])
        + engineered
        + ["time_in_cycles", "cycle_normalised"]
    )
    return df, feature_names
