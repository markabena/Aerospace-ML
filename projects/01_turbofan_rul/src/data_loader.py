"""
Loading and RUL-labelling for the C-MAPSS turbofan dataset.

The raw files carry no header and no RUL column -- RUL has to be derived
from the run-to-failure structure of the training data. That derivation
is the single most important modelling decision in this project, so it
lives here rather than being buried in a training script.
"""

import pandas as pd

from . import config


def _read_cmapss_file(filepath):
    """
    Read one raw C-MAPSS .txt file into a DataFrame.

    Two format quirks worth knowing, both of which bite people:

    1. The files are space-separated with no header, so column names
       must be supplied explicitly.
    2. Every line ends with trailing whitespace, which pandas reads as
       two extra all-NaN columns. `sep=r"\\s+"` collapses runs of
       whitespace and sidesteps the problem entirely.
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=config.ALL_COLUMNS,
        engine="python",
    )
    # Defensive: if a file ever does produce phantom columns, drop them
    # rather than letting all-NaN features reach a model.
    df = df.dropna(axis=1, how="all")
    return df


def load_train(subset=config.DEFAULT_SUBSET, data_dir=None):
    """Load a training subset, e.g. train_FD001.txt."""
    data_dir = data_dir or config.RAW_DATA_DIR
    path = data_dir / f"train_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for how to obtain the "
            f"C-MAPSS dataset -- the raw files are not committed to this repo."
        )
    return _read_cmapss_file(path)


def load_test(subset=config.DEFAULT_SUBSET, data_dir=None):
    """
    Load a test subset plus its ground-truth RUL file.

    Returns (test_df, true_rul_series).

    The test files are TRUNCATED: each engine's history stops at some
    point before failure. RUL_FD00X.txt gives the true remaining cycles
    for each engine at its final recorded row -- one value per engine,
    in unit_number order.
    """
    data_dir = data_dir or config.RAW_DATA_DIR
    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"

    for path in (test_path, rul_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. See data/README.md for how to obtain the "
                f"C-MAPSS dataset."
            )

    test_df = _read_cmapss_file(test_path)
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, engine="python")
    true_rul = true_rul.iloc[:, 0]
    true_rul.index = range(1, len(true_rul) + 1)  # align to unit_number
    true_rul.name = "true_rul"
    return test_df, true_rul


def add_training_rul(df, cap=config.RUL_CAP):
    """
    Derive the RUL label for training data.

    Training engines run to failure, so the last recorded cycle for each
    engine IS the failure point. RUL at any earlier cycle is simply:

        RUL = (that engine's final cycle) - (current cycle)

    >>> The piecewise-linear cap

    Left uncapped, RUL is a straight line from ~300 down to 0, which
    implicitly claims a brand-new engine is detectably different from
    one at cycle 50. It isn't -- degradation hasn't begun and the
    sensors show nothing. Training on that forces the model to invent a
    signal, and it pays for the invention with accuracy near failure,
    which is the only region anyone cares about.

    Capping RUL at `cap` says: "healthy is healthy; start counting down
    only once decline is plausible." This is standard practice on
    C-MAPSS and a genuine modelling judgement, not a data-cleaning step.
    Set cap=None to disable it and see the difference for yourself.
    """
    max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df = df.copy()
    df["rul"] = max_cycles - df["time_in_cycles"]
    if cap is not None:
        df["rul"] = df["rul"].clip(upper=cap)
    return df


def add_test_rul(test_df, true_rul, cap=config.RUL_CAP):
    """
    Derive RUL for every row of the truncated test data.

    For a test engine, RUL at its LAST row is given by RUL_FD00X.txt.
    Working backwards from there:

        RUL at cycle t = (final RUL) + (final cycle - t)
    """
    max_cycles = test_df.groupby("unit_number")["time_in_cycles"].transform("max")
    final_rul = test_df["unit_number"].map(true_rul)

    test_df = test_df.copy()
    test_df["rul"] = final_rul + (max_cycles - test_df["time_in_cycles"])
    if cap is not None:
        test_df["rul"] = test_df["rul"].clip(upper=cap)
    return test_df


def constant_sensors(df, tolerance=1e-9):
    """
    Identify sensors that never vary -- they carry zero information and
    should be dropped before modelling.

    On FD001 specifically, several of the 21 sensors are flat because
    that subset has only one operating condition. Finding them
    empirically rather than hardcoding a list means the same function
    works unchanged on FD002/FD004, where different sensors vary.
    """
    sensor_std = df[config.SENSOR_COLUMNS].std()
    return sorted(sensor_std[sensor_std < tolerance].index.tolist())


def summarise(df, name="dataset"):
    """Quick structural summary -- run this before trusting any file."""
    n_units = df["unit_number"].nunique()
    cycles = df.groupby("unit_number")["time_in_cycles"].max()
    print(f"{name}: {len(df):,} rows | {n_units} engines")
    print(
        f"  cycles per engine -- min {cycles.min()}, "
        f"median {int(cycles.median())}, max {cycles.max()}"
    )
    if "rul" in df.columns:
        print(f"  RUL range: {df['rul'].min()} to {df['rul'].max()}")
