"""
Checks for src/sequences.py: windows stay inside one engine, labels line
up with the window's last cycle, and the scaler only learns from train.

Run:  python tests/test_sequences.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import sequences as sq  # noqa: E402


def check(name, condition):
    print(f"  {'ok ' if condition else 'FAIL'}  {name}")
    return condition


def toy():
    # Engine 1: 6 cycles valued 1..6. Engine 2: 4 cycles valued 101..104.
    # RUL counts down to 0 at each engine's last cycle.
    return pd.DataFrame({
        "unit_number": [1] * 6 + [2] * 4,
        "time_in_cycles": list(range(1, 7)) + list(range(1, 5)),
        "s": [1, 2, 3, 4, 5, 6, 101, 102, 103, 104],
        "rul": [5, 4, 3, 2, 1, 0, 3, 2, 1, 0],
    })


def main():
    r = []
    df = toy()

    X, y, units = sq.make_training_windows(df, ["s"], window=3)
    r.append(check("one window per cycle from cycle 3 onward (4 + 2 = 6)", len(X) == 6))
    r.append(check("shape is (n, window, features)", X.shape == (6, 3, 1)))

    mixes = any((w.min() < 100) and (w.max() > 100) for w in X[:, :, 0])
    r.append(check("no window mixes two engines", not mixes))

    first = X[0, :, 0].tolist()
    r.append(check("first window is cycles 1-3 of engine 1", first == [1, 2, 3]))
    r.append(check("label is RUL at the window's LAST cycle", y[0] == 3 and y[3] == 0))
    r.append(check("engine 2 windows labelled 1 then 0", y[units == 2].tolist() == [1, 0]))

    # Short history: engine 2 has 4 cycles, ask for a 6-cycle window
    Xs, ys, us = sq.make_training_windows(df, ["s"], window=6)
    e2 = Xs[us == 2]
    r.append(check("short engine yields one front-padded window",
                   len(e2) == 1 and e2[0, :, 0].tolist() == [101, 101, 101, 102, 103, 104]))
    r.append(check("padded window labelled with RUL at the last cycle", ys[us == 2].tolist() == [0]))

    Xl, ul = sq.make_last_windows(df, ["s"], window=3)
    r.append(check("last windows: one per engine, final cycles",
                   Xl[:, :, 0].tolist() == [[4, 5, 6], [102, 103, 104]] and ul.tolist() == [1, 2]))

    shuffled = df.sample(frac=1, random_state=0)
    Xsh, ysh, _ = sq.make_training_windows(shuffled, ["s"], window=3)
    r.append(check("row order in the input doesn't matter", np.array_equal(Xsh, X) and np.array_equal(ysh, y)))

    train = pd.DataFrame({"s": [0.0, 10.0], "c": [5.0, 5.0]})
    test = pd.DataFrame({"s": [20.0], "c": [5.0]})
    sc = sq.MinMaxScaler().fit(train, ["s", "c"])
    out = sc.transform(test)
    r.append(check("scaler uses train range only (test 20 -> 2.0)", out["s"].iloc[0] == 2.0))
    r.append(check("constant column scales without divide-by-zero", out["c"].iloc[0] == 0.0))

    if all(r):
        print(f"\nAll {len(r)} sequence checks PASSED")
        return 0
    print(f"\n{r.count(False)} of {len(r)} sequence checks FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
