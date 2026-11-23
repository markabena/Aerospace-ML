"""
Turn per-cycle rows into fixed-length sequences for a sequence model (C4).

The C3 models saw one row at a time, with history squeezed into
hand-built rolling features. An LSTM reads the raw history directly: each
sample is the last `window` cycles of one engine, and the label is the
RUL at the final cycle of that window.

Pure numpy/pandas, so it can be tested without PyTorch installed.
"""
import numpy as np


class MinMaxScaler:
    """
    Scale each sensor to [0, 1] using TRAINING min/max only.

    Fitting on train + test would let the test set decide the scale,
    which is a quiet form of leakage. Test values may land slightly
    outside [0, 1]; that is expected and harmless.
    """

    def fit(self, df, columns):
        self.columns = list(columns)
        self.min_ = df[self.columns].min()
        span = df[self.columns].max() - self.min_
        self.span_ = span.where(span > 0, 1.0)  # constant column -> no divide-by-zero
        return self

    def transform(self, df):
        out = df.copy()
        out[self.columns] = (df[self.columns] - self.min_) / self.span_
        return out


def _engine_arrays(df, columns):
    """Yield (unit_number, feature matrix, rul vector) per engine, in cycle order."""
    for unit, g in df.sort_values(["unit_number", "time_in_cycles"]).groupby("unit_number"):
        yield unit, g[columns].to_numpy(dtype=np.float32), g["rul"].to_numpy(dtype=np.float32)


def _pad_front(x, window):
    """Repeat the first row so a short history still fills the window."""
    if len(x) >= window:
        return x
    pad = np.repeat(x[:1], window - len(x), axis=0)
    return np.vstack([pad, x])


def make_training_windows(df, columns, window):
    """
    Every window of `window` consecutive cycles within each engine.

    Windows never span two engines: they are built engine by engine.
    Label = RUL at the window's LAST cycle, so the model only ever sees
    the past relative to the moment it is predicting.

    Returns X (n, window, n_features), y (n,), units (n,).
    """
    X, y, units = [], [], []
    for unit, feats, rul in _engine_arrays(df, columns):
        if len(feats) < window:
            # Too short for even one full window: use a single front-padded
            # window ending at the engine's last cycle.
            X.append(_pad_front(feats, window))
            y.append(rul[-1])
            units.append(unit)
            continue
        for e in range(window - 1, len(feats)):  # e = last cycle of the window
            X.append(feats[e - window + 1: e + 1])
            y.append(rul[e])
            units.append(unit)
    return np.stack(X), np.asarray(y, dtype=np.float32), np.asarray(units)


def make_last_windows(df, columns, window):
    """
    One window per engine: its final `window` cycles (front-padded if the
    history is shorter). This is the prediction point for the test set.

    Returns X (n_engines, window, n_features), units (n_engines,).
    """
    X, units = [], []
    for unit, feats, _ in _engine_arrays(df, columns):
        X.append(_pad_front(feats, window)[-window:])
        units.append(unit)
    return np.stack(X), np.asarray(units)
