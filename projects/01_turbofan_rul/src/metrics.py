"""
Evaluation metrics for RUL prediction.

Two numbers are reported for every model:

RMSE
    The usual average error, in cycles. Symmetric: being 10 cycles early
    costs the same as being 10 cycles late.

NASA scoring function  (Saxena et al., 2008, PHM'08 challenge)
    Asymmetric and exponential. With d = predicted - true:

        s = exp(-d / 13) - 1    if d < 0   (early: engine pulled too soon)
        s = exp( d / 10) - 1    if d >= 0  (late:  engine kept in service)

    A late prediction is penalised more heavily than an early one of the
    same size, because a late call means a degrading engine stays on the
    wing. Lower is better; a perfect model scores 0. Scores are summed
    over engines, so they are only comparable on the same test set.
"""
import numpy as np

EARLY_SCALE = 13.0
LATE_SCALE = 10.0


def _as_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def rmse(y_true, y_pred):
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def nasa_score_per_engine(y_true, y_pred):
    """Per-engine penalty, so you can see which engines drive the total."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    d = y_pred - y_true
    return np.where(d < 0, np.exp(-d / EARLY_SCALE) - 1, np.exp(d / LATE_SCALE) - 1)


def nasa_score(y_true, y_pred):
    return float(np.sum(nasa_score_per_engine(y_true, y_pred)))


def late_fraction(y_true, y_pred):
    """Share of engines where the model predicted MORE life than remained."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.mean(y_pred > y_true))


def summary(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "late_pct": 100 * late_fraction(y_true, y_pred),
    }
