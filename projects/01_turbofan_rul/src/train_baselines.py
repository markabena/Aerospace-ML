"""
C3: baseline models for FD001 RUL prediction.

Run:  python -m src.train_baselines     (after python -m src.build_dataset)

Protocol
--------
1. Model choice is made on TRAINING data only, using 5-fold GroupKFold
   split by engine. Every engine is wholly in train or wholly in
   validation; splitting rows at random would put cycle 100 of engine 7
   in training and cycle 101 in validation, and the score would be
   flattering nonsense.
2. Each model is then refit on all 100 training engines and scored ONCE
   on the test set, at each test engine's last recorded cycle, against
   the true RUL in RUL_FD001.txt. The test set is never used to choose
   anything.

Hyperparameters are fixed up front, not tuned, so these are honest
baselines for C4 to beat.
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from . import config, data_loader as dl, metrics  # noqa: E402

SEED = 42
N_FOLDS = 5
NON_FEATURES = ("unit_number", "rul")


def make_models():
    return {
        "Mean predictor": DummyRegressor(strategy="mean"),
        "Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, max_features=0.33,
            n_jobs=-1, random_state=SEED),
        "Gradient boosting": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20, random_state=SEED),
    }


def load_features(subset):
    d = config.PROCESSED_DATA_DIR
    train_path, test_path = d / f"train_{subset}_features.csv", d / f"test_{subset}_features.csv"
    if not (train_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            f"Feature files missing in {d}. Run `python -m src.build_dataset` first.")
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    feature_cols = [c for c in train.columns if c not in NON_FEATURES]
    return train, test, feature_cols


def last_cycle_rows(df):
    """One row per engine: its final recorded cycle (the prediction point)."""
    idx = df.groupby("unit_number")["time_in_cycles"].idxmax()
    return df.loc[idx].sort_values("unit_number")


def cross_validate(model, X, y, groups):
    """
    Grouped CV scored two ways:
      - all rows of the held-out engines (RMSE)
      - one random truncation point per held-out engine, which mimics
        the test set, where histories stop part-way through life.
    """
    rng = np.random.default_rng(SEED)
    all_rmse, cut_rmse = [], []
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = np.clip(model.predict(X.iloc[va]), 0, None)
        all_rmse.append(metrics.rmse(y.iloc[va], pred))

        va_groups = groups.iloc[va].to_numpy()
        picks = [rng.choice(np.flatnonzero(va_groups == g)) for g in np.unique(va_groups)]
        cut_rmse.append(metrics.rmse(y.iloc[va].to_numpy()[picks], pred[picks]))
    return float(np.mean(all_rmse)), float(np.std(all_rmse)), float(np.mean(cut_rmse))


def plot_predictions(true_rul, preds, path):
    order = np.argsort(true_rul)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(np.arange(len(order)), true_rul[order], "k-", lw=2, label="True RUL")
    for name, p in preds.items():
        ax.plot(np.arange(len(order)), p[order], ".", ms=6, alpha=0.8, label=name)
    ax.axhline(config.RUL_CAP, color="grey", ls=":", lw=1)
    ax.text(1, config.RUL_CAP + 2, f"training cap ({config.RUL_CAP})", color="grey", fontsize=8)
    ax.set_xlabel("Test engine (sorted by true RUL)")
    ax.set_ylabel("Remaining useful life (cycles)")
    ax.set_title("FD001 test set: predicted vs true RUL at last recorded cycle")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(model, feature_cols, path, top=15):
    imp = pd.Series(model.feature_importances_, index=feature_cols).nlargest(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index, imp.values, color="#2E5BA8")
    ax.set_xlabel("Impurity-based importance")
    ax.set_title(f"Random forest: top {top} features (FD001)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return imp[::-1]


def main(subset=config.DEFAULT_SUBSET):
    t0 = time.time()
    train, test, feature_cols = load_features(subset)
    X, y, groups = train[feature_cols], train["rul"], train["unit_number"]

    test_last = last_cycle_rows(test)
    _, true_rul_series = dl.load_test(subset)
    true_rul = test_last["unit_number"].map(true_rul_series).to_numpy(dtype=float)
    X_test = test_last[feature_cols]

    print(f"C3 baselines on {subset}: {X.shape[1]} features, "
          f"{groups.nunique()} train engines, {len(X_test)} test engines\n")

    rows, test_preds, fitted = [], {}, {}
    for name, model in make_models().items():
        cv_mean, cv_std, cv_cut = cross_validate(model, X, y, groups)
        model.fit(X, y)
        pred = np.clip(model.predict(X_test), 0, None)
        s = metrics.summary(true_rul, pred)
        s_capped = metrics.rmse(np.minimum(true_rul, config.RUL_CAP), pred)
        rows.append({
            "model": name,
            "cv_rmse": cv_mean, "cv_rmse_std": cv_std, "cv_rmse_truncated": cv_cut,
            "test_rmse": s["rmse"], "test_rmse_capped_truth": s_capped,
            "test_mae": s["mae"], "test_nasa_score": s["nasa_score"],
            "test_late_pct": s["late_pct"],
        })
        test_preds[name], fitted[name] = pred, model
        print(f"  {name:<18} CV RMSE {cv_mean:6.2f} ± {cv_std:4.2f} | "
              f"test RMSE {s['rmse']:6.2f} | NASA score {s['nasa_score']:9.1f} | "
              f"late {s['late_pct']:4.1f}%")

    results = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.RESULTS_DIR / f"c3_baselines_{subset}.csv", index=False, float_format="%.3f")

    per_engine = pd.DataFrame({"unit_number": test_last["unit_number"].to_numpy(), "true_rul": true_rul})
    for name, p in test_preds.items():
        per_engine[name] = np.round(p, 2)
    per_engine.to_csv(config.RESULTS_DIR / f"c3_test_predictions_{subset}.csv", index=False)

    plot_predictions(true_rul, {k: v for k, v in test_preds.items() if k != "Mean predictor"},
                     config.RESULTS_DIR / f"c3_pred_vs_true_{subset}.png")
    imp = plot_importance(fitted["Random forest"], feature_cols,
                          config.RESULTS_DIR / f"c3_feature_importance_{subset}.png")
    print("\nTop 5 random-forest features:")
    for f, v in imp.head(5).items():
        print(f"  {f:<32} {v:.3f}")

    print(f"\nResults written to {config.RESULTS_DIR}/  ({time.time()-t0:.0f}s)")
    return results


if __name__ == "__main__":
    main()
