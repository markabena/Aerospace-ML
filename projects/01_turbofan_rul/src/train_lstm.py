"""
C4: LSTM sequence model for FD001 RUL prediction.

Run:  python -m src.train_lstm        (needs PyTorch: pip install torch)

What changes from C3
--------------------
C3's tree models saw one row at a time, with history compressed into
hand-built rolling features. Here the model reads the last WINDOW cycles
of raw (scaled) sensor data and learns its own summary of the trajectory.

Protocol
--------
- Inputs: the 14 sensors kept by C2's selection, min-max scaled with
  training statistics only.
- 20 of the 100 training engines are held out as a validation set, used
  only for early stopping. The split is by engine, never by window.
- Neural nets are noisy from run to run, so the model is trained with
  N_SEEDS different seeds. We report the mean and spread across seeds,
  plus an ensemble that averages the seeds' predictions.
- The test set is scored once at the end, at each engine's last cycle,
  against RUL_FD001.txt. Nothing is tuned on it.
"""
import copy
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from . import config, data_loader as dl, features as ft, metrics  # noqa: E402
from . import sequences as sq  # noqa: E402

WINDOW = 30          # shortest FD001 test engine has 31 cycles
N_VAL_ENGINES = 20
N_SEEDS = 5
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.2
BATCH = 256
LR = 1e-3
MAX_EPOCHS = 80
PATIENCE = 10
SPLIT_SEED = 42
C3_BEST = {"model": "Gradient boosting (C3)", "test_rmse": 12.884, "test_nasa_score": 223.9}


class RULLSTM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN, num_layers=LAYERS,
                            batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(nn.Linear(HIDDEN, 32), nn.ReLU(),
                                  nn.Dropout(DROPOUT), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)   # read the final time step


def prepare(subset):
    train = dl.add_training_rul(dl.load_train(subset))
    test_raw, true_rul = dl.load_test(subset)
    test = dl.add_test_rul(test_raw, true_rul)

    sensors, _ = ft.select_sensors(train, verbose=False)
    scaler = sq.MinMaxScaler().fit(train, sensors)
    train_s, test_s = scaler.transform(train), scaler.transform(test)

    X, y, units = sq.make_training_windows(train_s, sensors, WINDOW)
    X_test, test_units = sq.make_last_windows(test_s, sensors, WINDOW)
    y_test = true_rul.loc[test_units].to_numpy(dtype=float)   # uncapped truth
    return X, y, units, X_test, y_test, test_units, sensors


def split_engines(units):
    rng = np.random.default_rng(SPLIT_SEED)
    val_units = rng.choice(np.unique(units), size=N_VAL_ENGINES, replace=False)
    is_val = np.isin(units, val_units)
    return ~is_val, is_val


def predict(model, X):
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(X)).numpy() * config.RUL_CAP
    return np.clip(p, 0, None)


def train_one(seed, X_tr, y_tr, X_va, y_va):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RULLSTM(X_tr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(X_tr)
    yt = torch.from_numpy(y_tr / config.RUL_CAP)    # train on 0-1 scale
    gen = torch.Generator().manual_seed(seed)

    best, best_state, best_epoch, wait, history = np.inf, None, 0, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        tr_loss = 0.0
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(idx)
        tr_rmse = np.sqrt(tr_loss / len(Xt)) * config.RUL_CAP
        va_rmse = metrics.rmse(y_va, predict(model, X_va))
        history.append((epoch, tr_rmse, va_rmse))

        if va_rmse < best - 1e-3:
            best, best_state, best_epoch, wait = va_rmse, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best, best_epoch, history


def plot_curves(histories, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for seed, h in histories.items():
        h = np.array(h)
        line, = ax.plot(h[:, 0], h[:, 2], lw=1.5, label=f"seed {seed} val")
        ax.plot(h[:, 0], h[:, 1], lw=1, ls="--", color=line.get_color(), alpha=0.5)
    ax.set_ylim(0, 40)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title("LSTM training (dashed) vs validation (solid) RMSE, FD001")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_vs_c3(y_test, lstm_pred, gb_pred, path):
    order = np.argsort(y_test)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, y_test[order], "k-", lw=2, label="True RUL")
    if gb_pred is not None:
        ax.plot(x, gb_pred[order], "o", ms=4, alpha=0.6, color="#2ca02c", label="Gradient boosting (C3)")
    ax.plot(x, lstm_pred[order], "o", ms=4, alpha=0.8, color="#E8700A", label="LSTM ensemble (C4)")
    ax.axhline(config.RUL_CAP, color="grey", ls=":", lw=1)
    ax.set_xlabel("Test engine (sorted by true RUL)")
    ax.set_ylabel("Remaining useful life (cycles)")
    ax.set_title("FD001 test set: LSTM vs gradient boosting")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def band_rmse(y, p):
    bands = {"<=50": y <= 50, "50-100": (y > 50) & (y <= 100), ">100": y > 100}
    return {k: metrics.rmse(y[m], p[m]) for k, m in bands.items()}


def main(subset=config.DEFAULT_SUBSET):
    t0 = time.time()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    X, y, units, X_test, y_test, test_units, sensors = prepare(subset)
    tr, va = split_engines(units)
    print(f"C4 LSTM on {subset}: {len(sensors)} sensors, window {WINDOW}, "
          f"{tr.sum():,} train windows / {va.sum():,} val windows, "
          f"{len(X_test)} test engines\n")

    rows, preds, histories = [], {}, {}
    for seed in range(N_SEEDS):
        model, val_rmse, best_epoch, hist = train_one(seed, X[tr], y[tr], X[va], y[va])
        p = predict(model, X_test)
        s = metrics.summary(y_test, p)
        preds[seed], histories[seed] = p, hist
        rows.append({"model": f"LSTM seed {seed}", "best_epoch": best_epoch, "val_rmse": val_rmse,
                     "test_rmse": s["rmse"], "test_mae": s["mae"],
                     "test_nasa_score": s["nasa_score"], "test_late_pct": s["late_pct"]})
        print(f"  seed {seed}: best epoch {best_epoch:3d} | val RMSE {val_rmse:5.2f} | "
              f"test RMSE {s['rmse']:5.2f} | NASA {s['nasa_score']:7.1f}  ({time.time()-t0:.0f}s)")

    per_seed = pd.DataFrame(rows)
    ens = np.mean(np.stack(list(preds.values())), axis=0)
    s = metrics.summary(y_test, ens)
    summary_rows = [
        {"model": "LSTM, mean of seeds",
         "test_rmse": per_seed.test_rmse.mean(), "test_rmse_std": per_seed.test_rmse.std(),
         "test_nasa_score": per_seed.test_nasa_score.mean(),
         "test_nasa_std": per_seed.test_nasa_score.std()},
        {"model": "LSTM ensemble", "test_rmse": s["rmse"], "test_mae": s["mae"],
         "test_nasa_score": s["nasa_score"], "test_late_pct": s["late_pct"]},
        C3_BEST,
    ]
    results = pd.concat([per_seed, pd.DataFrame(summary_rows)], ignore_index=True)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.RESULTS_DIR / f"c4_lstm_{subset}.csv", index=False, float_format="%.3f")

    per_engine = pd.DataFrame({"unit_number": test_units, "true_rul": y_test,
                               "lstm_ensemble": np.round(ens, 2)})
    for seed, p in preds.items():
        per_engine[f"lstm_seed_{seed}"] = np.round(p, 2)
    per_engine.to_csv(config.RESULTS_DIR / f"c4_test_predictions_{subset}.csv", index=False)

    gb_path = config.RESULTS_DIR / f"c3_test_predictions_{subset}.csv"
    gb = None
    if gb_path.exists():
        gb_df = pd.read_csv(gb_path).set_index("unit_number")
        gb = gb_df.loc[test_units, "Gradient boosting"].to_numpy()

    plot_curves(histories, config.RESULTS_DIR / f"c4_training_curves_{subset}.png")
    plot_vs_c3(y_test, ens, gb, config.RESULTS_DIR / f"c4_lstm_vs_gb_{subset}.png")

    print(f"\nLSTM mean of {N_SEEDS} seeds: RMSE {per_seed.test_rmse.mean():.2f} "
          f"± {per_seed.test_rmse.std():.2f} | NASA {per_seed.test_nasa_score.mean():.1f} "
          f"± {per_seed.test_nasa_score.std():.1f}")
    print(f"LSTM ensemble:          RMSE {s['rmse']:.2f} | NASA {s['nasa_score']:.1f} | "
          f"late {s['late_pct']:.0f}%")
    print(f"C3 gradient boosting:   RMSE {C3_BEST['test_rmse']:.2f} | NASA {C3_BEST['test_nasa_score']:.1f}")
    print("\nRMSE by true-RUL band:")
    print(f"  LSTM ensemble: { {k: round(v, 1) for k, v in band_rmse(y_test, ens).items()} }")
    if gb is not None:
        print(f"  GB (C3):       { {k: round(v, 1) for k, v in band_rmse(y_test, gb).items()} }")
    print(f"\nResults written to {config.RESULTS_DIR}/  ({time.time()-t0:.0f}s)")
    return results


if __name__ == "__main__":
    main()
