# Turbofan Engine Remaining Useful Life (RUL) Prediction

Predicting how many operating cycles a turbofan engine has left before
failure, using NASA's C-MAPSS run-to-failure dataset.

**Status:** in progress — Phase C1 (data loading and RUL labelling) complete.

---

## The problem

Given a stream of sensor readings from an aircraft engine, estimate its
**Remaining Useful Life** — the number of operating cycles before failure.
Predict too high and a degrading engine stays in service. Predict too low
and a healthy engine is pulled for unnecessary maintenance. The two errors
are not equally costly, and the evaluation reflects that.

C-MAPSS is the standard public benchmark for this problem, which means
results here are directly comparable to published work.

## Dataset

Four subsets of increasing difficulty, from the NASA Prognostics Center of
Excellence:

| Subset | Train / Test engines | Operating conditions | Fault modes |
|---|---|---|---|
| FD001 | 100 / 100 | 1 | 1 (HPC degradation) |
| FD002 | 260 / 259 | 6 | 1 |
| FD003 | 100 / 100 | 1 | 2 (HPC + fan) |
| FD004 | 248 / 249 | 6 | 2 |

Each row: `unit_number`, `time_in_cycles`, 3 operational settings, 21 sensors.

Training engines run to failure. Test engine histories are **truncated**
partway through, with ground-truth RUL supplied separately — so the task
is genuinely predictive, not interpolative.

The raw data is not committed here. See [`data/README.md`](data/README.md)
for how to obtain it.

## Key modelling decision: the RUL cap

RUL for training data is derived, not given:

```
RUL at cycle t = (engine's final cycle) - t
```

Left uncapped, this produces a straight line from ~300 down to 0 — which
implicitly asserts that a brand-new engine is detectably different from
one at cycle 50. It isn't. Degradation hasn't started, and the sensors
show nothing distinguishing.

Training against that forces the model to fit a signal that doesn't exist,
and it pays for the invention with accuracy near failure — the only region
that matters operationally.

This project applies the standard **piecewise-linear cap** (default 125
cycles): RUL is held flat while the engine is healthy, and counts down
only once decline is plausible. Set `cap=None` in `add_training_rul()` to
disable it and compare.

## Project structure

```
├── data/
│   ├── raw/              # C-MAPSS .txt files (not committed)
│   ├── processed/        # engineered feature sets
│   └── README.md         # how to obtain the dataset
├── src/
│   ├── config.py         # schema, paths, subset metadata
│   └── data_loader.py    # loading + RUL labelling
├── tests/
│   ├── make_mock_data.py # generates format-accurate mock files
│   └── test_data_loader.py
├── results/
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Then follow [`data/README.md`](data/README.md) to place the raw files.

## Verifying the loader without the dataset

The loader is tested against mock files that reproduce C-MAPSS's exact
format, including the trailing-whitespace quirk that produces phantom
columns when read naively:

```bash
python tests/make_mock_data.py
python tests/test_data_loader.py
```

Covers column parsing, RUL derivation for training and truncated test
data, the cap, and constant-sensor detection.

## Roadmap

- [x] **C1** — Data loading, RUL labelling, format verification
- [ ] **C2** — Feature engineering: rolling statistics, trend slopes, sensor selection
- [ ] **C3** — Baseline models (random forest / gradient boosting) + NASA scoring function
- [ ] **C4** — LSTM sequence model *(scope decision pending C3 results)*
- [ ] **C5** — Results write-up and comparison against published benchmarks

## Citation

A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation
Data Set", NASA Prognostics Data Repository, NASA Ames Research Center,
Moffett Field, CA.
