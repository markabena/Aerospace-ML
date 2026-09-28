# Obtaining the C-MAPSS dataset

The raw dataset is **not committed to this repository**. Download it and
place the files in `data/raw/`.

## Source

NASA Prognostics Center of Excellence (PCoE) Data Set Repository:
<https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>

Look for **"Turbofan Engine Degradation Simulation Data Set"** (dataset 6).

## Expected files

After extracting the archive, `data/raw/` should contain:

```
data/raw/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── train_FD002.txt
...
└── RUL_FD004.txt
```

This project starts with **FD001** — one operating condition, one fault
mode — and extends to the harder subsets later.

## File format

Space-separated, no header row, 26 columns in fixed order:

| Position | Column |
|---|---|
| 1 | `unit_number` |
| 2 | `time_in_cycles` |
| 3–5 | `op_setting_1` … `op_setting_3` |
| 6–26 | `sensor_1` … `sensor_21` |

Two quirks the loader in `src/data_loader.py` already handles:

1. **No header** — column names are supplied from `src/config.py`.
2. **Trailing whitespace on every line** — read naively, this produces two
   extra all-NaN columns. The loader collapses whitespace runs instead.

`RUL_FD00X.txt` is a single column: the true remaining cycles for each
test engine at its final recorded row, in `unit_number` order.

## Citation

A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation
Data Set", NASA Prognostics Data Repository, NASA Ames Research Center,
Moffett Field, CA.
