# P3 · Flight Data Analyser

Cleans a noisy first-stage ascent telemetry file with pandas and extracts the key ascent events: maximum dynamic pressure (max-Q), the throttle-down window, and main engine cutoff (MECO) conditions.

> **Data note:** `Falcon9_Ascent_Telemetry.csv` is **simulated**. It comes from a simplified vertical-ascent model using public, approximate Falcon 9 Block 5 parameters, with sensor noise and dropped readings added so there is something real to clean. It is not flight telemetry.

## Results

| Event | Value |
|---|---|
| Max-Q | 28.02 kPa at T+54 s |
| Throttle-down window | T+53 s to T+84 s |
| MECO | T+162 s, 118.83 km altitude, 2,441.7 m/s |
| Missing readings repaired | 4 → 0 (interpolation) |

## Run

```bash
python flight_data_analyser.py
```

Writes `Falcon9_Ascent_Telemetry_Cleaned.csv` and `Falcon9_Ascent_Summary.csv` next to the script. P4 reads both. Plain CSV is also the bridge to MATLAB (`readtable` / `writematrix`).

## Concepts

DataFrames, boolean masking, aggregation, missing-data handling, rolling windows, `.diff()` step detection, CSV export.
