# P5 · Engine Health Monitor

The first machine-learning project in the portfolio. A random forest classifier flags degraded rocket engines from static-fire sensor readings. It has the same problem shape as the capstone (predictive maintenance), on different hardware.

> **Data note:** `Merlin_Engine_Test_Data.csv` holds 1,000 **simulated** static-fire records for a Merlin-class LOX/RP-1 engine, with nominal values in a realistic range. Three physically motivated degradation modes are mixed in: turbopump bearing wear, injector fouling and coolant channel blockage. 15% of records are degraded, on purpose, to force the class-imbalance question.

## Results (200 held-out records)

| Metric | Value |
|---|---|
| Always-guess-nominal baseline | 85.0% accuracy |
| Random forest accuracy | 98.0% |
| Degraded recall (engines caught) | 90.0% (27 of 30) |
| Degraded precision | 96.4% |
| 5-fold cross-validated macro recall | 0.968 ± 0.021 |
| Logistic regression comparison | 97.5% accuracy |

**Why accuracy alone is a trap:** a model that never flags anything already scores 85%. The number that matters is **missed degraded engines** (3 here). A miss puts a failing engine back on the stand; a false alarm (1 here) costs only a teardown.

**Top features:** vibration RMS (0.274), chamber pressure (0.261), fuel flow (0.158), exhaust gas temperature (0.138). These are physically consistent with bearing wear and injector fouling.

## Run

```bash
python engine_health_monitor.py
```

Writes `Engine_Health_Predictions.csv` so any wrong call can be traced to a specific test record.

## Concepts

Features vs labels, stratified train/test split, scaling pipelines, confusion matrix, precision and recall, cross-validation, feature importance.
