# Foundations Track

Five tools built in order, each adding the next skill on the way to the predictive-maintenance capstone in [`projects/01_turbofan_rul`](../projects/01_turbofan_rul/). The track runs from coding a known formula (P1, P2), to handling messy data (P3, P4), to learning a rule from data when no formula exists (P5).

| # | Project | New skills | Data |
|---|---------|-----------|------|
| P1 | [Rocket Performance Evaluator](p1_rocket_performance/) | Functions, validation loops, rocket equation | User input |
| P2 | [Orbital Parameter Calculator](p2_orbital_parameters/) | Dictionaries, tuples, error handling, vis-viva | Validated against real orbits |
| P3 | [Flight Data Analyser](p3_flight_data_analyser/) | pandas cleaning, masking, rolling windows, event detection | Simulated ascent telemetry |
| P4 | [Flight Visualiser](p4_flight_visualiser/) | matplotlib subplots, event annotation | P3's output |
| P5 | [Engine Health Monitor](p5_engine_health_monitor/) | Classification, class imbalance, precision/recall, cross-validation | Simulated static-fire records |

P3 and P4 are a pipeline: P3 writes the cleaned data, and P4 only draws it. Run P3 first.

```bash
python p3_flight_data_analyser/flight_data_analyser.py
python p4_flight_visualiser/flight_visualiser.py
```

The scripts resolve file paths from their own location, so they run from any directory. Their results are checked in CI by [`tests/test_foundations.py`](../tests/test_foundations.py).
