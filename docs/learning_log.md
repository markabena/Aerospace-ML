# Learning Log

What each build taught, covering the Python, the ML and the engineering lesson. New concepts are learned inside a project, not in isolation.

| Build | Python / ML concepts | Engineering lesson |
|-------|----------------------|--------------------|
| P1 · Rocket Performance | Functions, constants, input validation, `math` | State model assumptions next to the results so the numbers aren't over-read |
| P2 · Orbital Parameters | Dictionaries, tuples, `try/except` | Validate against known real orbits before trusting a calculator |
| P3 · Flight Data Analyser | pandas, boolean masking, interpolation, rolling windows, `.diff()` | Detect events by *steps*, not levels, since acceleration rises naturally as mass burns off |
| P4 · Flight Visualiser | matplotlib Figure/Axes, shared axes, spans, annotations | Keep analysis and presentation in separate scripts |
| P5 · Engine Health Monitor | Train/test split, pipelines, scaling, random forest, precision/recall, CV | With 85/15 class balance, accuracy is misleading; track missed failures |
| C1 · Turbofan data | Headerless file parsing, label derivation, mock-data testing | The RUL cap is a modelling decision, not a data fact |
| C2 · Turbofan features | Grouped rolling windows, trend slopes, correlation filters | Test explicitly for cross-engine and future-information leakage |
