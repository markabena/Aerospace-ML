# 01 · Turbofan Remaining Useful Life Predictor

## Problem
Unplanned engine removals are among the most expensive events in aircraft operations. If you can predict how many cycles an engine has left before failure, you can schedule maintenance on your own terms instead of reacting to failures.

## Data
**NASA C-MAPSS** turbofan engine degradation simulation dataset (NASA Prognostics Center of Excellence data repository). It has four sub-datasets (FD001–FD004) covering different operating conditions and fault modes, with 21 sensor channels per engine cycle.

## Approach
1. Explore the data: sensor trends, correlations, and dropping constant channels.
2. Engineer features: rolling statistics, per-condition normalisation, and a piecewise-linear RUL target.
3. Baseline: Random Forest / Gradient Boosting (scikit-learn).
4. Deep model: LSTM on sliding windows (PyTorch).
5. Evaluate: RMSE and the NASA asymmetric scoring function, compared against published benchmarks.

## Results
_To be added._

## Run
_To be added._
