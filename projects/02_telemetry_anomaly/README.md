# 02 · Spacecraft Telemetry Anomaly Detection

## Problem
Spacecraft send thousands of telemetry channels to ground. Operators can't watch every channel by eye, so faults hide in the noise until they escalate. An automated detector that learns normal behaviour and flags deviations lets operators react early.

## Data
**NASA SMAP and MSL telemetry anomaly dataset** (Soil Moisture Active Passive satellite and the Curiosity rover), with expert-labelled anomaly windows.

## Approach
1. Explore per-channel signals and the labelled anomaly windows.
2. Baseline: rolling z-score and Isolation Forest.
3. Deep model: LSTM forecaster, with anomalies flagged when prediction error is high (dynamic thresholding), plus an autoencoder comparison.
4. Evaluate: precision, recall and F1 at the event level.

## Results
_To be added._

## Run
_To be added._
