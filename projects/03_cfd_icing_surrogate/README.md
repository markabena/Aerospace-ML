# 03 · CFD Icing Surrogate Model

## Problem
A single CFD icing simulation takes hours. Design studies need hundreds of them. A machine-learning surrogate trained on a set of CFD runs can predict results in milliseconds and make wide design sweeps practical.

## Data
CFD results from my B.Eng final-year icing study (ANSYS), covering inputs such as airspeed, temperature, liquid water content, droplet size and angle of attack, and outputs such as ice mass and aerodynamic penalties. Data will be published only where permitted.

## Approach
1. Assemble the input–output table from the CFD cases and export it via CSV (MATLAB/ANSYS → pandas).
2. Baselines: polynomial regression and Gaussian Process regression.
3. Neural surrogate: small multilayer perceptron (PyTorch).
4. Evaluate: held-out error, uncertainty estimates, and comparison against unseen CFD cases.

## Results
_To be added._

## Run
_To be added._
