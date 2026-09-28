import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src import features as ft
from src import config

print("="*64)
print("TEST 1: rolling windows must NOT cross engine boundaries")
# Engine 1 flat at 100, engine 2 flat at 200. If the window leaks across
# the boundary, engine 2's first rolling mean will be pulled below 200.
df = pd.DataFrame({
    "unit_number": [1]*10 + [2]*10,
    "time_in_cycles": list(range(1,11))*2,
    "sensor_2": [100.0]*10 + [200.0]*10,
})
out = ft.add_rolling_features(df, ["sensor_2"], windows=(5,))
e2_first = out[out.unit_number==2]["sensor_2_mean_5"].iloc[0]
print(f"  Engine 2 first rolling mean: {e2_first}  (must be exactly 200.0)")
assert e2_first == 200.0, "LEAK: engine 1 bled into engine 2"
e1_last = out[out.unit_number==1]["sensor_2_mean_5"].iloc[-1]
assert e1_last == 100.0
print("  No cross-engine leakage. PASS\n")

print("TEST 2: rolling windows must look BACKWARD only (no future info)")
# Step change at cycle 6. A backward window must show NO movement before it.
df = pd.DataFrame({
    "unit_number": [1]*10,
    "time_in_cycles": range(1,11),
    "sensor_2": [10.0]*5 + [90.0]*5,
})
out = ft.add_rolling_features(df, ["sensor_2"], windows=(3,))
pre = out["sensor_2_mean_3"].iloc[:5].unique()
print(f"  Rolling means before the step: {pre}  (must all be 10.0)")
assert (pre == 10.0).all(), "LEAK: future values visible before they occur"
print(f"  First value after step: {out['sensor_2_mean_3'].iloc[5]:.2f} (reacts only after)")
print("  Causal / backward-looking. PASS\n")

print("TEST 3: slope sign and magnitude")
df = pd.DataFrame({
    "unit_number": [1]*20 + [2]*20,
    "time_in_cycles": list(range(1,21))*2,
    # engine 1 rises by 2.0/cycle, engine 2 falls by 5.0/cycle
    "sensor_2": [10.0+2.0*i for i in range(20)] + [500.0-5.0*i for i in range(20)],
})
out = ft.add_rolling_features(df, ["sensor_2"], windows=(5,))
s1 = out[out.unit_number==1]["sensor_2_slope_5"].iloc[-1]
s2 = out[out.unit_number==2]["sensor_2_slope_5"].iloc[-1]
print(f"  Engine 1 slope: {s1:+.3f}  (expected +2.000)")
print(f"  Engine 2 slope: {s2:+.3f}  (expected -5.000)")
assert abs(s1-2.0) < 1e-9 and abs(s2+5.0) < 1e-9
print("  Slope recovers true gradient. PASS\n")

print("TEST 4: baseline deviation is per-engine, not global")
# Two engines with different starting levels, same degradation amount.
df = pd.DataFrame({
    "unit_number": [1]*30 + [2]*30,
    "time_in_cycles": list(range(1,31))*2,
    "sensor_2": [100.0]*20 + [110.0]*10 + [500.0]*20 + [510.0]*10,
})
out = ft.add_baseline_deviation(df, ["sensor_2"], baseline_cycles=20)
d1 = out[out.unit_number==1]["sensor_2_dev_from_baseline"].iloc[-1]
d2 = out[out.unit_number==2]["sensor_2_dev_from_baseline"].iloc[-1]
print(f"  Engine 1 (base 100 -> 110): deviation {d1:+.1f}")
print(f"  Engine 2 (base 500 -> 510): deviation {d2:+.1f}")
assert abs(d1-10.0)<1e-9 and abs(d2-10.0)<1e-9, "baseline not per-engine"
print("  Identical degradation -> identical feature despite 5x level gap. PASS\n")

print("TEST 5: no NaNs leak into engineered features")
rng = np.random.default_rng(0)
df = pd.DataFrame({
    "unit_number": np.repeat([1,2,3], 40),
    "time_in_cycles": list(range(1,41))*3,
    "sensor_2": rng.normal(500,5,120),
    "sensor_4": rng.normal(1400,9,120),
})
out, names = ft.build_features(df, ["sensor_2","sensor_4"])
eng = [c for c in out.columns if any(t in c for t in ("_mean_","_std_","_slope_","_dev_"))]
n_nan = int(out[eng].isna().sum().sum())
print(f"  Engineered columns: {len(eng)} | NaNs: {n_nan}")
assert n_nan == 0
print("  Clean. PASS\n")

print("TEST 6: sensor selection drops the right things")
rng = np.random.default_rng(2)
n = 600
rul = np.tile(np.arange(200,0,-1), 3)
df = pd.DataFrame({
    "unit_number": np.repeat([1,2,3], 200),
    "time_in_cycles": list(range(1,201))*3,
    "rul": rul,
})
for s in config.SENSOR_COLUMNS:
    df[s] = rng.normal(500, 5, n)
df["sensor_1"]  = 518.67                      # zero variance
df["sensor_6"]  = np.where(rng.random(n)>0.5, 21.60, 21.61)  # 2 distinct only
df["sensor_11"] = 50 - rul*0.05 + rng.normal(0,0.3,n)        # strong signal
kept, stats = ft.select_sensors(df, verbose=True)
print()
assert "sensor_1" not in kept, "failed to drop zero-variance sensor"
assert "sensor_6" not in kept, "failed to drop near-binary sensor_6"
assert "sensor_11" in kept, "wrongly dropped a strong-signal sensor"
print("  sensor_1 dropped (zero var), sensor_6 dropped (2 distinct), sensor_11 kept. PASS\n")

print("="*64)
print("ALL FEATURE TESTS PASSED")
