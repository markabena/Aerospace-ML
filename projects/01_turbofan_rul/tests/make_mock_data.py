"""
Generate mock files that reproduce the C-MAPSS format EXACTLY -- including
the trailing-whitespace quirk -- so the loader is verified before the real
data arrives. Values are meaningless; only the structure matters.
"""
import numpy as np
from pathlib import Path

rng = np.random.default_rng(1)
out = Path(__file__).parent / "mock_data"
out.mkdir(exist_ok=True)

def write_file(path, engine_lifetimes, flat_sensors=(1, 5, 10)):
    lines = []
    for unit, lifetime in enumerate(engine_lifetimes, start=1):
        for cycle in range(1, lifetime + 1):
            settings = [f"{v:.4f}" for v in rng.normal(0, 0.002, 3)]
            sensors = []
            for s in range(1, 22):
                if s in flat_sensors:
                    sensors.append("100.00")          # constant sensor
                else:
                    drift = (cycle / lifetime) * 2.0  # degradation trend
                    sensors.append(f"{rng.normal(500 + drift, 0.5):.2f}")
            # NOTE the trailing space -- real C-MAPSS files have this
            lines.append(f"{unit} {cycle} " + " ".join(settings + sensors) + " ")
    path.write_text("\n".join(lines) + "\n")

# Training: run-to-failure
train_lifetimes = [150, 200, 175]
write_file(out / "train_FD001.txt", train_lifetimes)

# Test: truncated before failure
test_lifetimes = [80, 120, 95]
write_file(out / "test_FD001.txt", test_lifetimes)

# Ground truth RUL at each test engine's final cycle
(out / "RUL_FD001.txt").write_text("70\n45\n130\n")

print(f"Mock files written to {out}")
for f in sorted(out.iterdir()):
    print(f"  {f.name}")
