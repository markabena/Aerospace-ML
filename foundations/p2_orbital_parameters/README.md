# P2 · Orbital Parameter Calculator

Computes the parameters that describe an orbit around Earth, the Moon or Mars from either a single altitude (circular) or a periapsis/apoapsis pair (elliptical): orbital velocity, period, semi-major axis, eccentricity, and velocity at periapsis and apoapsis via the vis-viva equation.

## Validation against real orbits

The script checks itself against known orbits before opening the interactive menu:

| Orbit | Quantity | Calculated | Reference |
|---|---|---|---|
| ISS (408 km) | Velocity | 7.664 km/s | ~7.66 km/s |
| ISS (408 km) | Period | 1 h 32 m 43 s | ~92.7 min |
| GEO (35,786 km) | Period | 23 h 56 m 3 s | one sidereal day |
| GTO (200 × 35,786 km) | Eccentricity | 0.730 | ~0.73 |
| GTO | Perigee / apogee velocity | 10.239 / 1.597 km/s | ~10.2 / ~1.6 km/s |

These checks also run automatically in CI (`tests/test_foundations.py`).

## Run

```bash
python orbital_calculator.py
```

Standard library only.

## Concepts

Dictionaries of body constants, tuples for multi-value returns, `try/except` input handling.
