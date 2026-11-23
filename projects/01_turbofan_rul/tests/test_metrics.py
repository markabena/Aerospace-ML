"""
Checks for src/metrics.py. Pure maths, so no data files needed.

Run:  python tests/test_metrics.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics as m  # noqa: E402


def check(name, condition):
    print(f"  {'ok ' if condition else 'FAIL'}  {name}")
    return condition


def main():
    results = []

    results.append(check("perfect prediction scores 0 on everything",
                         m.rmse([50, 80], [50, 80]) == 0
                         and m.nasa_score([50, 80], [50, 80]) == 0))

    results.append(check("RMSE of a constant 10-cycle error is 10",
                         math.isclose(m.rmse([10, 20, 30], [20, 30, 40]), 10.0)))

    # Hand-computed values for d = +10 and d = -13
    results.append(check("late by 10 cycles costs e^1 - 1",
                         math.isclose(m.nasa_score([50], [60]), math.e - 1)))
    results.append(check("early by 13 cycles costs e^1 - 1",
                         math.isclose(m.nasa_score([50], [37]), math.e - 1)))

    late = m.nasa_score([50], [70])
    early = m.nasa_score([50], [30])
    results.append(check("late is penalised harder than early of the same size",
                         late > early))

    results.append(check("score is a sum over engines",
                         math.isclose(m.nasa_score([50, 50], [60, 60]),
                                      2 * (math.e - 1))))

    results.append(check("late_fraction counts over-predictions only",
                         m.late_fraction([10, 10, 10, 10], [5, 10, 15, 20]) == 0.5))

    try:
        m.rmse([1, 2], [1, 2, 3])
        results.append(check("shape mismatch raises", False))
    except ValueError:
        results.append(check("shape mismatch raises", True))

    if all(results):
        print(f"\nAll {len(results)} metric checks PASSED")
        return 0
    print(f"\n{results.count(False)} of {len(results)} metric checks FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
