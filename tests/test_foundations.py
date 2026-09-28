"""
Checks for the foundations track (P1-P5).

The physics scripts are checked against published reference values, and
the data scripts are run end to end, so a change that breaks a result
fails CI rather than going unnoticed.
"""
import math

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import pytest
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from conftest import load_script

p1 = load_script("foundations/p1_rocket_performance/rocket_eval.py", "p1")
p2 = load_script("foundations/p2_orbital_parameters/orbital_calculator.py", "p2")


# -- P1: rocket performance ---------------------------------------------------

def test_tsiolkovsky_matches_closed_form():
    # A mass ratio of e gives delta-v = Isp * g0 exactly.
    dv = p1.tsiolkovsky(isp=300, wet_mass=math.e * 1000, dry_mass=1000)
    assert dv == pytest.approx(300 * p1.G0)


def test_thrust_to_weight():
    assert p1.thrust_to_weight(thrust=2 * 1000 * p1.G0, wet_mass=1000) == pytest.approx(2.0)


# -- P2: orbital parameters vs real orbits ------------------------------------

def test_iss_orbit():
    assert p2.circular_orbital_velocity(408) == pytest.approx(7.66, abs=0.01)
    assert p2.orbital_period(408) / 60 == pytest.approx(92.7, abs=0.2)


def test_geostationary_period_is_one_sidereal_day():
    assert p2.orbital_period(35786) == pytest.approx(86164, rel=1e-3)


def test_gto_transfer_orbit():
    assert p2.eccentricity(200, 35786) == pytest.approx(0.73, abs=0.005)
    v_p, v_a = p2.apoapsis_periapsis_velocities(200, 35786)
    assert v_p == pytest.approx(10.24, abs=0.05)
    assert v_a == pytest.approx(1.60, abs=0.05)


# -- P3 -> P4: ascent telemetry pipeline --------------------------------------

def test_ascent_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    p3 = load_script("foundations/p3_flight_data_analyser/flight_data_analyser.py", "p3")
    p4 = load_script("foundations/p4_flight_visualiser/flight_visualiser.py", "p4")

    # Redirect every output into a temp dir so the test never touches the repo.
    for name in ("CLEANED_OUTPUT_FILE", "SUMMARY_OUTPUT_FILE"):
        monkeypatch.setattr(p3, name, tmp_path / getattr(p3, name).name)
    p3.main()

    monkeypatch.setattr(p4, "TELEMETRY_FILE", p3.CLEANED_OUTPUT_FILE)
    monkeypatch.setattr(p4, "SUMMARY_FILE", p3.SUMMARY_OUTPUT_FILE)
    monkeypatch.setattr(p4, "OUTPUT_IMAGE", tmp_path / "profile.png")
    p4.main()

    summary = pd.read_csv(p3.SUMMARY_OUTPUT_FILE).iloc[0]
    assert summary["max_q_time_s"] == 54
    assert summary["throttle_down_start_s"] < summary["max_q_time_s"] < summary["throttle_down_end_s"]
    assert (tmp_path / "profile.png").stat().st_size > 0


# -- P5: engine health classifier ---------------------------------------------

def test_engine_health_model_catches_degraded_engines():
    p5 = load_script("foundations/p5_engine_health_monitor/engine_health_monitor.py", "p5")
    X, y = p5.split_features_and_labels(p5.load_data(p5.DATA_FILE))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = p5.build_model().fit(X_train, y_train)
    recall = recall_score(y_test, model.predict(X_test), pos_label="degraded")
    assert recall >= 0.85, f"Degraded recall fell to {recall:.2f}"
