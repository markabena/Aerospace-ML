"""Runs the turbofan RUL project's own test scripts against mock data."""
import subprocess
import sys
from pathlib import Path

import pytest

PROJECTS = Path(__file__).resolve().parents[1] / "projects"
TURBOFAN = PROJECTS / "01_turbofan_rul"


@pytest.mark.parametrize("script", ["test_data_loader.py", "test_features.py"])
def test_turbofan_suite(script):
    subprocess.run([sys.executable, "tests/make_mock_data.py"], cwd=TURBOFAN,
                   check=True, capture_output=True)
    result = subprocess.run([sys.executable, f"tests/{script}"], cwd=TURBOFAN,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "PASSED" in result.stdout


def test_every_project_has_a_readme():
    for project in PROJECTS.iterdir():
        if project.is_dir():
            assert (project / "README.md").is_file(), f"{project.name} has no README"
