"""Checks that every project follows the repository standard."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = sorted(p for p in (ROOT / "projects").iterdir() if p.is_dir())


def test_projects_exist():
    assert len(PROJECTS) >= 1


def test_each_project_has_standard_layout():
    for project in PROJECTS:
        assert (project / "README.md").is_file(), f"{project.name} missing README.md"
        for sub in ("notebooks", "src", "results"):
            assert (project / sub).is_dir(), f"{project.name} missing {sub}/"
