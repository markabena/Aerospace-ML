"""Shared helpers: load a portfolio script as a module by file path."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
