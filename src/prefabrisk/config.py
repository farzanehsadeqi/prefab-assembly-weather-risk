"""Load model parameters from config/params.yaml."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else PROJECT_ROOT / "config" / "params.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)