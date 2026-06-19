# src/data_io.py

from blimpy import Waterfall
from pathlib import Path


def load_fil(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    wf = Waterfall(str(path))

    return wf
