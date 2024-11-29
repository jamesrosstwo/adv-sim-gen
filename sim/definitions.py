import os
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.resolve().absolute()
DATA_PATH = ROOT_PATH / "data"
OUT_PATH = ROOT_PATH / "out"