import os
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.resolve().absolute()
DATA_PATH = ROOT_PATH / "data"
OUT_PATH = ROOT_PATH / "out"
RESOURCES_PATH = ROOT_PATH / "resources"
SEED = 42

os.path.join(str(ROOT_PATH), str(ROOT_PATH / "sim"))