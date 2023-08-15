import os
from pathlib import Path

def get_root(path):
    return os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent, path)