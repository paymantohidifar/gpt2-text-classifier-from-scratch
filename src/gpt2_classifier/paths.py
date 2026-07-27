"""Central filesystem path conventions for the gpt2_classifier package.

Every module that reads or writes data/model artifacts should import the
constants defined here rather than hardcoding cwd-relative paths, so behavior
does not depend on the working directory a script happens to be launched from.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
PLOTS_DIR: Path = PROJECT_ROOT / "plots"


DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
