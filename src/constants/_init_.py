from pathlib import Path
import os

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"
PARAMS_FILE_PATH = PROJECT_ROOT / "params.yaml"
