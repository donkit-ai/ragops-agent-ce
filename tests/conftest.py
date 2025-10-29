import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path for tests without installation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

os.environ.setdefault("RAGOPS_API_URL", "http://localhost:8080")
