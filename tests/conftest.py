"""
Makes the repository root importable no matter where pytest is invoked from, so
``from api import app`` / ``from model.lm import TinyGPT`` resolve consistently.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
