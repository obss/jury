import os
from pathlib import Path

from jury.core import Jury

SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(SOURCE_DIR)
TEST_DIR = os.path.join(PROJECT_ROOT, "tests")
TEST_DATA_DIR = os.path.join(TEST_DIR, "test_data")

HOME_DIR = Path.home()
CACHE_DIR = HOME_DIR / ".cache" / "jury"
os.makedirs(CACHE_DIR, exist_ok=True)

__version__ = "1.1.2"
