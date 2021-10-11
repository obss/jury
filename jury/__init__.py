import os

from jury.core import Jury
from jury.metrics import load_metric

SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(SOURCE_DIR)

__version__ = "2.0.0"
