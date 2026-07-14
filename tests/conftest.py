import os
import sys

import matplotlib

# Use a non-interactive backend so plotting tests never require a display.
matplotlib.use("Agg")

# Make the project modules importable regardless of the pytest invocation dir.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
