"""
CENTRALIZED CONFIGURATION — Cross-Layer IDS
All configurable values in one place, overridable via environment variables.
"""
import os

# Serial connection
IDS_PORT = os.environ.get('IDS_PORT', None)  # None = auto-detect
IDS_BAUD = int(os.environ.get('IDS_BAUD', '115200'))

# Paths
IDS_RESULTS_DIR = os.environ.get('IDS_RESULTS_DIR', 'demo_results')
IDS_MODEL_PATH = os.environ.get('IDS_MODEL_PATH', 'models/full_29_model.pkl')
IDS_LOG_FILE = os.environ.get('IDS_LOG_FILE', 'ids_demo.log')

# Physics constants (must match generate_dataset.py and firmware)
WHEELBASE = 2.7
STEERING_RATIO = 16.0
DT = 0.1
