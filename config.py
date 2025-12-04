import os
from pathlib import Path

# Paths
BASE_DIR = Path(os.getcwd())
OUTPUT_DIR = BASE_DIR / "output"

# Model Settings
DEFAULT_MODEL = "yolov8n.pt"  # Change to your specific weights path
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# Tracking Settings
GAP_SECONDS = 5.0      # For the GapCounter logic
MIN_HISTORY = 5        # Minimum frames the object must be detected to be tracked
TRAIL_LENGTH = 30      # Length of the trail behind objects (for trajectory drawing)

# Visual Settings
FONT_SCALE = 0.6
THICKNESS = 2
TEXT_COLOR = (255, 255, 255)

# StrongSort specific
STRONGSORT_WEIGHTS = Path('mobilenetv2_x1_4.pt')