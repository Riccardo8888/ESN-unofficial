"""Configuration constants for the slither.io reservoir-computing pipeline.

Extracted (Phase 3) from the canonical
`humand_data_1015_slither_copy_use_brain_connectome_reservoir_executed.ipynb`.
"""
ANGLE_MIN = -40
ANGLE_MAX = 40
ANGLE_RESOLUTION = 5
NUM_ANGLE_BINS = int((ANGLE_MAX - ANGLE_MIN) / ANGLE_RESOLUTION) + 1  # 17
OUTPUT_DIM = NUM_ANGLE_BINS + 1  # + boost channel = 18

WINDOW_LEN = 25
WINDOW_STRIDE = 15
MIN_FRAMES_PER_SESSION = 100
TEST_RATIO = 0.25

ALPHA = 1e-2        # ridge regularization for the readout
T_WASHOUT = 5       # leading timesteps discarded per window
LEAK_RANGE = (0.1, 0.3)
INPUT_SCALE = 0.6
