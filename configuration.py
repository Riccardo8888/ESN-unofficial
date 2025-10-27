"""
Configuration file for Slither.io Echo State Network Training
==============================================================

This file contains all configuration parameters for training an ESN
to predict player actions (direction and sprint) from game state.
"""

from pathlib import Path

# ===========================================
# DATA PATHS
# ===========================================
# Path to slither.io scraper data directory (LOCAL in this workspace)
SLITHER_DATA_PATH = Path(__file__).parent / "data"

# Path to save trained models and results
OUTPUT_PATH = Path("/Users/nick/Desktop/SSN-Folder/ESN-unofficial/slither_esn_results")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ===========================================
# DATA PARAMETERS
# ===========================================
# Input dimensions from slither.io data
ANGULAR_BINS = 64      # Angular resolution of polar grid
RADIAL_BINS = 24       # Radial resolution of polar grid
CHANNELS = 4           # Food, Enemy Body, My Body, Enemy Heads
INPUT_DIM = ANGULAR_BINS * RADIAL_BINS * CHANNELS  # 6144 inputs per frame

# Output dimensions (player actions)
OUTPUT_DIM = 3         # mx (direction x), my (direction y), boost (0/1)

# ===========================================
# PREDICTION PARAMETERS
# ===========================================
# Number of frames to predict into the future
PREDICTION_HORIZON = 1  # At frame t, predict player actions at frame t+5

# Minimum frames per session to be considered valid
MIN_FRAMES_PER_SESSION = 100

# ===========================================
# TRAIN/TEST SPLIT
# ===========================================
# Fraction of data to use for testing
TEST_SPLIT = 0.2       # 20% test, 80% train

# Random seed for reproducibility
RANDOM_SEED = 42

# ===========================================
# ECHO STATE NETWORK PARAMETERS
# ===========================================
# Reservoir size
N_RESERVOIR = 2000     # Number of neurons in the reservoir

# Spectral radius (controls dynamics)
SPECTRAL_RADIUS = 1.1

# Input scaling
INPUT_SCALE = 1.0

# Leak rate range (for leaky integrator neurons)
LEAK_RATE_MIN = 0.05
LEAK_RATE_MAX = 0.2

# Sparsity of reservoir connectivity (None = dense, or value between 0-1)
SPARSITY = 0.85         # 90% of connections are zero

# ===========================================
# TRAINING PARAMETERS
# ===========================================
# Ridge regression regularization parameter
ALPHA = 0.005

# Washout period (frames to discard at start of each sequence)
WASHOUT = 50

# ===========================================
# EVALUATION PARAMETERS
# ===========================================
# Threshold for classifying boost as active (0.5 = standard binary threshold)
BOOST_THRESHOLD = 0.5

# Print training progress every N frames
PROGRESS_INTERVAL = 1000

# ===========================================
# ADDITIONAL METADATA
# ===========================================
# Include additional features beyond grid data
USE_VELOCITY = True    # Include snake velocity as input
USE_HEADING = True     # Include snake heading as input
USE_DISTANCE_TO_BORDER = True  # Include distance to border as input
USE_ANGULAR_VELOCITY = True  # Include rate of heading change (d_heading/dt)
USE_DISTANCE_VELOCITY = True  # Include rate of distance change (d_distance/dt)

# Calculate actual input dimension with metadata
if USE_VELOCITY:
    INPUT_DIM += 1
if USE_HEADING:
    INPUT_DIM += 2  # sin and cos of heading
if USE_DISTANCE_TO_BORDER:
    INPUT_DIM += 1
if USE_ANGULAR_VELOCITY:
    INPUT_DIM += 1  # Rate of heading change
if USE_DISTANCE_VELOCITY:
    INPUT_DIM += 1  # Rate of border distance change

# ===========================================
# DISPLAY CONFIGURATION
# ===========================================
def print_config():
    """Print the current configuration"""
    print("=" * 60)
    print("SLITHER.IO ESN TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Data Path: {SLITHER_DATA_PATH}")
    print(f"Output Path: {OUTPUT_PATH}")
    print(f"\nInput Dimensions:")
    print(f"  - Grid: {ANGULAR_BINS}x{RADIAL_BINS}x{CHANNELS} = {ANGULAR_BINS * RADIAL_BINS * CHANNELS}")
    if USE_VELOCITY:
        print(f"  - Velocity: 1")
    if USE_HEADING:
        print(f"  - Heading (sin, cos): 2")
    if USE_DISTANCE_TO_BORDER:
        print(f"  - Distance to border: 1")
    print(f"  - Total: {INPUT_DIM}")
    print(f"\nOutput Dimensions: {OUTPUT_DIM} (mx, my, boost)")
    print(f"\nPrediction Horizon: {PREDICTION_HORIZON} frames")
    print(f"\nESN Configuration:")
    print(f"  - Reservoir size: {N_RESERVOIR}")
    print(f"  - Spectral radius: {SPECTRAL_RADIUS}")
    print(f"  - Input scale: {INPUT_SCALE}")
    print(f"  - Leak rate: [{LEAK_RATE_MIN}, {LEAK_RATE_MAX}]")
    print(f"  - Sparsity: {SPARSITY}")
    print(f"\nTraining Configuration:")
    print(f"  - Alpha (regularization): {ALPHA}")
    print(f"  - Washout: {WASHOUT} frames")
    print(f"  - Test split: {TEST_SPLIT * 100:.0f}%")
    print(f"  - Random seed: {RANDOM_SEED}")
    print(f"  - Min frames per session: {MIN_FRAMES_PER_SESSION}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
