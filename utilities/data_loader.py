"""
Data Loader for Slither.io Echo State Network
==============================================

This module handles loading and preprocessing Slither.io game data
from Zarr format for ESN training.
"""

import numpy as np
import zarr
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configuration import *


def discover_sessions(data_path: Path) -> List[Path]:
    """
    Discover all available session/game directories in the data path.
    
    Supports two structures:
    1. OLD: data/user/session_* (mixed games in one session)
    2. NEW: data/AI_bot/game_* (one game per directory)
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        List of paths to valid session/game directories
    """
    sessions = []
    
    # Look for user directories
    if data_path.exists():
        for user_dir in data_path.iterdir():
            if user_dir.is_dir():
                # Look for session directories within user dir (OLD structure)
                for session_dir in user_dir.iterdir():
                    if session_dir.is_dir():
                        # Check both session_* (old) and game_* (new) patterns
                        if (session_dir.name.startswith('session_') or 
                            session_dir.name.startswith('game_')):
                            # Check if it contains Zarr data
                            if (session_dir / 'grids').exists():
                                sessions.append(session_dir)
    
    return sorted(sessions)


def load_session(session_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Load a single session from Zarr storage.
    
    Args:
        session_path: Path to session directory
        
    Returns:
        Dictionary containing arrays: grids, player_inputs, timestamps,
        headings, velocities, distances_to_border, boost_states
        Returns None if session is invalid or too short
    """
    try:
        # Load Zarr group (compatible with both Zarr 2.x and 3.x)
        try:
            # Try Zarr 3.x API first
            from zarr import open_group
            root = open_group(str(session_path), mode='r')
        except (ImportError, AttributeError):
            # Fallback to Zarr 2.x API
            store = zarr.DirectoryStore(str(session_path))
            root = zarr.group(store=store)
        
        # Load arrays
        grids = np.array(root['grids'])  # [frames, 64, 24, 4]
        player_inputs = np.array(root['player_inputs'])  # [frames, 3]
        timestamps = np.array(root['timestamps'])  # [frames]
        headings = np.array(root['headings'])  # [frames]
        velocities = np.array(root['velocities'])  # [frames]
        distances_to_border = np.array(root['distances_to_border'])  # [frames]
        boost_states = np.array(root['boost_states'])  # [frames]
        
        # Check if session has minimum number of frames
        n_frames = grids.shape[0]
        if n_frames < MIN_FRAMES_PER_SESSION:
            print(f"  Skipping {session_path.name}: only {n_frames} frames (min: {MIN_FRAMES_PER_SESSION})")
            return None
        
        # Load metadata if available
        metadata_file = session_path / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        return {
            'grids': grids,
            'player_inputs': player_inputs,
            'timestamps': timestamps,
            'headings': headings,
            'velocities': velocities,
            'distances_to_border': distances_to_border,
            'boost_states': boost_states,
            'metadata': metadata,
            'session_path': session_path
        }
        
    except Exception as e:
        print(f"  Error loading {session_path.name}: {e}")
        return None


def normalize_grids(grids: np.ndarray) -> np.ndarray:
    """
    Normalize grid data to [0, 1] range.
    
    Args:
        grids: Array of shape [frames, angular, radial, channels]
        
    Returns:
        Normalized grids
    """
    # Grids should already be normalized from the collection process
    # but we clip to ensure [0, 1] range
    return np.clip(grids, 0.0, 1.0)


def prepare_features(session_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Prepare input features for ESN by flattening grids and adding metadata.
    
    Args:
        session_data: Dictionary containing session arrays
        
    Returns:
        Feature array of shape [frames, INPUT_DIM]
    """
    n_frames = session_data['grids'].shape[0]
    
    # Flatten grids: [frames, 64, 24, 4] -> [frames, 6144]
    grids_flat = session_data['grids'].reshape(n_frames, -1)
    grids_flat = normalize_grids(session_data['grids']).reshape(n_frames, -1)
    
    features = [grids_flat]
    
    # Add velocity if enabled
    if USE_VELOCITY:
        velocity = session_data['velocities'].reshape(-1, 1)
        # Normalize velocity (typical max is around 100-200)
        velocity = velocity / 200.0
        features.append(velocity)
    
    # Add heading (sin and cos) if enabled
    if USE_HEADING:
        heading = session_data['headings'].reshape(-1, 1)
        heading_sin = np.sin(heading)
        heading_cos = np.cos(heading)
        features.extend([heading_sin, heading_cos])
    
    # Add angular velocity if enabled (rate of heading change)
    if USE_ANGULAR_VELOCITY:
        heading = session_data['headings']
        # Calcola differenza con wrapping circolare
        angular_vel = np.diff(heading, prepend=heading[0])
        # Wrap to [-π, π]
        angular_vel = np.arctan2(np.sin(angular_vel), np.cos(angular_vel))
        # Normalize to [-1, 1] (typical max angular velocity is ~π rad/frame)
        angular_vel = angular_vel / np.pi
        features.append(angular_vel.reshape(-1, 1))
    
    # Add distance to border if enabled
    if USE_DISTANCE_TO_BORDER:
        distance = session_data['distances_to_border'].reshape(-1, 1)
        # Normalize distance (typical game radius is 21600)
        distance = distance / 21600.0
        features.append(distance)
    
    # Add distance velocity if enabled (rate of distance change)
    if USE_DISTANCE_VELOCITY:
        distance = session_data['distances_to_border']
        # Calcola velocità di avvicinamento/allontanamento dal bordo
        distance_vel = np.diff(distance, prepend=distance[0])
        # Normalize (typical max change is ~200 units/frame)
        distance_vel = distance_vel / 200.0
        # Clip to reasonable range
        distance_vel = np.clip(distance_vel, -1.0, 1.0)
        features.append(distance_vel.reshape(-1, 1))
    
    # Add previous angle delta if enabled (for temporal coherence)
    if USE_PREVIOUS_ANGLE:
        # Calculate angle from player_inputs (mx, my)
        mx = session_data['player_inputs'][:, 0]
        my = session_data['player_inputs'][:, 1]
        angle_delta = np.arctan2(my, mx)
        
        # Shift by 1 frame to get "previous" angle
        prev_angle_delta = np.roll(angle_delta, shift=1)
        prev_angle_delta[0] = 0.0  # First frame has no previous angle
        
        # Encode as sin/cos to avoid discontinuity at -π/π
        prev_angle_sin = np.sin(prev_angle_delta)
        prev_angle_cos = np.cos(prev_angle_delta)
        features.extend([prev_angle_sin.reshape(-1, 1), prev_angle_cos.reshape(-1, 1)])
    
    # Concatenate all features
    X = np.concatenate(features, axis=1)
    
    return X


def convert_to_angle_bins(player_inputs: np.ndarray) -> np.ndarray:
    """
    Convert (mx, my, boost) to (angle_bin_one_hot, boost) format.
    
    Args:
        player_inputs: Array [frames, 3] with (mx, my, boost)
        
    Returns:
        Array [frames, NUM_ANGLE_BINS + 1] with one-hot angle + boost
    """
    mx = player_inputs[:, 0]
    my = player_inputs[:, 1]
    boost = player_inputs[:, 2]
    
    # Calculate angle from (mx, my)
    angles_rad = np.arctan2(my, mx)
    angles_deg = np.degrees(angles_rad)
    
    # Clip to [-90, +90] range
    angles_deg = np.clip(angles_deg, ANGLE_MIN, ANGLE_MAX)
    
    # Convert to continuous bin position (not rounded)
    # Example: 2.5° → bin_pos = 18.5 (between bin 18 and 19)
    bin_positions = (angles_deg - ANGLE_MIN) / ANGLE_RESOLUTION
    bin_positions = np.clip(bin_positions, 0, NUM_ANGLE_BINS - 1)
    
    # SOFT LABEL ENCODING: Split probability between adjacent bins
    # Instead of hard one-hot, distribute weight between neighbors
    n_frames = len(player_inputs)
    y_angle_soft = np.zeros((n_frames, NUM_ANGLE_BINS), dtype=np.float32)
    
    for i in range(n_frames):
        pos = bin_positions[i]
        
        # Get lower and upper bin indices
        lower_bin = int(np.floor(pos))
        upper_bin = int(np.ceil(pos))
        
        # Ensure bounds
        lower_bin = np.clip(lower_bin, 0, NUM_ANGLE_BINS - 1)
        upper_bin = np.clip(upper_bin, 0, NUM_ANGLE_BINS - 1)
        
        if lower_bin == upper_bin:
            # Exact bin center (e.g., 0.0°, 5.0°, 10.0°)
            y_angle_soft[i, lower_bin] = 1.0
        else:
            # Between two bins - linear interpolation
            # Example: pos=18.3 → 70% to bin 18, 30% to bin 19
            weight_upper = pos - lower_bin  # Distance from lower bin
            weight_lower = 1.0 - weight_upper
            
            y_angle_soft[i, lower_bin] = weight_lower
            y_angle_soft[i, upper_bin] = weight_upper
    
    # Concatenate with boost
    y_new = np.concatenate([y_angle_soft, boost.reshape(-1, 1)], axis=1)
    
    return y_new


def create_sequences_with_horizon(X: np.ndarray, y: np.ndarray, 
                                  horizon: int = PREDICTION_HORIZON) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-output sequences where output is shifted by horizon frames.
    
    At frame t, we have input X[t] and we want to predict output y[t+horizon].
    
    Args:
        X: Input features [frames, input_dim]
        y: Output targets [frames, output_dim]
        horizon: Number of frames to predict ahead
        
    Returns:
        X_seq, y_seq where X_seq[i] corresponds to y_seq[i+horizon]
    """
    n_frames = X.shape[0]
    
    # We can only use frames where we have future output available
    valid_frames = n_frames - horizon
    
    if valid_frames <= 0:
        return np.array([]), np.array([])
    
    # Input is frames 0 to n-horizon-1
    X_seq = X[:valid_frames]
    
    # Output is frames horizon to n-1
    y_seq = y[horizon:horizon+valid_frames]
    
    return X_seq, y_seq


def load_all_data(data_path: Path, 
                  verbose: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[str]]:
    """
    Load and prepare all available sessions.
    
    Args:
        data_path: Path to data directory
        verbose: Print progress information
        
    Returns:
        Tuple of (X_list, y_list, session_names, usernames) where:
        - X_list: List of input arrays for each session
        - y_list: List of output arrays for each session
        - session_names: List of session names for tracking
        - usernames: List of usernames for each session
    """
    # Discover sessions
    session_paths = discover_sessions(data_path)
    
    if verbose:
        print(f"\nFound {len(session_paths)} session(s) in {data_path}")
    
    X_list = []
    y_list = []
    session_names = []
    usernames = []
    
    for i, session_path in enumerate(session_paths):
        # Determine if this is a session or game
        is_game = session_path.name.startswith('game_')
        item_type = "game" if is_game else "session"
        
        if verbose:
            print(f"\nLoading {item_type} {i+1}/{len(session_paths)}: {session_path.name}")
        
        # Load session/game
        session_data = load_session(session_path)
        if session_data is None:
            continue
        
        # Prepare features
        X = prepare_features(session_data)
        y_raw = session_data['player_inputs']  # [frames, 3] (mx, my, boost)
        
        # Convert to angle bins (one-hot encoding)
        y = convert_to_angle_bins(y_raw)  # [frames, NUM_ANGLE_BINS + 1]
        
        # Create sequences with prediction horizon
        X_seq, y_seq = create_sequences_with_horizon(X, y, horizon=PREDICTION_HORIZON)
        
        if X_seq.shape[0] == 0:
            if verbose:
                print(f"  Skipping: not enough frames for horizon={PREDICTION_HORIZON}")
            continue
        
        username = session_data['metadata'].get('username', 'unknown')
        
        # For AI_bot games, extract username from parent directory if not in metadata
        if username == 'unknown' and is_game:
            username = session_path.parent.name  # AI_bot
        
        X_list.append(X_seq)
        y_list.append(y_seq)
        session_names.append(session_path.name)
        usernames.append(username)
        
        if verbose:
            print(f"  ✓ Loaded: {X_seq.shape[0]} frames, user: {username}")
    
    if verbose:
        print(f"\n✓ Successfully loaded {len(X_list)} session(s)/game(s)")
        total_frames = sum(X.shape[0] for X in X_list)
        print(f"  Total frames: {total_frames}")
    
    return X_list, y_list, session_names, usernames


def train_test_split_chunks(X_list: List[np.ndarray], 
                           y_list: List[np.ndarray],
                           session_names: List[str],
                           usernames: List[str] = None,
                           test_size: float = TEST_SPLIT,
                           n_chunks: int = 3,
                           random_seed: int = RANDOM_SEED) -> Tuple:
    """
    Split data into training and testing sets by taking random chunks from each session.
    This creates a more homogeneous test set compared to taking entire sessions.
    
    Args:
        X_list: List of input arrays (one per session)
        y_list: List of output arrays (one per session)
        session_names: List of session names
        usernames: List of usernames for each session (optional)
        test_size: Fraction of data for testing (per session)
        n_chunks: Number of random chunks to extract from each session for test
        random_seed: Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test, test_user_indices (all concatenated)
        test_user_indices maps each test sample to its session index for user analysis
    """
    np.random.seed(random_seed)
    
    X_train_all = []
    y_train_all = []
    X_test_all = []
    y_test_all = []
    test_user_indices = []  # Track which session each test sample belongs to
    
    for i, (X_session, y_session) in enumerate(zip(X_list, y_list)):
        n_frames = len(X_session)
        chunk_size = max(1, int(n_frames * test_size / n_chunks))
        
        # Generate random start indices for test chunks
        max_start = n_frames - chunk_size
        if max_start <= 0:
            # Session too short, put all in train
            X_train_all.append(X_session)
            y_train_all.append(y_session)
            continue
        
        # Sample n_chunks random positions
        test_starts = np.random.choice(max_start, size=min(n_chunks, max_start), replace=False)
        
        # Create mask for test indices
        test_mask = np.zeros(n_frames, dtype=bool)
        for start in test_starts:
            end = min(start + chunk_size, n_frames)
            test_mask[start:end] = True
        
        # Split into train/test
        X_test_session = X_session[test_mask]
        y_test_session = y_session[test_mask]
        
        X_test_all.append(X_test_session)
        y_test_all.append(y_test_session)
        
        # Track session index for each test sample
        test_user_indices.extend([i] * len(X_test_session))
        
        X_train_all.append(X_session[~test_mask])
        y_train_all.append(y_session[~test_mask])
    
    # Concatenate all
    X_train = np.concatenate(X_train_all, axis=0) if X_train_all else np.array([])
    y_train = np.concatenate(y_train_all, axis=0) if y_train_all else np.array([])
    X_test = np.concatenate(X_test_all, axis=0) if X_test_all else np.array([])
    y_test = np.concatenate(y_test_all, axis=0) if y_test_all else np.array([])
    test_user_indices = np.array(test_user_indices)
    
    return X_train, y_train, X_test, y_test, test_user_indices


def train_test_split(X_list: List[np.ndarray], 
                     y_list: List[np.ndarray],
                     session_names: List[str],
                     usernames: List[str] = None,
                     test_size: float = TEST_SPLIT,
                     random_seed: int = RANDOM_SEED,
                     use_chunks: bool = True) -> Tuple:
    """
    Split data into training and testing sets.
    
    Args:
        X_list: List of input arrays
        y_list: List of output arrays
        session_names: List of session names
        usernames: List of usernames for each session (optional)
        test_size: Fraction of data for testing
        random_seed: Random seed for reproducibility
        use_chunks: If True, take random chunks from each session (recommended)
                   If False, take entire sessions for test (old behavior)
        
    Returns:
        If use_chunks=True: X_train, y_train, X_test, y_test, test_user_indices (concatenated arrays)
        If use_chunks=False: X_train, y_train, X_test, y_test, train_names, test_names (lists)
    """
    if use_chunks:
        # New method: random chunks from each session
        return train_test_split_chunks(X_list, y_list, session_names, usernames,
                                      test_size, n_chunks=3, random_seed=random_seed)
    else:
        # Old method: entire sessions
        np.random.seed(random_seed)
        
        n_sessions = len(X_list)
        n_test = max(1, int(n_sessions * test_size))
        
        # Random indices for test set
        indices = np.arange(n_sessions)
        np.random.shuffle(indices)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        # Split data
        X_train = [X_list[i] for i in train_indices]
        y_train = [y_list[i] for i in train_indices]
        train_names = [session_names[i] for i in train_indices]
        
        X_test = [X_list[i] for i in test_indices]
        y_test = [y_list[i] for i in test_indices]
        test_names = [session_names[i] for i in test_indices]
        
        return X_train, y_train, X_test, y_test, train_names, test_names


def concatenate_sessions(X_list: List[np.ndarray], 
                         y_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate multiple sessions into single arrays.
    
    Args:
        X_list: List of input arrays
        y_list: List of output arrays
        
    Returns:
        X, y concatenated arrays
    """
    if len(X_list) == 0:
        return np.array([]), np.array([])
    
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    return X, y


if __name__ == "__main__":
    """Test data loading"""
    print_config()
    
    print("\n" + "=" * 60)
    print("TESTING DATA LOADER")
    print("=" * 60)
    
    # Load all data
    X_list, y_list, session_names = load_all_data(SLITHER_DATA_PATH, verbose=True)
    
    if len(X_list) > 0:
        # Split into train/test using random chunks (new method)
        X_train, y_train, X_test, y_test = train_test_split(
            X_list, y_list, session_names, use_chunks=True
        )
        
        print(f"\n{'=' * 60}")
        print("TRAIN/TEST SPLIT (usando chunk casuali da ogni sessione)")
        print("=" * 60)
        print(f"Total sessions: {len(X_list)}")
        print(f"Training frames: {X_train.shape[0]}")
        print(f"Test frames: {X_test.shape[0]}")
        print(f"Test ratio: {X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]):.1%}")
        print(f"Input dimension: {X_train.shape[1]}")
        print(f"Output dimension: {y_train.shape[1]}")
        
        print(f"\nOutput statistics (training set):")
        print(f"  mx: mean={y_train[:, 0].mean():.3f}, std={y_train[:, 0].std():.3f}")
        print(f"  my: mean={y_train[:, 1].mean():.3f}, std={y_train[:, 1].std():.3f}")
        print(f"  boost: mean={y_train[:, 2].mean():.3f} (fraction active)")
    else:
        print("\n⚠ No data found! Check SLITHER_DATA_PATH in configuration.py")
