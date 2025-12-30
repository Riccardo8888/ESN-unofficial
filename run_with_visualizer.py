#!/usr/bin/env python3
"""
Run ESN WebSocket Server with Real-time Visualization
======================================================

Launches both the websocket server and the PyGame visualizer in parallel.
"""

import sys
import threading
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from websocket_reservoir import ESNControlServer
from visualizer import ProbabilityVisualizer


def run_server(server):
    """Run websocket server in a separate thread."""
    import asyncio
    asyncio.run(server.start())


def main():
    """Launch server and visualizer together."""
    print("\n" + "=" * 60)
    print("ESN SERVER + VISUALIZER LAUNCHER")
    print("=" * 60)
    
    # Find latest model
    model_path = Path(__file__).parent / "slither_esn_results"
    if not model_path.exists():
        print(f"\n❌ No models found in {model_path}")
        print("   Please train a model first: python3 train_slither_reservoir.py")
        return 1
    
    # Get most recent model
    model_dirs = sorted([d for d in model_path.iterdir() if d.is_dir()], 
                       key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not model_dirs:
        print(f"\n❌ No model directories found in {model_path}")
        return 1
    
    # Try different model filenames
    latest_model = None
    for filename in ["reservoir_model.npz", "slither_esn_model.npz"]:
        test_path = model_dirs[0] / filename
        if test_path.exists():
            latest_model = test_path
            break
    
    if latest_model is None:
        print(f"\n❌ Model file not found in {model_dirs[0]}")
        print(f"   Looked for: reservoir_model.npz, slither_esn_model.npz")
        return 1
    
    print(f"\n✓ Using model: {latest_model.parent.name}")
    
    # Create server
    print("\n📡 Starting WebSocket server...")
    server = ESNControlServer(model_path=latest_model, host='localhost', port=8765)
    
    # Create visualizer
    print("🎨 Starting visualizer...")
    visualizer = ProbabilityVisualizer()
    visualizer.connect_to_server(server)
    
    # Run server in background thread
    server_thread = threading.Thread(target=run_server, args=(server,), daemon=True)
    server_thread.start()
    
    print("\n" + "=" * 60)
    print("✓ Server running on ws://localhost:8765")
    print("✓ Visualizer window opened")
    print("\nWaiting for game connection...")
    print("Close visualizer window or press Ctrl+C to stop")
    print("=" * 60)
    
    # Run visualizer in main thread (needs main thread for GUI)
    try:
        visualizer.run(update_rate=30)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    print("\n✓ Shutting down...")
    return 0


if __name__ == '__main__':
    sys.exit(main())
