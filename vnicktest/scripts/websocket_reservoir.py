#!/usr/bin/env python3
"""
WebSocket Server for Slither.io ESN Real-Time Control
======================================================

Server WebSocket che riceve frame dal client e ritorna comandi di controllo
usando un modello ESN precedentemente addestrato.

Protocollo:
- Client invia frame di gioco (grid + metadata) a 10 Hz
- Server risponde con comandi (angleDelta + boost)
- Una connessione alla volta per garantire performance

"""

import asyncio
import websockets
import json
import numpy as np
import time
from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).parent))

from vnicktest.scripts.configuration import *
from reservoir import Reservoir


class ESNControlServer:
    """
    Server WebSocket per controllo ESN real-time.
    """
    
    def __init__(self, model_path: Path, host='0.0.0.0', port=8765):
        """
        Inizializza il server.
        
        Args:
            model_path: Path al file .npz del modello addestrato
            host: Host su cui ascoltare (default: 0.0.0.0 per tutte le interfacce)
            port: Porta WebSocket (default: 8765)
        """
        self.host = host
        self.port = port
        self.model_path = model_path
        self.active_connection = None
        
        # History for temporal derivatives (previous frame values)
        self.previous_frame = {
            'heading': None,
            'distanceToBorder': None,
            'angle_delta': None
        }
        
        # Temporal smoothing: track previous angle distribution
        self.previous_angle_probs = None  # For 50/50 smoothing with current frame
        
        # Latest predictions (for visualization)
        self.latest_angle_probs = None
        self.latest_boost_prob = None
        self.latest_command = None
        
        # Load model
        print("\n" + "=" * 60)
        print("LOADING ESN MODEL")
        print("=" * 60)
        self.load_model()
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'total_inference_time': 0.0,
            'start_time': None,
            'session_id': None
        }
        
    def load_model(self):
        """
        Carica il modello ESN da file .npz.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        
        # Load model data
        model_data = np.load(self.model_path)
        
        # Extract parameters
        W_in = model_data['W_in']
        W = model_data['W']
        self.W_out = model_data['W_out']
        leak = model_data['leak']
        n_inputs = int(model_data['n_inputs'])
        n_neurons = int(model_data['n_neurons'])
        spectral_radius = float(model_data['spectral_radius'])
        
        # Reconstruct reservoir (with dummy initialization - will be overwritten)
        self.reservoir = Reservoir(
            n_inputs=n_inputs,
            n_neurons=n_neurons,
            rhow=spectral_radius,  # Use rhow parameter
            inp_scaling=1.0,
            leak_range=(leak.min(), leak.max())
        )
        
        # Set weights manually (overwrite initialization)
        self.reservoir.win = W_in
        self.reservoir.w = W
        self.reservoir.leak = leak
        
        print(f"✓ Model loaded successfully!")
        print(f"  - Input dim: {n_inputs}")
        print(f"  - Reservoir size: {n_neurons}")
        print(f"  - Spectral radius: {spectral_radius:.2f}")
        print(f"  - Output dim: {self.W_out.shape[1]}")
        print(f"  - Leak rate: [{leak.min():.2f}, {leak.max():.2f}]")
        
        # Initialize reservoir state
        self.reservoir_state = np.zeros(n_neurons)
        
    def prepare_features(self, frame_data: dict) -> np.ndarray:
        """
        Prepara le features dal frame ricevuto (stesso ordine del training).
        
        Args:
            frame_data: Dictionary con dati del frame
            
        Returns:
            Feature vector matching the trained model's input dimension
        """
        # Extract grid (flatten)
        grid_flat = np.array(frame_data['grid'], dtype=np.float32).flatten()  # (6144,)
        
        # Extract metadata
        metadata = frame_data['metadata']
        
        # Build feature vector (MATCH TRAINING ORDER!)
        # Based on configuration.py settings used during training
        features_list = [grid_flat]  # Start with grid
        
        # Add velocity if model expects it
        if USE_VELOCITY:
            features_list.append([metadata['velocity'] / 200.0])  # normalized
        
        # Add heading (sin and cos) if model expects it
        if USE_HEADING:
            features_list.append([metadata['headingSin']])
            features_list.append([metadata['headingCos']])
        
        # Add angular velocity if model expects it
        if USE_ANGULAR_VELOCITY:
            # Ricostruisci heading da sin/cos
            current_heading = np.arctan2(metadata['headingSin'], metadata['headingCos'])
            
            if self.previous_frame['heading'] is not None:
                # Calcola cambio angolare con wrapping
                angular_vel = current_heading - self.previous_frame['heading']
                angular_vel = np.arctan2(np.sin(angular_vel), np.cos(angular_vel))
                # Normalize
                angular_vel = angular_vel / np.pi
            else:
                angular_vel = 0.0  # First frame
            features_list.append([angular_vel])
            
            # Update previous heading
            self.previous_frame['heading'] = current_heading
        
        # Add distance to border if model expects it
        if USE_DISTANCE_TO_BORDER:
            features_list.append([metadata['distanceToBorder'] / 21600.0])  # normalized
        
        # Add distance velocity if model expects it
        if USE_DISTANCE_VELOCITY:
            current_distance = metadata['distanceToBorder']
            if self.previous_frame['distanceToBorder'] is not None:
                # Calcola variazione distanza dal bordo
                distance_vel = current_distance - self.previous_frame['distanceToBorder']
                # Normalize and clip
                distance_vel = np.clip(distance_vel / 200.0, -1.0, 1.0)
            else:
                distance_vel = 0.0  # First frame
            features_list.append([distance_vel])
            
            # Update previous distance
            self.previous_frame['distanceToBorder'] = current_distance
        
        # Add previous angle delta if model expects it (for temporal coherence)
        if USE_PREVIOUS_ANGLE:
            if self.previous_frame['angle_delta'] is not None:
                # Use previous angle delta (sin, cos)
                prev_angle_sin = np.sin(self.previous_frame['angle_delta'])
                prev_angle_cos = np.cos(self.previous_frame['angle_delta'])
            else:
                # First frame: no previous angle
                prev_angle_sin = 0.0
                prev_angle_cos = 1.0  # cos(0) = 1
            
            features_list.append([prev_angle_sin])
            features_list.append([prev_angle_cos])
            
            # Note: angle_delta will be updated AFTER prediction in compute_control_command
        
        # NOTE: snake_length is NOT included in the current model
        # This matches the data_loader.py logic which doesn't add snake_length
        
        features = np.concatenate(features_list)
        
        # Verify dimension matches loaded model
        if features.shape[0] != self.reservoir.n_inputs:
            raise ValueError(
                f"Feature dimension mismatch! "
                f"Prepared {features.shape[0]} features but model expects {self.reservoir.n_inputs}"
            )
        
        return features
    
    def predict(self, features: np.ndarray) -> tuple:
        """
        Esegue inferenza ESN su un singolo frame.
        
        Args:
            features: Feature vector [6150]
            
        Returns:
            (mx_pred, my_pred, boost_prob)
        """
        # Update reservoir state using the update method
        self.reservoir_state = self.reservoir.update(features, self.reservoir_state)
        
        # Compute output - ANGLE CLASSIFICATION format
        # Output: [NUM_ANGLE_BINS] angle logits + [1] boost logit
        prediction = self.reservoir_state @ self.W_out  # [NUM_ANGLE_BINS + 1]
        
        angle_logits = prediction[:NUM_ANGLE_BINS]  # First 29 values
        boost_logit = float(prediction[NUM_ANGLE_BINS])  # Last value (raw logit)
        
        # Apply softmax to get angle probabilities
        angle_probs = self._softmax(angle_logits)
        
        # Apply sigmoid to get boost probability [0, 1]
        boost_prob = 1.0 / (1.0 + np.exp(-boost_logit))
        
        # SHARPEN probabilities with power function to emphasize higher values
        # This makes the distribution more peaked around the most likely angles
        # Higher exponent = more peaked distribution (more deterministic)
        # Configured in configuration.py: SHARPENING_EXPONENT
        
        angle_probs_sharpened = angle_probs ** SHARPENING_EXPONENT
        # Re-normalize after sharpening
        angle_probs_sharpened = angle_probs_sharpened / np.sum(angle_probs_sharpened)
        
        # TEMPORAL SMOOTHING: DISABLED (was causing issues)
        # Direct use of sharpened probabilities without averaging with previous frame
        
        return angle_probs_sharpened, boost_prob
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities from logits."""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)
    
    def compute_control_command(self, frame_data: dict, angle_probs: np.ndarray, boost_prob: float) -> dict:
        """
        Calcola il comando di controllo dal prediction ESN usando SAMPLING.
        
        Args:
            frame_data: Frame originale (per current heading)
            angle_probs: Probability distribution over angle bins [NUM_ANGLE_BINS]
            boost_prob: Boost probability [0, 1]
            
        Returns:
            Command dictionary
        """
        # Current heading (solo per logging, non serve per il calcolo!)
        current_heading = frame_data['metadata']['heading']
        
        # SAMPLE angle from probability distribution (NO smoothing)
        # Sample bin index according to probabilities
        sampled_bin = np.random.choice(NUM_ANGLE_BINS, p=angle_probs)
        
        # Convert bin to angle in degrees
        angle_deg = sampled_bin * ANGLE_RESOLUTION + ANGLE_MIN  # e.g., bin 18 → 0°
        
        # Convert to radians (use bin center directly, no Gaussian noise)
        angle_delta = np.radians(angle_deg)
        
        # Calculate confidence from distribution entropy
        # High entropy (uniform) = low confidence
        # Low entropy (peaked) = high confidence
        entropy = -np.sum(angle_probs * np.log(angle_probs + 1e-10))
        max_entropy = np.log(NUM_ANGLE_BINS)  # Maximum possible entropy
        confidence = 1.0 - (entropy / max_entropy)  # Normalize to [0, 1]
        
        # Confidence scale for logging (no scaling applied now since we sample)
        confidence_scale = 1.0
        
        # Boost decision - DETERMINISTIC with threshold
        # Since model was trained with raw outputs (not sigmoid during training),
        # we need a higher threshold after applying sigmoid in inference
        # Threshold tuned empirically: 0.6 works well (was 0.20 on raw logits)
        boost_decision = (boost_prob > 0.50)
        
        # Boost confidence (distance from 0.5 threshold)
        boost_confidence = boost_prob  # Maps [0,1] to [0,1] with peak at extremes
        
        # Calculate predicted angle for logging (angle_delta + current_heading)
        predicted_angle_abs = current_heading + angle_delta
        predicted_angle_abs = np.arctan2(np.sin(predicted_angle_abs), np.cos(predicted_angle_abs))
        
        # Update previous angle delta for next frame (temporal coherence)
        self.previous_frame['angle_delta'] = angle_delta
        
        return {
            'angleDelta': float(angle_delta),
            'boost': bool(boost_decision),
            'confidence': float(boost_confidence),
            'predictedAngle': float(predicted_angle_abs),  # For logging only
            'directionConfidence': float(confidence),  # Magnitude-based confidence
            'confidenceScale': float(confidence_scale)  # Scaling factor applied
        }
    
    def process_frame(self, message: str) -> str:
        """
        Processa un frame e ritorna il comando di controllo.
        
        Args:
            message: JSON string del frame
            
        Returns:
            JSON string della risposta
        """
        start_time = time.time()
        
        # Parse frame
        frame_data = json.loads(message)
        
        # Log incoming frame info
        current_heading_rad = frame_data['metadata']['heading']
        current_heading_deg = np.degrees(current_heading_rad)
        frame_idx = frame_data.get('frameIndex', '?')
        print(f"\n📥 Frame {frame_idx} | Current heading: {current_heading_deg:+7.2f}° ({current_heading_rad:.3f} rad)")
        
        # Prepare features
        features = self.prepare_features(frame_data)
        
        # ESN inference - returns angle probabilities and boost prob
        angle_probs, boost_prob = self.predict(features)
        
        # Save for visualization
        self.latest_angle_probs = angle_probs
        self.latest_boost_prob = boost_prob
        
        # Compute control command using SAMPLING from distribution
        command = self.compute_control_command(frame_data, angle_probs, boost_prob)
        self.latest_command = command
        
        # Processing time
        processing_time = time.time() - start_time
        
        # Build response
        response = {
            'type': 'control',
            'timestamp': time.time(),
            'frameIndex': frame_data['frameIndex'],
            'command': {
                'angleDelta': command['angleDelta'],
                'boost': command['boost']
            },
            'confidence': command['confidence'],
            'meta': {
                'processingTime': processing_time,
                'predictedAngle': command['predictedAngle']
            }
        }
        
        # Update stats
        self.stats['frames_processed'] += 1
        self.stats['total_inference_time'] += processing_time
        
        # Log real-time command (every frame) with context
        angle_delta_deg = np.degrees(command['angleDelta'])
        predicted_angle_deg = np.degrees(command['predictedAngle'])
        boost_str = "true " if command['boost'] else "false"
        dir_conf = command['directionConfidence']
        conf_scale = command['confidenceScale']
        
        # Color coding based on direction confidence
        conf_emoji = "🟢" if dir_conf > 0.8 else "🟡" if dir_conf > 0.5 else "🟠" if dir_conf > 0.3 else "🔴"
        
        # Boost probability display (show actual probability used for random decision)
        boost_emoji = "⚡" if command['boost'] else "  "
        
        # Find top 3 angle bins for display
        top3_bins = np.argsort(angle_probs)[-3:][::-1]
        top3_angles = [bin_idx * ANGLE_RESOLUTION + ANGLE_MIN for bin_idx in top3_bins]
        top3_probs = [angle_probs[bin_idx] for bin_idx in top3_bins]
        top3_str = ", ".join([f"{angle:+.0f}°({prob:.2f})" for angle, prob in zip(top3_angles, top3_probs)])
        
        print(f"📤 Sampled: {angle_delta_deg:+7.2f}° | "
              f"Top3: [{top3_str}] | "
              f"{boost_emoji}BOOST {boost_str} (p={boost_prob:.2f}) | "
              f"{conf_emoji} Conf: {dir_conf:.3f}")
        
        # Log statistics every 50 frames
        if self.stats['frames_processed'] % 50 == 0:
            avg_time = self.stats['total_inference_time'] / self.stats['frames_processed']
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"\n📊 Processed {self.stats['frames_processed']} frames | "
                  f"Avg inference: {avg_time*1000:.2f}ms | "
                  f"Max FPS: {fps:.1f}\n")
        
        return json.dumps(response)
    
    async def handle_client(self, websocket, path):
        """
        Gestisce una connessione client.
        """
        # Politica: una connessione alla volta
        if self.active_connection is not None:
            print(f"⚠️  Client connection rejected (server busy): {websocket.remote_address}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Server busy - another client is already connected'
            }))
            await websocket.close()
            return
        
        self.active_connection = websocket
        print(f"\n✅ Client connected: {websocket.remote_address}")
        
        # Reset stats
        self.stats['frames_processed'] = 0
        self.stats['total_inference_time'] = 0.0
        self.stats['start_time'] = time.time()
        self.stats['session_id'] = str(int(time.time() * 1000))
        
        # Reset temporal smoothing for new session
        self.previous_angle_probs = None
        
        # Send ready message
        ready_msg = {
            'type': 'ready',
            'sessionId': self.stats['session_id'],
            'modelInfo': {
                'reservoirSize': self.reservoir.n_neurons,
                'inputDim': self.reservoir.n_inputs,
                'outputDim': self.W_out.shape[1],
                'spectralRadius': float(self.reservoir.spectral_radius),
                'expectedPerformance': {
                    'angularError': 36,  # From training results
                    'boostAccuracy': 0.81,
                    'directionRMSE': 0.48
                }
            }
        }
        
        await websocket.send(json.dumps(ready_msg))
        print(f"📤 Sent ready message (session: {self.stats['session_id']})")
        
        try:
            # Main loop: receive frames and send commands
            async for message in websocket:
                try:
                    response = self.process_frame(message)
                    await websocket.send(response)
                except Exception as e:
                    print(f"❌ Error processing frame: {e}")
                    error_response = {
                        'type': 'error',
                        'message': str(e)
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"\n🔌 Client disconnected: {websocket.remote_address}")
        finally:
            self.active_connection = None
            
            # Print final stats
            if self.stats['frames_processed'] > 0:
                duration = time.time() - self.stats['start_time']
                avg_time = self.stats['total_inference_time'] / self.stats['frames_processed']
                print(f"\n📊 Session Statistics:")
                print(f"  - Duration: {duration:.1f}s")
                print(f"  - Frames processed: {self.stats['frames_processed']}")
                print(f"  - Avg inference time: {avg_time*1000:.2f}ms")
                print(f"  - Avg FPS: {self.stats['frames_processed']/duration:.1f}")
    
    async def start(self):
        """
        Avvia il server WebSocket.
        """
        print("\n" + "=" * 60)
        print("STARTING WEBSOCKET SERVER")
        print("=" * 60)
        print(f"Host: {self.host}")
        print(f"Port: {self.port}")
        print(f"URL: ws://{self.host}:{self.port}")
        print("=" * 60)
        print("\n🚀 Server ready! Waiting for client connection...")
        print("   (Press Ctrl+C to stop)\n")
        
        # Start server (compatible with both websockets 10.x and 11.x)
        try:
            # Try websockets 11.x+ (no path parameter)
            async with websockets.serve(
                lambda ws: self.handle_client(ws, "/"),
                self.host,
                self.port
            ):
                await asyncio.Future()  # Run forever
        except TypeError:
            # Fallback for older websockets versions
            async with websockets.serve(self.handle_client, self.host, self.port):
                await asyncio.Future()  # Run forever


def find_latest_model(output_path: Path = OUTPUT_PATH) -> Path:
    """
    Trova l'ultimo modello addestrato nella cartella output.
    
    Args:
        output_path: Path alla cartella output
        
    Returns:
        Path al file reservoir_model.npz più recente
    """
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_path}")
    
    # Find all training directories
    training_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('training_')]
    
    if not training_dirs:
        raise FileNotFoundError(f"No training directories found in: {output_path}")
    
    # Sort by timestamp (most recent first)
    training_dirs.sort(reverse=True)
    
    # Find first directory with model file
    for training_dir in training_dirs:
        model_path = training_dir / 'reservoir_model.npz'
        if model_path.exists():
            return model_path
    
    raise FileNotFoundError("No reservoir_model.npz found in any training directory")


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description='WebSocket Server for Slither.io ESN Real-Time Control'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model file (.npz). If not specified, uses latest trained model.'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to listen on (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8765,
        help='Port to listen on (default: 8765)'
    )
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model:
        model_path = Path(args.model)
    else:
        print("No model specified, searching for latest trained model...")
        model_path = find_latest_model()
        print(f"✓ Found latest model: {model_path}")
    
    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        print("\nPlease train a model first using:")
        print("  python3 train_slither_reservoir.py")
        return 1
    
    # Create and start server
    try:
        server = ESNControlServer(
            model_path=model_path,
            host=args.host,
            port=args.port
        )
        
        # Run server
        asyncio.run(server.start())
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
