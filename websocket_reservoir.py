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
from datetime import datetime
import argparse
import sys

sys.path.append(str(Path(__file__).parent))

from configuration import *
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
            'distanceToBorder': None
        }
        
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
            current_heading = metadata['heading']
            if self.previous_frame['heading'] is not None:
                # Calcola cambio angolare con wrapping
                angular_vel = current_heading - self.previous_frame['heading']
                angular_vel = np.arctan2(np.sin(angular_vel), np.cos(angular_vel))
                # Normalize
                angular_vel = angular_vel / np.pi
            else:
                angular_vel = 0.0  # First frame
            features_list.append([angular_vel])
        
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
        
        # Update previous frame values for next iteration
        self.previous_frame['heading'] = metadata['heading']
        self.previous_frame['distanceToBorder'] = metadata['distanceToBorder']
        
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
        
        # Compute output
        prediction = self.reservoir_state @ self.W_out  # [3]
        
        mx_pred = float(prediction[0])
        my_pred = float(prediction[1])
        boost_prob = float(prediction[2])
        
        return mx_pred, my_pred, boost_prob
    
    def compute_control_command(self, frame_data: dict, mx_pred: float, my_pred: float, boost_prob: float) -> dict:
        """
        Calcola il comando di controllo dal prediction ESN.
        
        Args:
            frame_data: Frame originale (per current heading)
            mx_pred: Predicted mouse X (normalized direction)
            my_pred: Predicted mouse Y (normalized direction)
            boost_prob: Boost probability [0, 1]
            
        Returns:
            Command dictionary
        """
        # Current heading
        current_heading = frame_data['metadata']['heading']
        
        # Predicted angle (from mx, my vector)
        predicted_angle = np.arctan2(my_pred, mx_pred)
        
        # Calculate angleDelta
        angle_delta = predicted_angle - current_heading
        
        # Normalize to [-π, π]
        angle_delta = np.arctan2(np.sin(angle_delta), np.cos(angle_delta))
        
        # CLAMP angle delta to prevent extreme turns (max ±45° per frame)
        MAX_ANGLE_DELTA = np.radians(45)  # ±45° max
        if abs(angle_delta) > MAX_ANGLE_DELTA:
            angle_delta = np.sign(angle_delta) * MAX_ANGLE_DELTA
            # Log when clamping happens
            if self.stats['frames_processed'] % 10 == 0:
                print(f"⚠️  Clamped large angleDelta from {np.degrees(predicted_angle - current_heading):.1f}° to {np.degrees(angle_delta):.1f}°")
        
        # Boost decision (LOWERED threshold to 0.45 due to model bias)
        boost_decision = (boost_prob > 0.45)
        
        # Confidence (distance from threshold)
        confidence = boost_prob if boost_decision else (1.0 - boost_prob)
        
        return {
            'angleDelta': float(angle_delta),
            'boost': bool(boost_decision),
            'confidence': float(confidence),
            'predictedAngle': float(predicted_angle)
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
        
        # ESN inference
        mx_pred, my_pred, boost_prob = self.predict(features)
        
        # Compute control command
        command = self.compute_control_command(frame_data, mx_pred, my_pred, boost_prob)
        
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
        
        print(f"📤 Predicted: {predicted_angle_deg:+7.2f}° | "
              f"Delta: {angle_delta_deg:+7.2f}° | "
              f"BOOST {boost_str} | "
              f"Conf: {command['confidence']:.2f} | "
              f"mx: {mx_pred:+.3f}, my: {my_pred:+.3f}")
        
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
