#!/usr/bin/env python3
"""
Test Client per WebSocket ESN Server
=====================================

Client di test che invia frame mock al server WebSocket e verifica le risposte.
"""

import asyncio
import websockets
import json
import numpy as np
import time
from pathlib import Path

async def test_connection():
    """
    Test della connessione e dello scambio di messaggi.
    """
    uri = "ws://127.0.0.1:8765"
    
    print("=" * 60)
    print("TESTING ESN WEBSOCKET SERVER")
    print("=" * 60)
    print(f"Connecting to: {uri}")
    
    async with websockets.connect(uri) as websocket:
        print("✅ Connected!")
        
        # Wait for ready message
        ready_msg = await websocket.recv()
        ready_data = json.loads(ready_msg)
        print(f"\n📥 Received ready message:")
        print(f"   Session ID: {ready_data['sessionId']}")
        print(f"   Model info: {ready_data['modelInfo']}")
        
        # Create mock frame
        print("\n" + "=" * 60)
        print("SENDING TEST FRAMES")
        print("=" * 60)
        
        n_frames = 100
        latencies = []
        
        for i in range(n_frames):
            # Create mock frame data
            frame = {
                'timestamp': time.time(),
                'sessionId': ready_data['sessionId'],
                'frameIndex': i,
                'deltaTime': 0.1,
                'username': 'test_user',
                
                # Mock grid (random noise)
                'grid': np.random.rand(6144).tolist(),
                
                'gridMeta': {
                    'angularBins': 64,
                    'radialBins': 24,
                    'channels': 4,
                    'rMin': 60,
                    'rMax': 3200,
                    'alphaWarp': 6.0
                },
                
                'metadata': {
                    'heading': 1.57 + (i * 0.01),  # Slowly rotate
                    'headingSin': np.sin(1.57 + i * 0.01),
                    'headingCos': np.cos(1.57 + i * 0.01),
                    'velocity': 5.78,
                    'boost': False,
                    'distanceToBorder': 15000.0,
                    'gameRadius': 21600,
                    'snakeLength': 100
                },
                
                'playerInput': {
                    'mx': 0.23,
                    'my': -0.45,
                    'boost': 0
                }
            }
            
            # Send frame
            send_time = time.time()
            await websocket.send(json.dumps(frame))
            
            # Receive response
            response = await websocket.recv()
            recv_time = time.time()
            
            latency = (recv_time - send_time) * 1000  # ms
            latencies.append(latency)
            
            response_data = json.loads(response)
            
            # Print first, middle, and last frames
            if i == 0 or i == n_frames // 2 or i == n_frames - 1:
                print(f"\n📊 Frame {i}:")
                print(f"   Latency: {latency:.2f}ms")
                print(f"   Command: angleDelta={np.degrees(response_data['command']['angleDelta']):.1f}°, "
                      f"boost={response_data['command']['boost']}")
                print(f"   Confidence: {response_data['confidence']:.3f}")
                print(f"   Processing time: {response_data['meta']['processingTime']*1000:.2f}ms")
            
            # Small delay to simulate 10 Hz
            await asyncio.sleep(0.1)
        
        # Statistics
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        print(f"Frames sent: {n_frames}")
        print(f"Avg latency: {np.mean(latencies):.2f}ms")
        print(f"Min latency: {np.min(latencies):.2f}ms")
        print(f"Max latency: {np.max(latencies):.2f}ms")
        print(f"Std latency: {np.std(latencies):.2f}ms")
        
        # Check if latency is acceptable
        avg_latency = np.mean(latencies)
        if avg_latency < 20:
            print(f"\n✅ PASS: Latency < 20ms (target met!)")
        elif avg_latency < 50:
            print(f"\n⚠️  WARNING: Latency between 20-50ms (acceptable but not ideal)")
        else:
            print(f"\n❌ FAIL: Latency > 50ms (too slow!)")
        
        print("\n✅ Test completed successfully!")


async def test_multiple_connections():
    """
    Test che il server rifiuti connessioni multiple.
    """
    uri = "ws://127.0.0.1:8765"
    
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE CONNECTIONS (should reject)")
    print("=" * 60)
    
    async with websockets.connect(uri) as ws1:
        print("✅ First connection established")
        
        # Try second connection (should be rejected)
        try:
            async with websockets.connect(uri) as ws2:
                msg = await ws2.recv()
                data = json.loads(msg)
                if data['type'] == 'error':
                    print(f"✅ Second connection rejected as expected: {data['message']}")
                else:
                    print(f"❌ Second connection accepted (should have been rejected!)")
        except websockets.exceptions.ConnectionClosed:
            print("✅ Second connection closed immediately (expected)")


def main():
    """
    Main entry point.
    """
    print("\n🚀 Starting WebSocket ESN Server Test\n")
    print("⚠️  Make sure the server is running:")
    print("   python3 websocket_reservoir.py\n")
    
    try:
        # Test basic connection and frames
        asyncio.run(test_connection())
        
        # Test multiple connections
        asyncio.run(test_multiple_connections())
        
    except ConnectionRefusedError:
        print("\n❌ Connection refused!")
        print("   Make sure the server is running:")
        print("   python3 websocket_reservoir.py")
        return 1
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
