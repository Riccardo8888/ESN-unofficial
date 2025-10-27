#!/usr/bin/env python3
"""
Quick Test - WebSocket ESN Server
==================================
Test rapido per verificare che il server funzioni.
"""

import asyncio
import websockets
import json
import numpy as np
import time

async def quick_test():
    uri = "ws://127.0.0.1:8765"
    print(f"🔗 Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as ws:
            print("✅ Connected!")
            
            # Wait for ready
            ready_msg = await ws.recv()
            ready = json.loads(ready_msg)
            
            if ready.get('type') == 'error':
                print(f"❌ Server busy: {ready.get('message')}")
                return
            
            print(f"✅ Server ready (session: {ready['sessionId']})")
            print(f"   Model: {ready['modelInfo']['reservoirSize']} neurons")
            
            # Send 10 test frames
            print("\n📤 Sending 10 test frames...")
            latencies = []
            
            for i in range(10):
                frame = {
                    'timestamp': time.time(),
                    'frameIndex': i,
                    'grid': np.random.rand(6144).tolist(),
                    'metadata': {
                        'heading': 1.57,
                        'headingSin': np.sin(1.57),
                        'headingCos': np.cos(1.57),
                        'velocity': 5.78,
                        'boost': False,
                        'distanceToBorder': 15000.0,
                    }
                }
                
                start = time.time()
                await ws.send(json.dumps(frame))
                response = await ws.recv()
                latency = (time.time() - start) * 1000
                latencies.append(latency)
                
                resp = json.loads(response)
                if i == 0:
                    print(f"\n📊 First response:")
                    print(f"   angleDelta: {np.degrees(resp['command']['angleDelta']):.1f}°")
                    print(f"   boost: {resp['command']['boost']}")
                    print(f"   confidence: {resp['confidence']:.3f}")
                    print(f"   latency: {latency:.2f}ms")
                
                await asyncio.sleep(0.1)
            
            print(f"\n✅ Test completed!")
            print(f"   Avg latency: {np.mean(latencies):.2f}ms")
            print(f"   Min/Max: {np.min(latencies):.2f}/{np.max(latencies):.2f}ms")
            
    except ConnectionRefusedError:
        print("❌ Connection refused! Is the server running?")
        print("   Run: python3 websocket_reservoir.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    asyncio.run(quick_test())
