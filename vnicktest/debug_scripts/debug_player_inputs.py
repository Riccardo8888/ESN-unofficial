"""
Debug script to understand what player_inputs contains in training data
"""
import zarr
import numpy as np
from pathlib import Path

# Find first session
data_path = Path('../../data')
session_found = False

for user_dir in data_path.iterdir():
    if user_dir.is_dir():
        for session_dir in user_dir.iterdir():
            if session_dir.name.startswith('session_'):
                print(f'Analyzing: {session_dir}\n')
                root = zarr.open(str(session_dir), mode='r')
                
                # Get first 20 frames
                player_inputs = np.array(root['player_inputs'][:20])
                headings = np.array(root['headings'][:20])
                
                print(f'First 20 frames:')
                print(f'{"Frame":>5} | {"Heading(deg)":>12} | {"mx":>8} | {"my":>8} | {"boost":>5} | {"angle(mx,my)":>14}')
                print('-' * 85)
                
                for i in range(20):
                    h_deg = np.degrees(headings[i])
                    mx, my, boost = player_inputs[i]
                    angle_from_mx_my = np.degrees(np.arctan2(my, mx))
                    print(f'{i:5d} | {h_deg:12.2f} | {mx:8.3f} | {my:8.3f} | {boost:5.0f} | {angle_from_mx_my:14.2f}')
                
                # Statistics
                print(f'\n📊 Statistics (first 1000 frames):')
                player_inputs_1k = np.array(root['player_inputs'][:1000])
                mx_all = player_inputs_1k[:, 0]
                my_all = player_inputs_1k[:, 1]
                
                print(f'mx: min={mx_all.min():.3f}, max={mx_all.max():.3f}, mean={mx_all.mean():.3f}, std={mx_all.std():.3f}')
                print(f'my: min={my_all.min():.3f}, max={my_all.max():.3f}, mean={my_all.mean():.3f}, std={my_all.std():.3f}')
                
                # Check if values are normalized (-1, 1) or something else
                print(f'\n🔍 Value range analysis:')
                if abs(mx_all).max() <= 1.1 and abs(my_all).max() <= 1.1:
                    print('  ✓ Values appear to be normalized in [-1, 1]')
                    print('  → (mx, my) likely represents direction vector')
                else:
                    print('  ✗ Values exceed [-1, 1] range')
                    print('  → (mx, my) might be absolute pixel coordinates')
                
                session_found = True
                break
    if session_found:
        break

if not session_found:
    print('❌ No session data found!')
