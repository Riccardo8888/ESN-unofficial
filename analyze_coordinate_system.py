"""
CRITICAL ANALYSIS: Are (mx, my) relative to rotating canvas or absolute?

If relative → angle(mx,my) should be INDEPENDENT of heading
If absolute → angle(mx,my) - heading should show correlation
"""
import zarr
import numpy as np
from pathlib import Path

data_path = Path('data')

print("🔍 ANALISI SISTEMA DI COORDINATE\n")
print("Cerco correlazione tra heading e (mx, my)...\n")

for user_dir in data_path.iterdir():
    if user_dir.is_dir():
        for session_dir in user_dir.iterdir():
            if session_dir.name.startswith('session_'):
                print(f'📂 Analyzing: {session_dir.name}\n')
                root = zarr.open(str(session_dir), mode='r')
                
                # Get data
                player_inputs = np.array(root['player_inputs'][:500])
                headings = np.array(root['headings'][:500])
                
                mx = player_inputs[:, 0]
                my = player_inputs[:, 1]
                
                # Calculate angles
                angle_from_mx_my = np.arctan2(my, mx)
                
                # TEST 1: Se (mx,my) sono RELATIVI, angle(mx,my) dovrebbe essere INDIPENDENTE da heading
                # Calcoliamo la correlazione
                correlation = np.corrcoef(headings, angle_from_mx_my)[0, 1]
                
                print("TEST 1: Correlazione tra heading e angle(mx,my)")
                print(f"  Correlazione: {correlation:.4f}")
                if abs(correlation) < 0.3:
                    print("  ✅ RELATIVI: angle(mx,my) è indipendente da heading")
                else:
                    print("  ❌ ASSOLUTI: angle(mx,my) dipende da heading!")
                
                # TEST 2: Se (mx,my) sono ASSOLUTI, allora angle(mx,my) - heading dovrebbe essere ~costante
                # (sarebbe la direzione ASSOLUTA del mouse rispetto al mondo)
                angle_diff = angle_from_mx_my - headings
                # Normalize to [-pi, pi]
                angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
                
                print(f"\nTEST 2: Varianza di [angle(mx,my) - heading]")
                print(f"  Varianza: {np.var(angle_diff):.4f} rad²")
                print(f"  Std Dev:  {np.std(angle_diff):.4f} rad ({np.degrees(np.std(angle_diff)):.1f}°)")
                
                if np.std(angle_diff) < 0.5:  # <30° std
                    print("  ❌ ASSOLUTI: L'angolo relativo al mondo è quasi costante!")
                    print("  → Il giocatore punta sempre nella stessa direzione assoluta")
                else:
                    print("  ✅ RELATIVI: L'angolo relativo cambia molto")
                
                # TEST 3: Sample frames per vedere visualmente
                print(f"\n📋 Sample frames (primi 10):")
                print(f"{'Frame':>5} | {'Heading':>10} | {'angle(mx,my)':>13} | {'Differenza':>12}")
                print('-' * 50)
                for i in range(min(10, len(headings))):
                    h_deg = np.degrees(headings[i])
                    angle_deg = np.degrees(angle_from_mx_my[i])
                    diff_deg = np.degrees(angle_diff[i])
                    print(f'{i:5d} | {h_deg:10.2f}° | {angle_deg:13.2f}° | {diff_deg:12.2f}°')
                
                # TEST 4: Visualize distribution
                print(f"\n📊 Distribuzione angle(mx,my) - heading:")
                hist, bins = np.histogram(np.degrees(angle_diff), bins=10, range=(-180, 180))
                max_bar = max(hist)
                for i in range(len(hist)):
                    bar_length = int(40 * hist[i] / max_bar) if max_bar > 0 else 0
                    print(f"  {bins[i]:6.1f}° to {bins[i+1]:6.1f}°: {'█' * bar_length} {hist[i]}")
                
                print(f"\n" + "="*70)
                print("CONCLUSIONE:")
                if abs(correlation) > 0.5:
                    print("❌ (mx, my) sono COORDINATE ASSOLUTE (non ruotano col canvas)")
                    print("   → L'extension NON sta catturando coordinate relative!")
                    print("   → Il modello NON può imparare il comportamento corretto!")
                    exit(0)
                else:
                    print("✅ (mx, my) sono COORDINATE RELATIVE (ruotano col canvas)")
                    print("   → L'extension funziona correttamente")
                    print("   → Il bias nei dati è REALE (giocatori preferiscono sinistra)")
                    exit(0)

print("❌ No session data found!")