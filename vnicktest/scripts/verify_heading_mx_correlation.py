"""
Verifica se mx è correlato con heading (direzione assoluta nel mondo)
Se c'è correlazione forte → il modello può imparare strategie spaziali assolute
"""
import zarr
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo
import matplotlib.pyplot as plt

data_path = Path('data')

print("🔍 ANALISI CORRELAZIONE HEADING ↔ MX\n")
print("Se il modello riceve heading come input, può imparare:")
print("  'Quando vado verso Nord (heading=90°), gira a sinistra (mx<0)'")
print("  Questo crea un bias spaziale assoluto!\n")

all_headings = []
all_mx = []
all_my = []

# Collect data from all sessions
for user_dir in data_path.iterdir():
    if user_dir.is_dir():
        for session_dir in user_dir.iterdir():
            if session_dir.name.startswith('session_'):
                root = zarr.open(str(session_dir), mode='r')
                
                headings = np.array(root['headings'][:])
                player_inputs = np.array(root['player_inputs'][:])
                
                all_headings.extend(headings)
                all_mx.extend(player_inputs[:, 0])
                all_my.extend(player_inputs[:, 1])

all_headings = np.array(all_headings)
all_mx = np.array(all_mx)
all_my = np.array(all_my)

print(f"📊 Dati totali: {len(all_headings)} frames\n")

# Converti heading in gradi per leggibilità
headings_deg = np.degrees(all_headings)

# TEST 1: Correlazione lineare
corr_mx = np.corrcoef(all_headings, all_mx)[0, 1]
corr_my = np.corrcoef(all_headings, all_my)[0, 1]

print("TEST 1: Correlazione lineare heading ↔ (mx, my)")
print(f"  Corr(heading, mx): {corr_mx:+.4f}")
print(f"  Corr(heading, my): {corr_my:+.4f}")

if abs(corr_mx) > 0.1 or abs(corr_my) > 0.1:
    print("  ⚠️  CORRELAZIONE RILEVATA!")
    print("  → Il modello PUÒ imparare strategie spaziali assolute!")
else:
    print("  ✓ Correlazione debole (< 0.1)")

# TEST 2: Analisi per bins di heading
print(f"\n📈 Analisi mx medio per direzione assoluta:")
print(f"{'Direzione':>15} | {'Heading Range':>15} | {'mx medio':>10} | {'my medio':>10} | {'N frames':>10}")
print("-" * 70)

directions = [
    ("EAST →", 0, 45),
    ("NE ↗", 45, 90),
    ("NORTH ↑", 90, 135),
    ("NW ↖", 135, 180),
    ("WEST ←", -180, -135),
    ("SW ↙", -135, -90),
    ("SOUTH ↓", -90, -45),
    ("SE ↘", -45, 0),
]

for direction_name, min_deg, max_deg in directions:
    mask = (headings_deg >= min_deg) & (headings_deg < max_deg)
    if mask.sum() > 0:
        mx_mean = all_mx[mask].mean()
        my_mean = all_my[mask].mean()
        n_frames = mask.sum()
        print(f"{direction_name:>15} | {min_deg:>6}° to {max_deg:>4}° | {mx_mean:>+9.3f} | {my_mean:>+9.3f} | {n_frames:>10}")

# TEST 3: Scatter plot heading vs mx
print("\n📊 Generando plot heading vs mx...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subsample per plot leggibile
n_samples = min(5000, len(all_headings))
indices = np.random.choice(len(all_headings), n_samples, replace=False)

axes[0].scatter(headings_deg[indices], all_mx[indices], alpha=0.3, s=1)
axes[0].set_xlabel('Heading (degrees)')
axes[0].set_ylabel('mx (left/right)')
axes[0].set_title(f'Heading vs mx (corr={corr_mx:+.3f})')
axes[0].axhline(0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
axes[0].grid(True, alpha=0.3)

axes[1].scatter(headings_deg[indices], all_my[indices], alpha=0.3, s=1)
axes[1].set_xlabel('Heading (degrees)')
axes[1].set_ylabel('my (up/down)')
axes[1].set_title(f'Heading vs my (corr={corr_my:+.3f})')
axes[1].axhline(0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heading_vs_mx_correlation.png', dpi=150)
print("  ✓ Salvato: heading_vs_mx_correlation.png")

# TEST 4: Heatmap 2D (heading vs mx)
print("\n📊 Generando heatmap heading vs mx...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Create 2D histogram
h, xedges, yedges = np.histogram2d(headings_deg, all_mx, bins=(36, 40), range=((-180, 180), (-1, 1)))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

im = ax.imshow(h.T, extent=extent, origin='lower', aspect='auto', cmap='hot', interpolation='nearest')
ax.set_xlabel('Heading (degrees)')
ax.set_ylabel('mx (left/right)')
ax.set_title('Distribuzione heading vs mx (heatmap)')
plt.colorbar(im, ax=ax, label='Numero di frames')

plt.tight_layout()
plt.savefig('heading_mx_heatmap.png', dpi=150)
print("  ✓ Salvato: heading_mx_heatmap.png")

print("\n" + "="*70)
print("CONCLUSIONE:")
if abs(corr_mx) > 0.15 or abs(corr_my) > 0.15:
    print("❌ FORTE CORRELAZIONE RILEVATA!")
    print("   → Il modello riceve heading come input")
    print("   → Può imparare che 'quando vado verso X, gira verso Y'")
    print("   → Il bias mx < 0 riflette una STRATEGIA SPAZIALE ASSOLUTA")
    print("   → NON è solo preferenza left/right ergonomica!")
    print("\n💡 SOLUZIONE:")
    print("   1. Rimuovi USE_HEADING=False per avere predizioni puramente reattive")
    print("   2. Oppure accetta il bias come feature del comportamento spaziale")
else:
    print("✅ Correlazione debole")
    print("   → Il bias è probabilmente ergonomico, non spaziale")
