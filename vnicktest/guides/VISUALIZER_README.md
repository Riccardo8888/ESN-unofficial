# ESN Real-time Probability Visualizer

## 📊 Overview

Visualizza in tempo reale le probabilità dei 37 bins di angolo predetti dall'ESN usando PyGame.

## 🚀 Utilizzo

### Opzione 1: Con il server WebSocket (modalità integrata)

```bash
python3 run_with_visualizer.py
```

Questo lancerà:
- Il server WebSocket ESN su `ws://localhost:8765`
- La finestra di visualizzazione PyGame

La visualizzazione si aggiornerà automaticamente ad ogni predizione dell'ESN.

### Opzione 2: Solo visualizer (modalità standalone/demo)

```bash
python3 visualizer.py
```

Mostra dati simulati per testare la visualizzazione.

## 🎨 Features

### Display principale
- **37 barre verticali**: Una per ogni bin di angolo da -90° a +90°
- **Colori**:
  - Blu: Probabilità normali
  - Arancione: Bin campionato/selezionato nell'ultimo frame
- **Scala dinamica**: L'altezza delle barre è normalizzata al valore massimo

### Informazioni mostrate
- **Frame count**: Numero di frame processati
- **Boost probability**: Probabilità di boost corrente
- **Sampled angle**: Angolo campionato nell'ultimo frame
- **Max probability**: Probabilità massima e angolo corrispondente
- **Valori numerici**: Probabilità esatte sulle barre alte

### Controlli
- **ESC**: Chiudi visualizer
- **X**: Chiudi finestra

## 🔧 Modifiche al modello

### Square delle probabilità

Il websocket ora applica **square** alle probabilità dopo il softmax:

```python
angle_probs_squared = angle_probs ** 2
angle_probs_squared = angle_probs_squared / np.sum(angle_probs_squared)
```

**Effetto**: I bin con probabilità più alta vengono enfatizzati ulteriormente, rendendo la distribuzione più "piccata" (peaked). Questo rende il sampling più deterministico e riduce il rumore nelle predizioni.

**Esempio**:
- Prima: `[0.3, 0.2, 0.1, ...]` 
- Dopo square: `[0.45, 0.2, 0.05, ...]` (il primo bin domina ancora di più)

## 📦 Requisiti

```bash
pip install pygame numpy
```

## 🎯 Come funziona

1. **websocket_reservoir.py** salva le ultime predizioni in:
   - `latest_angle_probs`: Array[37] con probabilità
   - `latest_boost_prob`: Float con probabilità boost
   - `latest_command`: Dict con comando campionato

2. **visualizer.py** legge questi valori e li visualizza:
   - Ogni frame PyGame (30 FPS)
   - Aggiorna le barre in tempo reale
   - Evidenzia il bin campionato

3. **run_with_visualizer.py** coordina:
   - Server WebSocket in thread separato
   - Visualizer in main thread (richiesto per GUI)

## 🐛 Troubleshooting

### "No models found"
Assicurati di aver trainato un modello:
```bash
python3 train_slither_reservoir.py
```

### "pygame.error: No available video device"
Stai usando SSH senza X11 forwarding. Usa la modalità standalone su una macchina con display.

### Visualizer si blocca
Premi ESC per chiudere correttamente. Se necessario, usa Ctrl+C.

## 📈 Interpretazione

### Distribuzione stretta (peaked)
- **Buono**: Il modello è sicuro della predizione
- Esempio: Una barra alta al centro, altre basse

### Distribuzione larga (uniform)
- **Meno sicuro**: Situazione ambigua o difficile
- Esempio: Più barre con altezza simile

### Boost alto (>50%)
- Rosso/arancione nella UI
- Il modello suggerisce di accelerare

### Angolo campionato vs max
- **Campionato**: Bin effettivamente scelto (può variare per randomness)
- **Max**: Bin con probabilità più alta (deterministico)
