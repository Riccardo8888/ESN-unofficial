# Slither.io Echo State Network Training

Questo progetto implementa un Echo State Network (ESN) per predire le azioni del giocatore (direzione e sprint) nel gioco Slither.io basandosi sullo stato del gioco.

## 🎯 Obiettivo

Dato lo stato del gioco al frame `t`, l'ESN predice le azioni del giocatore al frame `t+5` (configurabile):
- **Direzione** (mx, my): componenti normalizzate della direzione del movimento
- **Sprint** (boost): stato del boost (0 o 1)

## 📁 Struttura del Progetto

```
ESN-unofficial/
├── configuration.py          # Configurazione centrale
├── train_slither_esn.py     # Script principale di training
├── utilities/
│   ├── __init__.py
│   ├── data_loader.py       # Caricamento dati Zarr
│   ├── esn_model.py         # Implementazione ESN
│   └── metrics.py           # Metriche di valutazione
├── slither_esn_results/     # Output (creata automaticamente)
└── SLITHER_ESN_README.md    # Questa guida
```

## 🚀 Quick Start

### 1. Prerequisiti

Assicurati di avere i dati raccolti con il [slither.io-scraper](https://github.com/NickP005/slitherio-scraper):

```bash
/Users/nick/Desktop/slitherio-scraper/backend/data/
├── nick/
│   ├── session_1234567890/
│   │   ├── grids/
│   │   ├── player_inputs/
│   │   ├── timestamps/
│   │   └── ...
│   └── session_1234567891/
└── riccardo/
    └── session_...
```

### 2. Installazione Dipendenze

```bash
cd /Users/nick/Desktop/SSN-Folder/ESN-unofficial
pip install numpy zarr
```

### 3. Configurazione

Modifica `configuration.py` se necessario:

```python
# Percorso ai dati
SLITHER_DATA_PATH = Path("/Users/nick/Desktop/slitherio-scraper/backend/data")

# Predizione
PREDICTION_HORIZON = 5  # Frame futuri da predire

# ESN
N_RESERVOIR = 1000      # Neuroni nel reservoir
SPECTRAL_RADIUS = 1.25  # Raggio spettrale
```

### 4. Eseguire il Training

```bash
python train_slither_esn.py
```

L'output includerà:
- Metriche di accuracy sul training set
- Metriche di accuracy sul test set
- Modello salvato
- Risultati JSON

## 📊 Output e Risultati

### Directory di Output

```
slither_esn_results/
└── training_20241027_143022/
    ├── slither_esn_model.pkl      # Modello ESN salvato
    └── training_results.json      # Metriche e configurazione
```

### Metriche Valutate

**Direzione (mx, my):**
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Angular Error**: Errore angolare in gradi

**Boost:**
- **Accuracy**: Percentuale di previsioni corrette
- **Precision/Recall/F1**: Metriche di classificazione
- **Confusion Matrix**: TP, TN, FP, FN

**Esempio di Output:**
```
TEST SET METRICS
============================================================

📊 Overall:
  MSE (all outputs): 0.042156

🎯 Direction Prediction (mx, my):
  RMSE: mx=0.1842, my=0.2105, avg=0.1974
  MAE:  mx=0.1456, my=0.1623, avg=0.1540
  Angular Error: 18.45° (0.3220 rad)

⚡ Boost Prediction (binary):
  Accuracy:  0.8923 (89.23%)
  Precision: 0.8456
  Recall:    0.7834
  F1 Score:  0.8133
```

## 🔧 Configurazione Avanzata

### Parametri ESN

Modifica in `configuration.py`:

```python
# Dimensione del reservoir
N_RESERVOIR = 1000  # Più grande = più capacità ma più lento

# Raggio spettrale (dinamica del reservoir)
SPECTRAL_RADIUS = 1.25  # >1 = memoria più lunga

# Sparsity (connettività)
SPARSITY = 0.9  # 0.9 = 90% connessioni a zero

# Leak rate (leaky integrator)
LEAK_RATE_MIN = 0.1
LEAK_RATE_MAX = 0.3

# Regolarizzazione
ALPHA = 1e-3  # Ridge regression
```

### Dati di Input

Configurazione features in `configuration.py`:

```python
# Oltre alla griglia polare (6144 valori), includi:
USE_VELOCITY = True             # Velocità del serpente
USE_HEADING = True              # Direzione (sin, cos)
USE_DISTANCE_TO_BORDER = True   # Distanza dal bordo
```

## 📈 Interpretazione dei Risultati

### Metriche di Direzione

- **Angular Error < 20°**: Buona predizione della direzione
- **Angular Error 20-40°**: Predizione moderata
- **Angular Error > 40°**: Predizione debole

### Metriche di Boost

- **Accuracy > 85%**: Buona predizione del boost
- **F1 Score > 0.75**: Bilanciamento precision/recall accettabile

### Overfitting

Il confronto Train vs Test mostra:
```
Test/Train MSE ratio: 1.15 ✓ (good generalization)
Train-Test Accuracy diff: 0.03 ✓ (good generalization)
```

- Ratio < 1.2: Buona generalizzazione
- Ratio > 1.5: Possibile overfitting (ridurre N_RESERVOIR o aumentare ALPHA)

## 🧪 Test dei Componenti

### Test Data Loader
```bash
python utilities/data_loader.py
```

### Test ESN Model
```bash
python utilities/esn_model.py
```

### Test Metrics
```bash
python utilities/metrics.py
```

## 🔍 Troubleshooting

### Errore: "No data found"

Verifica che i dati esistano:
```bash
ls -la /Users/nick/Desktop/slitherio-scraper/backend/data/*/session_*
```

### Errore: "Module not found"

Installa le dipendenze:
```bash
pip install numpy zarr
```

### Training troppo lento

Riduci la dimensione del reservoir:
```python
N_RESERVOIR = 500  # invece di 1000
```

### Accuracy troppo bassa

1. Aumenta il reservoir: `N_RESERVOIR = 2000`
2. Riduci prediction horizon: `PREDICTION_HORIZON = 3`
3. Aumenta i dati (gioca più partite)

## 📚 Struttura Dati Slither.io

### Input (Frame t)
- **Grid**: [64, 24, 4] griglia polare
  - Canale 0: Cibo
  - Canale 1: Corpi nemici
  - Canale 2: Corpo proprio
  - Canale 3: Teste nemici
- **Metadata**: velocity, heading, distance_to_border

### Output (Frame t+5)
- **player_inputs**: [mx, my, boost]
  - mx: Componente X direzione [-1, 1]
  - my: Componente Y direzione [-1, 1]
  - boost: Sprint attivo [0, 1]

## 🎓 Come Funziona l'ESN

1. **Reservoir Computing**: Il reservoir è una rete ricorrente sparsa con pesi fissi
2. **Input Encoding**: Lo stato del gioco viene proiettato nel reservoir
3. **State Collection**: Gli stati interni del reservoir vengono raccolti
4. **Readout Training**: Solo i pesi di output sono addestrati (ridge regression)
5. **Prediction**: Dato nuovo input, il reservoir genera stati e produce l'output

**Vantaggi ESN:**
- Training veloce (solo regression lineare)
- Buono per sequenze temporali
- Non richiede backpropagation

## 📝 Note Tecniche

### Prediction Horizon

Con `PREDICTION_HORIZON = 5` e sampling rate 10 Hz:
- Previsione = 0.5 secondi nel futuro
- Abbastanza per pianificare il movimento
- Non troppo lontano da essere imprevedibile

### Washout Period

I primi `WASHOUT` frames sono scartati per permettere al reservoir di stabilizzarsi. Default: 50 frames.

### Train/Test Split

Split casuale delle sessioni (non dei frame):
- 80% sessioni per training
- 20% sessioni per test
- Evita data leakage tra train e test

## 🤝 Contributi

Sviluppato da Nick e Riccardo per il progetto Echo State Networks.

## 📄 Licenza

Vedi il repository principale ESN-unofficial.

---

Per domande o problemi, consulta il codice commentato o contatta gli sviluppatori.
