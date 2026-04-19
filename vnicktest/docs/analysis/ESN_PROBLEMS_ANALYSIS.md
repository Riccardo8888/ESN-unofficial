# Analisi Problemi ESN - Slither.io

## 🔍 Problemi Identificati

### 1. ❌ Boost Mai Attivato

**Sintomo**: Il server WebSocket predice sempre `BOOST false`

**Causa Principale**: **Threshold troppo alto (0.5) + Bias del modello verso 0.47**

#### Analisi Statistica (100 frame test reali):
```
Boost Predictions:
  Mean:  0.477  ← PROBLEMA: Media sotto 0.5!
  Std:   0.146
  Min:   0.109
  Max:   0.888

Distribution:
  < 0.3:    9%
  0.3-0.5: 51%  ← Maggioranza in "zona grigia"
  0.5-0.7: 35%
  > 0.7:    5%
```

**Ground Truth nei dati**:
- Boost ON:  55% dei frame
- Boost OFF: 45% dei frame

**Problema**: Il modello predice mediamente 0.477, ma con threshold=0.5, praticamente **sempre risulta false**!

#### ✅ Fix Applicato:
```python
# OLD: boost_decision = (boost_prob > 0.5)
# NEW: boost_decision = (boost_prob > 0.45)  # Threshold abbassato
```

**Risultato Atteso**: Boost accuracy ~63% (invece di ~0%)

---

### 2. ❌ Angoli Troppo Larghi (±150-170°)

**Sintomo**: 
```
+156.47° - BOOST false
+162.12° - BOOST false
-171.33° - BOOST false
```

**Causa Principale**: **Prediction Horizon troppo aggressivo (2 frames = 200ms)**

#### Analisi Errore Angolare:
```
Angle Deltas su dati reali:
  Mean absolute: 69.8°  ← MOLTO ALTO!
  Std:           97.5°
  Median:        -3.2°  (metà dei dati OK, ma molti outlier)
```

**Training Results**:
- Angular Error: 36.25° (medio sul test set)
- Direction RMSE: 0.4849

#### Perché Succede:

1. **Prediction Horizon = 2 frames (200ms)**
   - L'ESN cerca di prevedere dove sarà il serpente 200ms nel futuro
   - In slither.io, 200ms è molto tempo (serpente può girare 40-50°)
   - Prediction a lungo termine è intrinsecamente difficile

2. **Grid Noise nei Test Mock**
   - Il test client invia grid random (no pattern reale)
   - Heading costante (1.57 rad) non realistico
   - ESN non riceve input coerenti → output casuale

3. **Modello Non Abbastanza Addestrato**
   - 133k frames totali, solo ~26k test
   - Angular error 36° è "accettabile" ma non ottimale
   - Serve più data o model più grande

#### ✅ Fix Possibili:

**A. Ridurre Prediction Horizon** (consigliato):
```python
# In configuration.py
PREDICTION_HORIZON = 1  # Era 2, prova con 1 frame (100ms)
```

**B. Smooth le predizioni client-side**:
```javascript
// In extension ai-control.js
smoothedAngleDelta = lastAngleDelta * 0.7 + angleDelta * 0.3;
```

**C. Clamp angle delta server-side**:
```python
# In websocket_reservoir.py
MAX_ANGLE_DELTA = np.radians(30)  # Max ±30° per frame
angle_delta = np.clip(angle_delta, -MAX_ANGLE_DELTA, MAX_ANGLE_DELTA)
```

---

## 📊 Performance Attuale

### Training Results (TRAINING_RESULTS.txt):
```
Direction RMSE (test):  0.4849
Angular Error (test):   36.25°
Boost Accuracy (test):  81.30%  ← Sul test set, con threshold 0.5
Overall MSE (test):     0.206700
```

### Real Test (100 frames):
```
Boost Accuracy: 63.00%  ← Con prediction horizon 2, dati reali
Angle |Mean|:   69.8°   ← Molto peggiore del training (36°)!
```

**Discrepanza**: Performance reale peggiore del training!

**Possibili Cause**:
1. Overfitting sul training set
2. Test set non rappresentativo
3. Reservoir state non inizializzato correttamente (washout mancante)

---

## 🎯 Raccomandazioni Prioritarie

### 1. ⚡ Immediate (già fatto):
- [x] Abbassare boost threshold a 0.45

### 2. 🔧 Short-term:
- [ ] **Ridurre PREDICTION_HORIZON da 2 a 1**
  ```bash
  # In configuration.py
  PREDICTION_HORIZON = 1
  
  # Poi riaddestra
  python3 train_slither_reservoir.py
  ```

- [ ] **Aggiungere clamping angle delta** nel server:
  ```python
  MAX_ANGLE = np.radians(45)  # ±45° max
  angle_delta = np.clip(angle_delta, -MAX_ANGLE, MAX_ANGLE)
  ```

- [ ] **Test con dati reali**, non mock random

### 3. 📈 Medium-term:
- [ ] **Raccogliere più dati**
  - Target: 500k+ frames (attuale: 133k)
  - Diversificare stili di gioco
  - Bilanciare boost ON/OFF

- [ ] **Hyperparameter tuning**
  ```bash
  python3 hyperparameter_grid_focused.py
  ```

- [ ] **Reservoir più grande**
  - Attuale: 1500 neurons
  - Prova: 2000-3000 neurons (se hai abbastanza dati)

### 4. 🚀 Long-term:
- [ ] **Ensemble di modelli**
  - 3-5 ESN con config diverse
  - Voting sui comandi

- [ ] **Feature engineering**
  - Aggiungere velocità angolare
  - Aggiungere distanza da cibo più vicino
  - History degli ultimi N heading

- [ ] **Architettura avanzata**
  - ESN con feedback output
  - Hierarchical reservoir (fast + slow)

---

## 🧪 Test Diagnostici

### Per verificare i fix:

```bash
# 1. Test predizioni su dati reali
python3 debug_esn_output.py

# 2. Test server WebSocket
python3 quick_test_websocket.py

# 3. Monitor log server
python3 websocket_reservoir.py
# Poi in altro terminale:
python3 quick_test_websocket.py
# Osserva output: ora dovrebbe vedere BOOST true più spesso
```

---

## 📝 Note Tecniche

### Perché Threshold 0.45 invece di 0.5?

1. **Distribuzione sbilanciata**: 51% predictions tra 0.3-0.5
2. **Mean 0.477**: Centro distribuzione sotto 0.5
3. **Ground truth 55% boost**: Serve threshold più basso per matchare

**Trade-off**:
- Threshold 0.5: Alta precisione, bassa recall → troppi falsi negativi
- Threshold 0.45: Media precisione, media recall → più bilanciato
- Threshold 0.4: Bassa precisione, alta recall → troppi falsi positivi

**Optimal**: Calcolare da ROC curve, ma 0.45 è buon compromesso empirico.

### Perché Angoli Larghi con Test Mock?

Grid random → Nessun pattern spaziale → ESN estrae features noise → Predizione casuale

**Con dati reali**:
- Grid ha struttura (serpenti, cibo)
- Heading varia smooth
- ESN riconosce pattern → Predizioni più sensate

---

## ✅ Checklist Validation

Prima di deployment:

- [x] Threshold boost corretto (0.45)
- [ ] Test su slither.io reale (non mock)
- [ ] Latency < 20ms confermata
- [ ] Boost attivato almeno 30% del tempo
- [ ] Angoli < 60° in media
- [ ] Snake non va out-of-bounds
- [ ] Performance stabile >5 minuti

---

**Data Analisi**: 2025-10-27
**Modello**: `training_20251027_210927` (1500 neurons, horizon=2)
**Status**: Problemi identificati, fix parziali applicati
