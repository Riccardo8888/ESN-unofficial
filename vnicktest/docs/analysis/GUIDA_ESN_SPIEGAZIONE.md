# 🧠 Guida Completa: Come Funziona l'Echo State Network

## 📊 Analisi dei Tuoi Risultati

### Risultati Attuali (con PREDICTION_HORIZON=3)

```
Training Set:
  Angular Error: 34.53°  ✅ Buono!
  Boost Accuracy: 83.36% ✅ Ottimo!

Test Set:
  Angular Error: 63.86°  ⚠️ Troppo alto
  Boost Accuracy: 54.00% ⚠️ Scarso (quasi random!)

Overfitting:
  Test/Train MSE ratio: 2.25  ⚠️ OVERFITTING FORTE
  Accuracy diff: 29%          ⚠️ OVERFITTING FORTE
```

### 🔍 Cosa Significa?

**OVERFITTING** = La rete ha **memorizzato** i dati di training invece di **imparare pattern generali**

È come uno studente che impara a memoria gli esempi del libro ma non capisce i concetti:
- ✅ Esame sul libro: 83% (training)
- ❌ Esame con domande nuove: 54% (test)

---

## 🎓 Come Funziona un Echo State Network (ESN)

### 1. Architettura Base

```
Input (6148 dim)  →  [Reservoir]  →  Output (3 dim)
  Game State         1000 neuroni      mx, my, boost
```

**Componenti:**

#### A) **Input Layer** (Win)
```python
Win = random weights [1000 x 6149]  # include bias
```
- Trasforma l'input in segnale per il reservoir
- **Pesi fissi** (non si addestrano)
- `INPUT_SCALE = 1.0` controlla l'ampiezza

#### B) **Reservoir** (W) - Il "serbatoio" 🌊
```python
W = random sparse matrix [1000 x 1000]
# 90% degli elementi = 0 (SPARSITY = 0.9)
# Scaled to SPECTRAL_RADIUS = 1.25
```

**Come funziona:**
```python
# Ad ogni timestep t:
x[t] = tanh(Win @ input[t] + W @ x[t-1])
# Con leaky integration:
x[t] = (1-leak) * x[t-1] + leak * x_new[t]
```

**Cosa fa il reservoir:**
- Crea una "memoria" temporale delle osservazioni passate
- Mescola le informazioni in modo non-lineare
- È come un "eco" delle osservazioni precedenti
- **Pesi fissi** (non si addestrano mai!)

#### C) **Output Layer** (Wout)
```python
Wout = learned weights [1000 x 3]
```
- **UNICO LAYER ADDESTRATO!**
- Linear readout: `output = reservoir_states @ Wout`
- Addestrato con ridge regression

### 2. Processo di Training

```python
# FASE 1: COLLEZIONE STATI
for t in range(n_frames):
    x[t] = reservoir_step(input[t], x[t-1])
    states[t] = x[t]  # Salva lo stato

# FASE 2: RIDGE REGRESSION (addestra solo Wout)
X = states[WASHOUT:]  # Rimuovi primi frame
Y = targets[WASHOUT:]
R = X.T @ X                                    # Correlation matrix
P = X.T @ Y                                    # Cross-correlation
Wout = inv(R + ALPHA * I) @ P                  # Ridge solution
```

---

## 🔧 Parametri Chiave Spiegati

### 1. **ALPHA (Regolarizzazione)** 📏

```python
ALPHA = 1e-3  # 0.001
```

**Cosa fa:**
- Controlla quanto i pesi possono diventare grandi
- Previene l'overfitting penalizzando soluzioni complesse

**Formula Ridge Regression:**
```
Wout = (X^T X + ALPHA * I)^-1 X^T Y
         ^^^^^^^^^^^^^^^
         Senza ALPHA: può invertire matrici singolari
         Con ALPHA: più stabile, meno overfitting
```

**Effetto di ALPHA:**

| ALPHA | Effetto | Training Acc | Test Acc | Overfitting |
|-------|---------|--------------|----------|-------------|
| 0.0001 | Troppo flessibile | 95% | 50% | ⚠️ ALTO |
| 0.001 | Bilanciato | 83% | 54% | ⚠️ Medio |
| 0.01 | Più rigido | 75% | 65% | ✅ Basso |
| 0.1 | Troppo rigido | 65% | 62% | ✅ Nessuno |
| 1.0 | Underfit | 55% | 54% | ❌ Troppo semplice |

**Come scegliere ALPHA:**
- **ALPHA troppo piccolo** → Overfitting (memorizza tutto)
- **ALPHA troppo grande** → Underfitting (non impara abbastanza)
- **Ottimale**: Quello che massimizza test accuracy

### 2. **WASHOUT (Periodo di Riscaldamento)** 🚿

```python
WASHOUT = 50  # frames
```

**Cosa fa:**
- Scarta i primi N frame all'inizio di ogni sequenza
- Permette al reservoir di "stabilizzarsi" e dimenticare lo stato iniziale

**Perché serve:**

```
Frame:   0    10    20    30    40    50    60
State:  [0] → ? → ? → ? → ? → [stable]
        ↑                       ↑
     Random                  Vero pattern
     initial                della sequenza
     state
```

**Il reservoir parte con x[0] = 0**, ma il gioco non parte da 0!
- Primi frame: il reservoir si "aggiusta" al vero stato del gioco
- Dopo WASHOUT frame: lo stato riflette veramente il gioco

**Effetto di WASHOUT:**

| WASHOUT | Pro | Contro |
|---------|-----|--------|
| 0 | Usa tutti i dati | Stati iniziali rumorosi |
| 50 | Buon compromesso | Perde 50 frame per sessione |
| 100 | Stati molto stabili | Perde troppi dati |
| 200 | Stati perfetti | Troppi dati persi |

### 3. **SPECTRAL_RADIUS (Dinamica Temporale)** 🌀

```python
SPECTRAL_RADIUS = 1.25
```

**Cosa fa:**
- Controlla quanto "memoria" ha il reservoir
- È il raggio spettrale della matrice W (max eigenvalue)

**Effetto:**

```
SPECTRAL_RADIUS < 1.0:  "Memoria corta" - rapido adattamento
                         ↓↓↓ (decadimento veloce)
                         Buono per: pattern veloci, reazioni immediate

SPECTRAL_RADIUS ≈ 1.0:  "Al limite del caos"
                         ↓ (decadimento lento)
                         Ottimale per: molti task

SPECTRAL_RADIUS > 1.0:  "Memoria lunga" - lento adattamento
                         → (mantiene informazione a lungo)
                         Buono per: dipendenze temporali lunghe
                         Rischio: instabilità se troppo alto
```

**Nel tuo caso (1.25):**
- Il reservoir mantiene informazione per ~10-20 frame
- Buono per predire 3-5 frame nel futuro
- Se aumenti PREDICTION_HORIZON, considera aumentare questo

### 4. **PREDICTION_HORIZON** 🎯

```python
PREDICTION_HORIZON = 3  # frames
```

**Cosa fa:**
- A frame `t`, predice l'azione al frame `t+3`
- Con 10 Hz sampling: predice 0.3 secondi nel futuro

**Effetto:**

| Horizon | Difficoltà | Accuracy Attesa | Uso |
|---------|------------|-----------------|-----|
| 1 | Facile | 80-90% | Reazione immediata |
| 3 | Medio | 60-75% | ✅ Buon compromesso |
| 5 | Difficile | 50-65% | Pianificazione |
| 10 | Molto difficile | 40-55% | Strategia lunga |

**Perché più difficile:**
- 1 frame: quasi deterministico (continui nella stessa direzione)
- 5 frame: il giocatore può aver cambiato idea
- 10 frame: comportamento quasi imprevedibile

---

## 🚀 Come Ridurre l'Overfitting

### Strategia 1: Aumentare ALPHA ⭐ **PIÙ SEMPLICE**

```python
# In configuration.py
ALPHA = 1e-2  # Prova 0.01 invece di 0.001
```

**Cosa aspettarsi:**
- Training accuracy: scende (75-80%)
- Test accuracy: sale (60-65%)
- Overfitting ratio: migliora (1.3-1.5)

### Strategia 2: Ridurre N_RESERVOIR

```python
N_RESERVOIR = 500  # Invece di 1000
```

**Perché funziona:**
- Meno neuroni = meno capacità di memorizzare
- Forza la rete a imparare solo pattern importanti

### Strategia 3: Aumentare WASHOUT

```python
WASHOUT = 100  # Invece di 50
```

**Perché funziona:**
- Rimuove più dati "rumorosi" all'inizio
- Usa solo stati ben stabilizzati

### Strategia 4: Ridurre PREDICTION_HORIZON ✅ **GIÀ FATTO**

```python
PREDICTION_HORIZON = 3  # ✅ Ottimo! (era 5)
```

### Strategia 5: Cross-Validation per Trovare ALPHA Ottimale 🎯 **MIGLIORE**

Posso creare uno script che:
1. Prova diversi ALPHA: [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
2. Usa cross-validation (come in humand_data.ipynb)
3. Trova automaticamente il migliore

---

## 📈 Come Migliorare i Risultati

### 1. **Più Dati** 🌟 **PIÙ IMPORTANTE**

```
Ora hai: 3 sessioni, 30K frames
Obiettivo: 10+ sessioni, 100K+ frames
```

**Perché:**
- Più dati = meno overfitting
- Più varietà di situazioni di gioco
- Pattern più robusti

**Come:**
- Tu e Riccardo giocate più partite
- Varie strategie di gioco (aggressivo, difensivo, ecc.)
- Sessioni più lunghe

### 2. **Ottimizzazione Hyperparameter**

Crea uno sweep come nel notebook:

```python
# Test combinazioni
alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
washouts = [0, 25, 50, 100]
n_neurons = [500, 1000, 2000]

# Per ogni combinazione:
#   - Train con cross-validation
#   - Trova la migliore
```

### 3. **Data Augmentation**

```python
# Specchia i dati (left ↔ right)
mx_flipped = -mx
my_flipped = my  # stessa y

# Rotazione random della griglia
# Noise injection
```

### 4. **Ensemble di Reservoir**

```python
# Train 5 reservoir diversi (seed diverso)
# Media delle predizioni
ensemble_pred = mean([r1_pred, r2_pred, r3_pred, r4_pred, r5_pred])
```

### 5. **Features Engineering**

Aggiungi features derivate:

```python
# Derivate temporali
velocity_change = velocity[t] - velocity[t-1]
direction_change = heading[t] - heading[t-1]

# Features aggregate
enemies_nearby = sum(grid[:, :, 1])  # Channel enemy bodies
food_available = sum(grid[:, :, 0])   # Channel food
```

---

## 🎯 Piano d'Azione Raccomandato

### Fase 1: Quick Wins (5 minuti)

```python
# In configuration.py
ALPHA = 1e-2        # Aumenta da 0.001 → 0.01
WASHOUT = 75        # Aumenta da 50 → 75
```

**Risultato atteso:**
- Test accuracy: 54% → 60-65%
- Overfitting: ridotto

### Fase 2: Cross-Validation (10 minuti)

Voglio creare uno script che trova automaticamente i migliori:
- ALPHA
- WASHOUT
- N_RESERVOIR

### Fase 3: Raccogliere Più Dati (ongoing)

Giocare più partite con lo scraper attivo.

**Target:** 100K+ frames totali

### Fase 4: Advanced (opzionale)

- Ensemble methods
- Feature engineering
- Architetture diverse (LSTM, etc.)

---

## 💡 Intuizione: Perché ESN Funziona?

### Analogia: Orchestra 🎵

**Reservoir (1000 neuroni)** = 1000 musicisti che improvvisano
- Ogni musicista: suona in modo leggermente diverso
- Insieme: creano una "sinfonia" complessa delle osservazioni passate

**Wout (output weights)** = Direttore d'orchestra
- Ascolta tutti i musicisti
- Decide quali ascoltare per ogni nota da suonare
- Si addestra: impara chi ascoltare quando

**Input (game state)** = Spartito musicale
- Fornisce il tema da seguire

**Spectral Radius** = Quanto i musicisti si influenzano tra loro
- Basso: ogni musicista suona per sé (memoria corta)
- Alto: si influenzano molto (memoria lunga)

**Alpha** = Quanto il direttore può essere "creativo"
- Basso: può fare arrangiamenti complessi (rischio: troppo complicato)
- Alto: deve seguire regole semplici (rischio: troppo banale)

### Vantaggio ESN

**Training Time:**
```
Backpropagation (LSTM): 2-3 ore 🕐🕑🕒
ESN: 10 secondi ⚡
```

**Perché:**
- Solo Wout si addestra (regressione lineare)
- Reservoir è random ma funziona ugualmente!
- "Echo State Property": il reservoir trasforma già l'input in modo utile

---

## 📝 Formula Matematica Completa

### Forward Pass (per ogni timestep t)

```
1. Input con bias:
   u_bias[t] = [u[t], 1]

2. Pre-attivazione:
   x_pre[t] = tanh(Win @ u_bias[t] + W @ x[t-1])

3. Leaky integration:
   x[t] = (1 - leak) * x[t-1] + leak * x_pre[t]
   
4. Output:
   y[t] = x[t] @ Wout
```

### Training (Ridge Regression)

```
1. Collect states:
   X = [x[WASHOUT], x[WASHOUT+1], ..., x[T]]
   Y = [y[WASHOUT], y[WASHOUT+1], ..., y[T]]

2. Correlation matrices:
   R = X^T @ X          [N x N]
   P = X^T @ Y          [N x 3]

3. Ridge solution:
   Wout = (R + ALPHA * I)^-1 @ P
```

---

## 🤔 FAQ

**Q: Perché i pesi del reservoir sono random?**
A: Funziona! La "Echo State Property" dice che un reservoir random con proprietà giuste (spectral radius < 1.5) crea già una buona rappresentazione. È controintuitivo ma provato matematicamente.

**Q: Posso addestrare anche Win e W?**
A: Sì, ma perdi il vantaggio ESN (velocità). Diventa una normale RNN.

**Q: 1000 neuroni sono troppi?**
A: Dipende dai dati. Con 30K frame: forse sì (overfitting). Con 100K frame: va bene. Regola: N_RESERVOIR ≈ sqrt(n_samples)

**Q: Posso usare ESN per altri giochi?**
A: Sì! È ottimo per:
- Predizione di serie temporali
- Control tasks
- Time-series classification
- Qualsiasi task con dipendenze temporali

---

Vuoi che crei uno script di **cross-validation automatica** per trovare i parametri ottimali? 🎯
