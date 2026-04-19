# Differenze tra le Due Implementazioni ESN

## 📋 Overview

Ho creato **due versioni** del training ESN per Slither.io:

1. **`train_slither_esn.py`** - Implementazione standalone ESN
2. **`train_slither_reservoir.py`** - Compatible con `reservoir.py` (raccomandato ✓)

## 🔄 Approccio Raccomandato: `train_slither_reservoir.py`

Questo script è **compatibile con il vostro lavoro in `humand_data.ipynb`** e usa `reservoir.py`.

### Perché usare questo approccio?

✅ **Stesso workflow** che avete già testato e debuggato  
✅ Usa la classe `Reservoir` esistente  
✅ Stesso metodo `compute_wout()` con ridge regression  
✅ Facile da integrare con il vostro codice esistente

## 📊 Confronto Dettagliato

### 1. Creazione del Reservoir

**`train_slither_reservoir.py` (raccomandato):**
```python
# Usa reservoir.py - STESSO del notebook
from reservoir import Reservoir

reservoir = Reservoir(
    n_inputs=INPUT_DIM,
    n_neurons=N_RESERVOIR,
    rhow=SPECTRAL_RADIUS,
    inp_scaling=INPUT_SCALE,
    leak_range=(LEAK_RATE_MIN, LEAK_RATE_MAX)
)
```

**`train_slither_esn.py` (alternativo):**
```python
# Implementazione standalone con più features
from utilities.esn_model import SlitherESN

esn = SlitherESN(
    n_inputs=INPUT_DIM,
    n_reservoir=N_RESERVOIR,
    spectral_radius=SPECTRAL_RADIUS,
    sparsity=0.9  # Feature aggiuntiva: sparsity
)
```

### 2. Training

**`train_slither_reservoir.py` (raccomandato):**
```python
# STESSO APPROCCIO di humand_data.ipynb
# Step 1: Collect states
X_train = reservoir.forward(U_train, collect_states=True)

# Step 2: Compute output weights
def compute_wout(X, Y, T_washout, alpha):
    X = X[T_washout:]
    Y = Y[T_washout:]
    R = X.T @ X
    P = X.T @ Y
    return np.linalg.inv(R + alpha * np.eye(X.shape[1])) @ P

wout = compute_wout(X_train, y_train, WASHOUT, ALPHA)

# Step 3: Predict
y_pred = X_test @ wout
```

**`train_slither_esn.py` (alternativo):**
```python
# Training tutto in una funzione
esn.train(U_train, y_train, alpha=ALPHA, washout=WASHOUT)

# Prediction integrata
y_pred = esn.predict(U_test)
```

### 3. Gestione degli Stati

**`train_slither_reservoir.py`:**
```python
# Stati gestiti esplicitamente (più controllo)
X = reservoir.forward(U, collect_states=True)  # [timesteps, n_neurons]
# Puoi ispezionare, modificare, salvare X
wout = compute_wout(X, Y, washout, alpha)
```

**`train_slither_esn.py`:**
```python
# Stati gestiti internamente (più automatico)
esn.train(U, Y)  # Stati raccolti internamente
# Meno flessibilità ma più semplice
```

## 🎯 Quale Usare?

### Usa `train_slither_reservoir.py` se:
- ✅ Vuoi **compatibilità** con il lavoro fatto in `humand_data.ipynb`
- ✅ Vuoi usare lo **stesso codice** che avete già testato
- ✅ Vuoi **massimo controllo** sugli stati del reservoir
- ✅ Vuoi fare **debug** e visualizzazioni degli stati
- ✅ Vuoi **cross-validation** come nel notebook (facile da aggiungere)

### Usa `train_slither_esn.py` se:
- 🔄 Vuoi un'interfaccia più **object-oriented**
- 🔄 Vuoi features aggiuntive come **sparsity** configurabile
- 🔄 Vuoi **save/load** del modello integrato
- 🔄 Preferisci un approccio più **automatizzato**

## 📁 File Structure

```
ESN-unofficial/
├── reservoir.py                    # La vostra implementazione (invariata)
├── train_slither_reservoir.py     # ✅ RACCOMANDATO - usa reservoir.py
├── train_slither_esn.py           # Alternativo - standalone
├── utilities/
│   ├── esn_model.py               # Solo per train_slither_esn.py
│   ├── data_loader.py             # Usato da entrambi
│   └── metrics.py                 # Usato da entrambi
└── configuration.py                # Configurazione condivisa
```

## 🚀 Quick Start

### Opzione 1: Approccio Compatibile (Raccomandato)

```bash
python3 train_slither_reservoir.py
```

### Opzione 2: Approccio Standalone

```bash
python3 train_slither_esn.py
```

## 🔬 Differenze Tecniche Chiave

### Input Weights (`W_in`)

**reservoir.py:**
```python
# Random uniform [-1, 1] con bias
self.win = np.random.uniform(-1., 1., size=(n_neurons, n_inputs+1)) * inp_scaling
```

**SlitherESN:**
```python
# Stessa cosa, ma più esplicito
self.W_in = np.random.uniform(-1.0, 1.0, size=(n_reservoir, n_inputs + 1)) * input_scale
```

### Reservoir Weights (`W`)

**reservoir.py:**
```python
# Dense random [-1, 1]
self.w = np.random.random((n_neurons, n_neurons)) * 2. - 1.
# Scale to spectral radius
self.w = self.w * rhow / spectral_radius_current
```

**SlitherESN:**
```python
# Può essere sparse
self.W_res = np.random.uniform(-1.0, 1.0, size=(n_reservoir, n_reservoir))
if sparsity:
    mask = np.random.rand(n_reservoir, n_reservoir) > sparsity
    self.W_res = self.W_res * mask
# Scale to spectral radius
self.W_res = self.W_res * (spectral_radius / current_radius)
```

### Forward Pass

**reservoir.py:**
```python
# Leaky integrator per neuron
for t in range(n_timesteps):
    ut = u[t,:]
    x_next = np.tanh(self.win @ np.concatenate((ut, [1])) + self.w @ x)
    x = (1. - self.leak) * x + self.leak * x_next
    X[t,:] = x
```

**SlitherESN:**
```python
# Stessa cosa, ma con leak_rates per-neuron
def _reservoir_step(self, x, u):
    u_with_bias = np.concatenate([u, [1.0]])
    x_pre = np.tanh(self.W_in @ u_with_bias + self.W_res @ x)
    x_new = (1.0 - self.leak_rates) * x + self.leak_rates * x_pre
    return x_new
```

### Ridge Regression

**Entrambi usano lo stesso approccio:**
```python
R = X.T @ X                                    # State correlation
P = X.T @ Y                                    # State-output correlation
W_out = np.linalg.inv(R + alpha * I) @ P      # Ridge solution
```

## 💡 Raccomandazione Finale

**Usa `train_slither_reservoir.py`** perché:

1. ✅ È **compatibile al 100%** con il vostro lavoro in `humand_data.ipynb`
2. ✅ Usa `reservoir.py` che avete già testato
3. ✅ Segue lo **stesso workflow** (forward → compute_wout → predict)
4. ✅ Più facile da **debuggare** se avete già familiarità con il notebook
5. ✅ Più facile **aggiungere cross-validation** come nel notebook

L'altro script (`train_slither_esn.py`) è valido ma introduce un'astrazione diversa che potreste non voler imparare ora.

## 📝 Note Importanti

### Path dei Dati

Ho modificato `configuration.py` per usare la cartella locale:

```python
# ORA usa la cartella locale data/
SLITHER_DATA_PATH = Path(__file__).parent / "data"

# PRIMA era:
# SLITHER_DATA_PATH = Path("/Users/nick/Desktop/slitherio-scraper/backend/data")
```

### Zarr Compatibility

Ho fixato la compatibilità con Zarr 3.x in `data_loader.py`:

```python
try:
    # Try Zarr 3.x API
    from zarr import open_group
    root = open_group(str(session_path), mode='r')
except:
    # Fallback to Zarr 2.x API
    store = zarr.DirectoryStore(str(session_path))
    root = zarr.group(store=store)
```

## 🎓 Prossimi Passi

1. **Testa con i dati locali:**
   ```bash
   python3 train_slither_reservoir.py
   ```

2. **Se vuoi cross-validation** (come nel notebook), posso aggiungerlo facilmente

3. **Se vuoi ottimizzare parametri** (alpha, washout, ecc.), posso creare uno script di tuning

4. **Se vuoi visualizzazioni** degli stati del reservoir, posso aggiungerle

---

**TL;DR**: Usa `train_slither_reservoir.py` - è compatibile con il vostro approccio in `humand_data.ipynb`!
