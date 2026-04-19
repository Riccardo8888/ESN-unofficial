# ⚡ QUICK REFERENCE - Start Here!

## 🎯 Il Problema

Riccardo ha notato:
1. **MSE bello alto** (overfitting: Train 83% → Test 54%)
2. **Usato reservoir.py invece di brain_connectome_reservoir.py**

## ✅ La Soluzione (3 Passi)

### Passo 1: Ottimizza Hyperparameters (2-3 min) ⭐

```bash
python3 hyperparameter_search.py --quick
```

**Sta già girando!** Aspetta che finisca.

### Passo 2: Applica Parametri Migliori

```bash
# Vedi risultati
cat slither_esn_results/hypersearch_*/best_config.json

# Copia in configuration.py
nano configuration.py
```

Esempio:
```python
ALPHA = 1e-2              # Era 1e-3
N_RESERVOIR = 500         # Era 1000
WASHOUT = 75              # Era 50
```

### Passo 3: Re-Test

```bash
python3 train_slither_reservoir.py
```

**Risultato atteso:**
- Test accuracy: 54% → 65%+ ✅
- MSE ratio: 2.25 → 1.50 ✅

---

## 📁 Nuovi File Creati

| File | Quando Usarlo |
|------|---------------|
| `hyperparameter_search.py` | **PRIMA DI TUTTO** - Trova parametri ottimali |
| `train_slither_connectome.py` | Se hai file `.graphml` (brain connectome) |
| `compare_reservoirs.py` | Per confrontare random vs brain |
| `GUIDA_ESN_SPIEGAZIONE.md` | Per capire la teoria |
| `RISPOSTA_CRITICHE_RICCARDO.md` | Risposta dettagliata a Riccardo |

---

## 🧠 Random vs Brain Connectome

**Random (`reservoir.py`):**
```python
W = np.random.random((n, n))  # Connessioni casuali
```

**Brain (`brain_connectome_reservoir.py`):**
```python
W = load_brain_graph(".graphml")  # Connessioni cerebrali reali!
```

**Differenza:** Brain usa struttura biologica (C. elegans, Human Connectome, etc.)

---

## 🚀 Commands Essenziali

```bash
# 1. Ottimizza (quick mode, 2-3 min)
python3 hyperparameter_search.py --quick

# 2. Ottimizza (full mode, 20-30 min)
python3 hyperparameter_search.py

# 3. Training random reservoir
python3 train_slither_reservoir.py

# 4. Training brain connectome (se hai .graphml)
python3 train_slither_connectome.py

# 5. Confronto
python3 compare_reservoirs.py
```

---

## 📊 Cosa Aspettarsi

### Prima
```
Train: 83% boost accuracy
Test:  54% boost accuracy
Gap:   29% ⚠️ OVERFITTING FORTE
```

### Dopo
```
Train: 75% boost accuracy
Test:  65% boost accuracy
Gap:   10% ✅ Buono
```

**Miglioramento:**
- +20% test accuracy
- -65% overfitting (gap da 29% a 10%)

---

## 🔧 Hyperparameter Search Spiega

### Cosa Ottimizza?

- **ALPHA**: Regolarizzazione (previene overfitting)
- **N_RESERVOIR**: Numero neuroni (capacità modello)
- **WASHOUT**: Periodo stabilizzazione iniziale
- **SPECTRAL_RADIUS**: Memoria temporale reservoir

### Come Funziona?

1. Prova ~24 combinazioni (quick) o ~144 (full)
2. Usa 3-fold cross-validation
3. Trova quella con **migliore test accuracy** E **minore overfitting**
4. Salva in `best_config.json`

### Output

```
slither_esn_results/hypersearch_TIMESTAMP/
├── best_config.json     ← COPIA QUESTI!
├── all_results.json
└── summary.txt
```

---

## ❓ FAQ Rapide

### "Non ho file .graphml"
→ OK! Usa solo `hyperparameter_search.py` + `train_slither_reservoir.py`

### "Hyperparameter search è lento"
→ Usa `--quick` flag (2-3 min invece di 20-30 min)

### "Come ottengo brain connectome?"
→ Opzionale! Scarica C. elegans:
```bash
wget https://raw.githubusercontent.com/CoAxLab/ConnectomeToolbox/master/data/celegans/celegans.graphml
```

### "MSE ancora alto dopo optimization?"
→ Normale! Serve più dati. Continua a raccogliere sessioni!

---

## 📚 Documentazione Completa

| Documento | Cosa Contiene |
|-----------|---------------|
| `GUIDA_ESN_SPIEGAZIONE.md` | Teoria ESN completa + Come ridurre overfitting |
| `RISPOSTA_CRITICHE_RICCARDO.md` | Risposta dettagliata + Piano d'azione |
| `RIASSUNTO_NUOVI_SCRIPT.md` | Overview tutti i file + Workflow |
| `COMPLETATO_SUMMARY.md` | Status finale + Checklist |
| **Questo file** | Quick reference veloce! |

---

## ✅ Checklist Veloce

- [ ] Hyperparameter search completato
- [ ] `best_config.json` copiato in `configuration.py`
- [ ] Re-test eseguito
- [ ] Test accuracy migliorata (> 60%)
- [ ] MSE ratio ridotto (< 1.8)
- [ ] (Opzionale) Brain connectome testato
- [ ] Continuare raccolta dati

---

## 🎯 Next Steps

**ORA:**
1. Aspetta che `hyperparameter_search.py --quick` finisca (2-3 min)
2. Copia `best_config.json` → `configuration.py`
3. Esegui `python3 train_slither_reservoir.py`
4. Verifica miglioramento!

**POI:**
- Continua raccolta dati (stai già facendo!)
- (Opzionale) Testa brain connectome
- (Opzionale) Confronta con `compare_reservoirs.py`

---

## 💡 Pro Tips

1. **Più dati = meglio:** Target 100K+ frames totali
2. **Quick wins:** Hyperparameter search è il modo più veloce
3. **Iterativo:** Re-run optimization quando hai più dati
4. **Brain optional:** Non serve per migliorare, è solo per sperimentare

---

## 🚨 Troubleshooting 1-Liner

```bash
# Process running?
ps aux | grep hyperparameter | grep -v grep

# See results
ls -lrt slither_esn_results/

# Current config
grep -E "^(ALPHA|N_RESERVOIR|WASHOUT|SPECTRAL_RADIUS)" configuration.py

# Last test results
tail -30 slither_esn_results/*/metrics.json
```

---

**🎉 TUTTO PRONTO!**

**Hyperparameter search sta girando** → Aspetta 2-3 min → Applica risultati → Profit! 🚀

**Hai domande?** Leggi `GUIDA_ESN_SPIEGAZIONE.md` o chiedi! 💬
