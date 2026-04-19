# WebSocket ESN Server - Quick Start Guide

## 🚀 Setup e Testing

### 1. Installa dipendenze WebSocket
```bash
pip install websockets
```

### 2. Verifica che hai un modello addestrato
```bash
ls -la output/
# Dovresti vedere almeno una cartella training_YYYYMMDD_HHMMSS/
# con dentro un file reservoir_model.npz
```

Se non hai un modello, addestralo prima:
```bash
python3 train_slither_reservoir.py
```

### 3. Avvia il server WebSocket
```bash
# Usa l'ultimo modello addestrato (automatico)
python3 websocket_reservoir.py

# Oppure specifica un modello
python3 websocket_reservoir.py --model output/training_20251027_193139/reservoir_model.npz

# Opzioni disponibili:
# --host HOST    Host to listen on (default: 0.0.0.0)
# --port PORT    Port to listen on (default: 8765)
```

Output atteso:
```
============================================================
LOADING ESN MODEL
============================================================
Loading model from: output/training_20251027_193139/reservoir_model.npz
✓ Model loaded successfully!
  - Input dim: 6150
  - Reservoir size: 1000
  - Spectral radius: 1.00
  - Output dim: 3
  - Leak rate: [0.10, 0.30]

============================================================
STARTING WEBSOCKET SERVER
============================================================
Host: 0.0.0.0
Port: 8765
URL: ws://0.0.0.0:8765
============================================================

🚀 Server ready! Waiting for client connection...
   (Press Ctrl+C to stop)
```

### 4. Testa il server (in un altro terminale)
```bash
python3 test_websocket_client.py
```

Output atteso:
```
============================================================
TESTING ESN WEBSOCKET SERVER
============================================================
Connecting to: ws://127.0.0.1:8765
✅ Connected!

📥 Received ready message:
   Session ID: 1730048400123
   Model info: {'reservoirSize': 1000, ...}

============================================================
SENDING TEST FRAMES
============================================================

📊 Frame 0:
   Latency: 5.23ms
   Command: angleDelta=2.5°, boost=False
   Confidence: 0.823
   Processing time: 3.45ms

...

============================================================
STATISTICS
============================================================
Frames sent: 100
Avg latency: 5.67ms
Min latency: 4.12ms
Max latency: 8.34ms
Std latency: 0.89ms

✅ PASS: Latency < 20ms (target met!)

✅ Test completed successfully!
```

---

## 🎮 Uso con Extension Browser

### 1. Configura extension
1. Apri popup extension
2. Check "Enable AI Control"
3. AI Server URL: `ws://127.0.0.1:8765`
4. Save Settings

### 2. Avvia server
```bash
python3 websocket_reservoir.py
```

### 3. Gioca su slither.io
1. Vai su https://slither.io
2. Click "Connect to AI" nel popup
3. Aspetta che spawni il verme
4. Click "Start AI Control"
5. **Il verme è ora controllato dall'AI! 🐍🤖**

### 4. Monitor performance
- **Popup**: Mostra latency real-time
- **Console browser** (F12): Log dettagliati ogni 50 frames
- **Terminal server**: Statistics ogni 50 frames

---

## 📊 Performance Attese

### Server Inference
- **Latency target**: < 20ms round-trip
- **Processing time**: ~3-5ms per frame
- **Max FPS**: ~200-300 (molto più veloce del necessario)

### Model Performance (da training)
- **Angular Error**: ~36° medio
- **Boost Accuracy**: ~81%
- **Direction RMSE**: ~0.48

### Network
- **Frame rate**: 10 Hz (100ms interval)
- **Payload size**: ~50KB JSON per frame
- **Bandwidth**: ~500KB/s

---

## 🔧 Troubleshooting

### "No training directories found"
```bash
# Addestra un modello prima
python3 train_slither_reservoir.py
```

### "Connection refused" dal client
```bash
# Verifica che il server sia in esecuzione
# In un altro terminale:
python3 websocket_reservoir.py

# Verifica che la porta sia aperta
netstat -an | grep 8765
```

### "Server busy" dall'extension
- Il server accetta **una connessione alla volta**
- Chiudi altre connessioni prima (Stop AI Control + Disconnect)
- Riavvia il server se necessario

### Latency alta (>50ms)
1. **Check CPU**: `top` o `htop` - verifica che non ci siano altri processi pesanti
2. **Check network**: Se usi `host=0.0.0.0`, prova `127.0.0.1` per local only
3. **Check model**: Modelli più grandi (>2000 neurons) potrebbero essere lenti

### AI fa movimenti strani
1. **Normalizzazione**: Verifica che grid e metadata siano normalizzati correttamente
2. **Coordinate**: Slither.io ha asse Y invertito (positivo = giù)
3. **Training data**: L'AI impara dai tuoi dati - se hai giocato male, anche l'AI giocherà male 😅
4. **Confidence**: Abilita confidence threshold nell'extension per ignorare comandi incerti

---

## 📈 Monitoring e Debugging

### Server logs
```bash
# Il server stampa ogni 50 frames:
📊 Processed 51 frames | Avg inference: 3.45ms | Max FPS: 289.9

# Alla disconnessione:
📊 Session Statistics:
  - Duration: 30.5s
  - Frames processed: 305
  - Avg inference time: 3.52ms
  - Avg FPS: 10.0
```

### Extension logs (Console browser)
```javascript
// Enable debug mode
AIControl.setDebugMode(true);

// Check status
AIControl.getStatus();

// Output:
{
    isConnected: true,
    isControlling: true,
    framesSent: 152,
    commandsReceived: 152,
    avgLatency: 5.3,
    errors: 0
}
```

### Test performance isolata
```bash
# Usa il test client per misurare solo inference performance
python3 test_websocket_client.py

# Invia 100 frames e misura latency
```

---

## 🎯 Next Steps

### Fase 3: Integration Testing
- [ ] Test con dati reali di gioco
- [ ] Test durata estesa (>5 minuti)
- [ ] Test reconnection automatico
- [ ] Misura win rate AI vs human

### Fase 4: Model Improvements
- [ ] Raccogliere più dati di training
- [ ] Hyperparameter tuning (usa `hyperparameter_search_binary.py`)
- [ ] Provare reservoir più grandi
- [ ] Ensemble di modelli

### Fase 5: Features Avanzate
- [ ] Human override (detect mouse movement)
- [ ] Auto-start AI quando game inizia
- [ ] Dashboard monitoring real-time
- [ ] Recording e replay di sessioni AI

---

## 🐛 Known Issues

### Issue 1: Movimenti non smooth
**Causa**: `angleDelta` applicato direttamente ogni frame senza smoothing

**Fix**: Nell'extension, applica filtro smoothing:
```javascript
// In ai-control.js
smoothedAngleDelta = lastAngleDelta * 0.7 + angleDelta * 0.3;
```

### Issue 2: Boost spam
**Causa**: Boost probability vicino a 0.5 causa flickering

**Fix**: Aggiungi hysteresis nell'extension:
```javascript
const BOOST_ON_THRESHOLD = 0.55;
const BOOST_OFF_THRESHOLD = 0.45;

if (boost_prob > BOOST_ON_THRESHOLD) boost = true;
else if (boost_prob < BOOST_OFF_THRESHOLD) boost = false;
// else: mantieni stato precedente
```

### Issue 3: Server crash su malformed JSON
**Causa**: Frame data corrotto o incompleto

**Fix**: Già gestito con try-catch, ma verifica logs per pattern

---

## 📚 Architecture Reference

### Data Flow
```
Extension (slither.io)
    ↓ [Frame Data - JSON]
WebSocket Server (websocket_reservoir.py)
    ↓ [Features preparation]
ESN Model (reservoir_model.npz)
    ↓ [Inference: mx, my, boost_prob]
Control Command Calculation
    ↓ [angleDelta + boost]
WebSocket Response
    ↓ [Command JSON]
Extension Control Application
    ↓ [window.xm, window.ym, KeyboardEvent]
Snake Movement! 🐍
```

### Feature Vector (6150 dims)
```python
[
    grid_flat[6144],      # Grid 64×24×4 flattened
    heading_sin,          # sin(heading)
    heading_cos,          # cos(heading)
    velocity,             # Normalized [0, 1]
    boost,                # 0 or 1
    distance_to_border,   # Normalized [0, 1]
    snake_length          # Raw value
]
```

### ESN Output (3 dims)
```python
[mx_pred, my_pred, boost_prob]

# Post-processing:
predicted_angle = arctan2(my_pred, mx_pred)
angle_delta = predicted_angle - current_heading
angle_delta = normalize_angle(angle_delta)  # [-π, π]
boost = (boost_prob > 0.5)
```

---

## 🎓 Tips per Migliorare Performance AI

1. **Raccogliere dati di qualità**: Gioca bene quando raccogli dati!
2. **Diversifica stili**: Raccogli dati da sessioni diverse (aggressivo, difensivo, etc.)
3. **Balance dataset**: Assicurati che boost=true e boost=false siano bilanciati
4. **Hyperparameter tuning**: Usa gli script di search per trovare config ottimale
5. **Reservoir size**: Prova 1500-2000 neurons se hai abbastanza dati (>200k frames)
6. **Prediction horizon**: Prova 3-4 frames (300-400ms) per planning più a lungo termine
7. **Ensemble**: Allena 3-5 modelli e fai voting sui comandi

---

## ✅ Checklist Deployment

- [x] Server WebSocket implementato
- [x] Model loading funzionante
- [x] Test client creato
- [x] Feature preparation corretta
- [x] Control command calculation corretta
- [x] Error handling robusto
- [ ] Test con extension browser
- [ ] Test sessione >5 minuti
- [ ] Misura win rate
- [ ] Deploy su server remoto (optional)

---

**Happy AI Gaming! 🐍🤖🎮**
