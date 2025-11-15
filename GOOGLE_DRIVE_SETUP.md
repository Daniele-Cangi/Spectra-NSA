# 📁 Setup Google Drive per Colab Training

## Struttura Cartelle

Crea questa cartella in Google Drive:

```
MyDrive/
└── anomalous_embedding/
    ├── anomalous_embedding_ultimate.py    (obbligatorio)
    ├── training_monitor.py                 (opzionale ma consigliato)
    └── anomalous_eval_suite.py            (opzionale per eval completa)
```

## Nome Cartella

**Nome esatto**: `anomalous_embedding`

**Percorso completo in Drive**: `/content/drive/MyDrive/anomalous_embedding/`

## Files da Caricare

### 1. anomalous_embedding_ultimate.py ✅ OBBLIGATORIO
- File principale per training
- Contiene il modello e la logica di training
- Dimensione: ~40KB

### 2. training_monitor.py ⭐ CONSIGLIATO
- Monitoring avanzato durante il training
- Health checks e metriche in tempo reale
- Opzionale ma utile per vedere progressi dettagliati

### 3. anomalous_eval_suite.py 📊 OPZIONALE
- Evaluation completa contro SOTA models
- Benchmark su 7 dataset STS + MS MARCO + clustering
- Usa dopo il training per analisi approfondita

## Come Caricare

1. Apri Google Drive nel browser
2. Vai in "Il mio Drive" (MyDrive)
3. Clicca "Nuova cartella" → chiama `anomalous_embedding`
4. Entra nella cartella
5. Trascina i 3 file Python dalla tua cartella locale

## Verifica Setup

Nel notebook Colab, dopo aver montato Drive ed eseguito la cella di copy:

```python
!ls -lh /content/anomalous/
```

Dovresti vedere:
```
-rw-r--r-- 1 root root  40K  anomalous_embedding_ultimate.py
-rw-r--r-- 1 root root  15K  training_monitor.py
-rw-r--r-- 1 root root  25K  anomalous_eval_suite.py
```

## Risoluzione Problemi

### "No such file or directory"
→ Controlla che:
- La cartella si chiami esattamente `anomalous_embedding` (no spazi, tutto minuscolo)
- I file siano dentro quella cartella (non in sottocartelle)
- Drive sia montato correttamente

### "training_monitor.py not found"
→ È normale se non hai caricato il file (è opzionale)
→ Il training funziona comunque, ma senza monitoring avanzato

### "anomalous_eval_suite.py not found"
→ È normale se non hai caricato il file (è opzionale)
→ Puoi fare eval base con `--mode eval` del file principale

## Backup Checkpoints

I checkpoints vengono salvati in:
```
/content/drive/MyDrive/anomalous_checkpoints/
```

Questa cartella viene creata automaticamente dal notebook.

## Spazio Richiesto su Drive

- **Files sorgente**: ~1 MB
- **Checkpoints durante training**: ~3-5 GB (per M456)
- **Totale consigliato**: almeno 10 GB liberi su Drive

---

✅ **Tutto pronto!** Dopo aver caricato i files, puoi eseguire il notebook Colab.
