# 📥 Dataset Information - Automatic Download

## Dataset Utilizzato

**Nome**: MS MARCO Hard Negatives  
**Source**: Hugging Face Datasets  
**ID**: `sentence-transformers/msmarco-hard-negatives`  
**Dimensione**: ~2-3 GB

## Download Automatico

✅ **Non devi scaricare nulla manualmente!**

Il dataset viene scaricato automaticamente da Hugging Face quando avvii il training per la prima volta.

### Processo di Download

1. Alla prima esecuzione del training, vedrai:
   ```
   Loading MS MARCO hard negatives…
   Downloading data files: 100%|██████████| ...
   Generating train split: ...
   ```

2. **Tempo richiesto**: 5-10 minuti (dipende dalla velocità internet di Colab)

3. **Cache**: Dopo il primo download, il dataset viene salvato nella cache di Hugging Face e non verrà riscaricato

### Dove Viene Salvato

Il dataset viene salvato automaticamente in:
```
~/.cache/huggingface/datasets/
```

Su Colab, questo si trova in:
```
/root/.cache/huggingface/datasets/
```

### Dettagli Dataset

- **Queries**: ~500K query di ricerca
- **Hard Negatives**: Documenti negativi difficili da MS MARCO
- **Formato**: Triplet (anchor, positive, negative)
- **Uso**: Training di embedding models con contrastive learning

### Samples Utilizzati

Per il training, il codice usa:
- **Max train samples**: 100,000 triplets
- **Validation**: 2% (~2,000 samples)

Questo è configurabile nel file `Config`:
```python
max_train_samples: int = 100000
val_ratio: float = 0.02
```

### Requisiti

**Spazio su disco necessario**:
- Dataset cache: ~2-3 GB
- Checkpoints durante training: ~3-5 GB
- **Totale**: ~6-8 GB liberi su runtime Colab

**Memoria RAM durante loading**: ~2-4 GB (temporaneamente)

### Cosa Succede se il Download Fallisce?

Se il download si interrompe:
1. Hugging Face riprende automaticamente dal punto di interruzione
2. Se persiste, verifica connessione internet di Colab
3. Puoi ri-eseguire la cella di training - riprenderà il download

### Alternative (Non Necessarie)

Se vuoi pre-scaricare il dataset localmente (non necessario per Colab):
```python
from datasets import load_dataset
ds = load_dataset("sentence-transformers/msmarco-hard-negatives", split="train")
print(f"Loaded {len(ds)} samples")
```

---

## 🎯 Bottom Line

✅ **Tutto automatico!**  
Avvia semplicemente il training, il dataset viene scaricato automaticamente la prima volta e cached per usi successivi.

**Tempo totale prima volta**: +5-10 min per download dataset + 6-8h training  
**Tempo esecuzioni successive**: Solo 6-8h training (dataset già cached)
