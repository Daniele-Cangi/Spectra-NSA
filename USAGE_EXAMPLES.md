# Spectra-NSA v0.2.0 - Usage Examples

Quick reference guide for the new features in Spectra-NSA v0.2.0-alpha.

---

## 🔄 Resume Training

### Basic Resume
```bash
# Train for 3 epochs
python anomalous_embedding_ultimate.py --mode train --epochs 3

# Training interrupted at step 5000? Resume from last checkpoint
python anomalous_embedding_ultimate.py --mode train --resume checkpoints/step-5000.pt
```

### Resume with Extended Training
```bash
# Original training: 3 epochs
python anomalous_embedding_ultimate.py --mode train --epochs 3

# Resume and train for 2 more epochs (total 5)
python anomalous_embedding_ultimate.py --mode train --resume checkpoints/step-18375.pt --epochs 5
```

### Resume from Best Checkpoint
```bash
# Resume from best STS checkpoint (recommended)
python anomalous_embedding_ultimate.py --mode train --resume checkpoints/best_sts.pt
```

---

## 💾 Gradient Checkpointing

### Enable for Memory Savings
```bash
# Default M456 with gradient checkpointing (~30% less VRAM)
python anomalous_embedding_ultimate.py --mode train --gradient-checkpointing

# M700 on 40GB GPU (requires checkpointing)
python anomalous_embedding_ultimate.py --size M700 --gradient-checkpointing --mode train

# M1B on 80GB GPU (optional checkpointing for safety)
python anomalous_embedding_ultimate.py --size M1B --gradient-checkpointing --mode train
```

### Trade-offs
- ✅ **VRAM usage**: Reduced by ~30%
- ⚠️ **Training speed**: Slower by ~20%
- ✅ **Use case**: Train larger models on smaller GPUs

---

## 🚀 Multi-GPU Training

### Setup (First Time)
```bash
# Configure Accelerate
accelerate config

# Answer the prompts:
# - Use distributed training? Yes
# - How many GPUs? [2/4/8]
# - Mixed precision? fp16 or no (currently use 'no' for stability)
```

### Basic Multi-GPU Training
```bash
# Launch on all available GPUs
accelerate launch anomalous_embedding_ultimate.py --mode train --epochs 3

# Effective batch size scales automatically:
# - 1 GPU: 8 × 8 = 64
# - 2 GPUs: 8 × 8 × 2 = 128
# - 4 GPUs: 8 × 8 × 4 = 256
```

### Multi-GPU + Resume
```bash
# Resume training on multiple GPUs
accelerate launch anomalous_embedding_ultimate.py \
    --mode train \
    --resume checkpoints/step-5000.pt
```

### Multi-GPU + Gradient Checkpointing
```bash
# Train M1B on 4x 40GB GPUs with gradient checkpointing
accelerate launch anomalous_embedding_ultimate.py \
    --size M1B \
    --gradient-checkpointing \
    --mode train \
    --epochs 3
```

---

## 🔥 Combined Workflows

### Scenario 1: Long Training with Interruptions
```bash
# Initial training
accelerate launch anomalous_embedding_ultimate.py --mode train --epochs 5

# Interrupted? Resume from automatic checkpoint
accelerate launch anomalous_embedding_ultimate.py \
    --mode train \
    --resume checkpoints/step-10000.pt
```

### Scenario 2: Large Model on Limited Hardware
```bash
# M700 on 2x 40GB GPUs with gradient checkpointing
accelerate launch anomalous_embedding_ultimate.py \
    --size M700 \
    --gradient-checkpointing \
    --mode train \
    --epochs 3
```

### Scenario 3: SOTA Push with M1B
```bash
# M1B on 8x 80GB GPUs (optimal configuration)
accelerate launch anomalous_embedding_ultimate.py \
    --size M1B \
    --mode train \
    --epochs 3

# Effective batch size: 8 × 8 × 8 = 512 (excellent for stability)
# Training time: ~2h (vs ~16h on single GPU)
```

### Scenario 4: Resume with Configuration Change
```bash
# Original: 3 epochs, no checkpointing
python anomalous_embedding_ultimate.py --mode train --epochs 3

# Resume with gradient checkpointing enabled (memory constraint discovered)
python anomalous_embedding_ultimate.py \
    --mode train \
    --resume checkpoints/step-5000.pt \
    --gradient-checkpointing \
    --epochs 5
```

---

## 📊 Monitoring and Evaluation

### Check Training Progress
```bash
# Checkpoints are saved every 500 steps
ls -lh checkpoints/

# Expected files:
# - step-500.pt, step-1000.pt, ... (incremental)
# - best_sts.pt (best Spearman correlation)
# - best_val.pt (best validation loss)
# - final.pt (end of training)
```

### Evaluate Checkpoint
```bash
# Evaluate best checkpoint
python anomalous_embedding_ultimate.py \
    --mode eval \
    --checkpoint checkpoints/best_sts.pt

# Expected output:
# Recall@10: 0.XXX | STS (Spearman): 0.XXX
```

### Extract Embeddings
```bash
# Extract embeddings from trained model
python anomalous_embedding_ultimate.py \
    --mode extract \
    --checkpoint checkpoints/best_sts.pt \
    --texts "Quantum mechanics reveals hidden patterns" "Simple sentence"

# Output: Embeddings shape: (2, 256)
```

---

## 🐛 Troubleshooting

### Resume Training Not Working
```bash
# Check checkpoint exists
ls -l checkpoints/step-5000.pt

# Verify checkpoint is valid (should show model, opt, lr_sched, etc.)
python -c "import torch; ckpt = torch.load('checkpoints/step-5000.pt', weights_only=False); print(list(ckpt.keys()))"
```

### Out of Memory with Gradient Checkpointing
```bash
# Reduce batch size further
# Edit Config in anomalous_embedding_ultimate.py:
# cfg.batch_train = 6  (reduce from 8)
# cfg.grad_accum = 10  (increase to maintain effective batch of 60)
```

### Multi-GPU Training Not Scaling
```bash
# Check GPU utilization
nvidia-smi

# All GPUs should show similar utilization
# If not, check Accelerate configuration:
accelerate env
```

---

## 💡 Best Practices

### 1. Always Use Best Checkpoint for Evaluation
```bash
# ❌ Don't use final.pt
python anomalous_embedding_ultimate.py --mode eval --checkpoint checkpoints/final.pt

# ✅ Use best_sts.pt
python anomalous_embedding_ultimate.py --mode eval --checkpoint checkpoints/best_sts.pt
```

### 2. Resume from Recent Checkpoint After Interruption
```bash
# Find latest checkpoint
ls -lt checkpoints/step-*.pt | head -1

# Resume from it
python anomalous_embedding_ultimate.py --mode train --resume checkpoints/step-18500.pt
```

### 3. Use Gradient Checkpointing for Models > 600M Parameters
```bash
# M456: Optional (fits in 40GB)
python anomalous_embedding_ultimate.py --mode train

# M700: Recommended (tight fit in 40GB)
python anomalous_embedding_ultimate.py --size M700 --gradient-checkpointing --mode train

# M1B: Required (doesn't fit in 40GB without it)
python anomalous_embedding_ultimate.py --size M1B --gradient-checkpointing --mode train
```

### 4. Multi-GPU for Faster Iteration
```bash
# Research: Fast iteration on 2-4 GPUs
accelerate launch anomalous_embedding_ultimate.py --mode train --epochs 1

# Production: Full training on 8 GPUs
accelerate launch anomalous_embedding_ultimate.py --mode train --epochs 3
```

---

## 📈 Expected Training Times

### Single GPU (A100 40GB)
| Model | No Checkpointing | With Checkpointing |
|-------|------------------|-------------------|
| M456  | 4-6h             | 5-7h              |
| M700  | OOM              | 10-12h            |
| M1B   | OOM              | OOM               |

### Multi-GPU (2x A100 40GB)
| Model | No Checkpointing | With Checkpointing |
|-------|------------------|-------------------|
| M456  | 2-3h             | 2.5-3.5h          |
| M700  | OOM              | 5-6h              |
| M1B   | OOM              | 12-14h            |

### Multi-GPU (4x A100 80GB)
| Model | No Checkpointing | With Checkpointing |
|-------|------------------|-------------------|
| M456  | 1-1.5h           | 1.2-1.8h          |
| M700  | 2-3h             | 2.5-3.5h          |
| M1B   | 4-5h             | 5-6h              |

---

## 🎯 Quick Decision Matrix

**Choose your configuration based on:**

| Hardware | Model | Gradient Checkpointing | Multi-GPU | Command |
|----------|-------|----------------------|-----------|---------|
| 1x 40GB  | M456  | No                   | No        | `python ... --mode train` |
| 1x 40GB  | M700  | Yes                  | No        | `python ... --size M700 --gradient-checkpointing --mode train` |
| 2x 40GB  | M456  | No                   | Yes       | `accelerate launch ... --mode train` |
| 2x 40GB  | M700  | Yes                  | Yes       | `accelerate launch ... --size M700 --gradient-checkpointing --mode train` |
| 4x 80GB  | M1B   | Optional             | Yes       | `accelerate launch ... --size M1B --mode train` |
| 8x 80GB  | M1B   | No                   | Yes       | `accelerate launch ... --size M1B --mode train` |

---

*For complete documentation, see [README.md](README.md) and [ULTIMATE_GUIDE.md](ULTIMATE_GUIDE.md)*
