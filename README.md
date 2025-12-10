# 🔥 Spectra-NSA: Neural Semantic Architecture

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()

> **⚠️ ALPHA RELEASE**: Spectra-NSA is under active development. Breaking changes may occur. Use for research and experimentation.

Advanced neural embedding model combining **spectral-entropy attention**, **Matryoshka representations**, and **anomalous detection** for state-of-the-art semantic similarity tasks.

---

## 🎯 Key Features

- 🧠 **Spectral-Entropy Attention** - Novel attention mechanism with learnable fractal enrichment
- 🎭 **Matryoshka Embeddings** - Multi-scale semantic representations (768→512→256)
- 🔍 **Anomalous Detection** - Built-in out-of-distribution detection via learned anomalous basis
- 📊 **Multiple Size Presets** - M456 (458M), M600 (600M), M700 (700M), M1B (1B params)
- 🎛️ **Component Toggles** - Flexible ablation studies via CLI flags
- 📈 **Real-time Monitoring** - Integrated SOTA comparison and health checks
- 💾 **Auto-backup to Google Drive** - Automatic checkpoint backup every 6 saves

---

## 📊 Model Sizes & Performance

| Model | Parameters | STS-B Target | VRAM | Training Time (A100) | Best For |
|-------|-----------|--------------|------|---------------------|----------|
| **M456** | 458M | >0.825 | ~35GB | 4-6h | **Colab Pro** ✅ |
| M600 | 600M | >0.835 | ~37GB | 6-8h | Balanced |
| M700 | 700M | >0.840 | ~38GB | 8-10h | SOTA target |
| M1B | 1.0B | >0.850 | ~55GB | 12-16h | Research |

**Default**: M456 - Optimized for Google Colab Pro (A100 40GB)

---

## 🚀 Quick Start

### Installation

```bash
pip install transformers datasets accelerate scipy torch
```

### Training (Default M456)

```bash
python anomalous_embedding_ultimate.py --mode train --epochs 3
```

**Expected output**:
- Training time: ~4-6 hours (A100, fp32 debug mode)
- STS-B score: >0.825
- Automatic checkpoints every 500 steps
- Drive backup every 6 checkpoints

### Evaluation

```bash
python anomalous_embedding_ultimate.py --mode eval --checkpoint checkpoints/best_sts.pt
```

### Embedding Extraction

```bash
python anomalous_embedding_ultimate.py \
    --mode extract \
    --checkpoint checkpoints/best_sts.pt \
    --texts "sample text" "another text"
```

---

## 🎛️ Advanced Usage

### Size Presets

```bash
# SOTA target (M700)
python anomalous_embedding_ultimate.py --size M700 --mode train --epochs 3

# Research scale (M1B) - requires 80GB GPU
python anomalous_embedding_ultimate.py --size M1B --mode train --epochs 3
```

### Resume Training

```bash
# Resume training from a checkpoint
python anomalous_embedding_ultimate.py --mode train --resume checkpoints/step-5000.pt

# Resume with different number of epochs
python anomalous_embedding_ultimate.py --mode train --resume checkpoints/step-5000.pt --epochs 5
```

### Memory Optimization

```bash
# Enable gradient checkpointing for lower VRAM usage
python anomalous_embedding_ultimate.py --mode train --gradient-checkpointing

# Combine with smaller batch size
python anomalous_embedding_ultimate.py --mode train --gradient-checkpointing --size M700
```

### Multi-GPU Training

```bash
# Launch with accelerate for multi-GPU training
accelerate launch anomalous_embedding_ultimate.py --mode train --epochs 3

# Configure accelerate (first time only)
accelerate config
```

### Ablation Studies

```bash
# Disable spectral attention
python anomalous_embedding_ultimate.py --no-spectral --mode train --epochs 1

# Disable anchor64 head
python anomalous_embedding_ultimate.py --no-anchor --mode train --epochs 1

# Minimal configuration
python anomalous_embedding_ultimate.py \
    --no-spectral --no-anchor --no-bridge --no-matryoshka \
    --mode train --epochs 1
```

---

## 🏗️ Architecture Overview

### Core Components

1. **CustomEncoder** - Transformer backbone with spectral-entropy attention
   - Learnable fractal depth mixing
   - Learnable spectral fusion weights
   - Temperature-annealed contrastive learning

2. **Matryoshka Heads** - Multi-scale embeddings
   - Semantic: 768, 512, 256 dimensions
   - Entity: 384, 192, 96 dimensions
   - Progressive nesting for efficient inference

3. **Anomalous Projection** - OOD detection
   - Learned anomalous basis (16 prototypes)
   - Spectral regularization
   - Ranking head for retrieval tasks

4. **LossStack** - Multi-objective training
   - InfoNCE (semantic, anchor, fast retrieval)
   - Triplet margin loss
   - Matryoshka angular alignment
   - Bridge loss (semantic↔entity)
   - Spectral entropy regularization

### Training Features

- **Temperature Scheduling**: Cosine decay (0.07→0.05)
- **Gradient Accumulation**: Effective batch size 64
- **Mixed Precision**: FP16 support (currently disabled for debugging)
- **Gradient Checkpointing**: Memory-efficient training for larger models
- **Resume Training**: Full checkpoint restoration with RNG states
- **Multi-GPU Support**: Distributed training via Accelerate
- **Early Stopping**: Configurable patience (default: disabled)
- **Auto-backup**: Google Drive sync every 6 checkpoints

---

## 📁 Project Structure

```
NSA_2.0/
├── anomalous_embedding_ultimate.py  # Main training script
├── training_monitor.py              # Real-time monitoring & health checks
├── anomalous_eval_suite.py          # Comprehensive evaluation suite
├── colab_training.ipynb             # Google Colab training notebook
├── ULTIMATE_GUIDE.md                # Detailed usage guide
├── DATASET_INFO.md                  # Dataset information
├── GOOGLE_DRIVE_SETUP.md            # Drive setup instructions
└── checkpoints/                     # Saved models (auto-created)
```

---

## 🔧 Configuration

Key parameters in `Config` dataclass:

```python
# Model Architecture
hidden_size: int = 1024          # M456 default
num_layers: int = 24
num_heads: int = 16
spectral_dim: int = 192
max_length: int = 160            # VRAM optimized

# Training
batch_train: int = 8             # Physical batch size
grad_accum: int = 8              # Effective batch = 64
epochs: int = 3
lr: float = 2e-4
fp16: bool = False               # Debug mode (use fp32)

# Monitoring
save_every: int = 500            # Checkpoint frequency
eval_every: int = 500            # STS evaluation frequency
early_stop_patience: int = 9999  # Disabled (duration by epochs)
```

---

## 📈 Training Metrics

### Expected Timeline (M456, 3 epochs, 98k samples)

| Step | Epoch | Event |
|------|-------|-------|
| 500 | 0.04 | First STS evaluation (~0.68) |
| 3,000 | 0.24 | **1st Drive backup** |
| 12,250 | 1.00 | End epoch 1 (STS ~0.78) |
| 24,500 | 2.00 | End epoch 2 (STS ~0.82) |
| 36,750 | 3.00 | **Final** (STS >0.825) |

**Total training steps**: ~36,750  
**Drive backups**: ~12 (every 3,000 steps)  
**Checkpoints saved**: ~73 (every 500 steps)

---

## 🎯 Performance Targets

### M456 (Default)
- ✅ STS-B > 0.825 (competitive with BGE-base)
- ✅ Fits in Colab Pro (40GB VRAM)
- ✅ Training completes in 4-6h (A100)
- ✅ No OOM errors

### M700 (SOTA)
- ✅ STS-B > 0.840 (competitive with GTE-large)
- ✅ NDCG@10 > 0.420
- ✅ Recall@1 > 0.550
- ✅ Training completes in 8-10h (A100)

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

**For M456**:
```python
# Edit Config in anomalous_embedding_ultimate.py
batch_train = 6        # Reduce from 8
max_length = 128       # Reduce from 160
```

**For M700**:
- Switch to M456 or use 80GB GPU
- Enable gradient checkpointing (future)

### Training Too Slow

Check GPU type:
```python
!nvidia-smi
```
Should show: **A100-SXM4-40GB** or **A100-80GB**

### STS Not Improving

1. Verify all components enabled (no `--no-*` flags)
2. Check temperature annealing is active
3. Monitor loss components (should decrease)
4. Wait until warmup ends (8% of training)

### Checkpoints Not Backing Up to Drive

Verify Drive is mounted:
```python
import os
print(os.path.exists("/content/drive/MyDrive"))  # Should be True
```

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{spectra_nsa,
  title = {Spectra-NSA: Neural Semantic Architecture - Advanced Embeddings},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/spectra-nsa},
  note = {Alpha release - Architecture under active development}
}
```

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

```
Copyright 2025

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## ⚠️ Disclaimer

**ALPHA SOFTWARE**: This architecture is experimental and under active development.

- ✅ **Use for**: Research, experimentation, benchmarking
- ⚠️ **Not recommended for**: Production systems without extensive testing
- 🔄 **Breaking changes**: May occur between releases
- 🐛 **Known issues**: NaN debugging enabled (fp32 mode for stability)

**Current Status**:
- Core architecture: **Stable**
- Training pipeline: **Stable**
- Evaluation suite: **Stable**
- Resume training: **Implemented** ✅
- Gradient checkpointing: **Implemented** ✅
- Multi-GPU support: **Implemented** ✅
- Mixed precision (fp16): **Disabled for debugging**

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Areas of interest**:
- Multi-GPU training optimization
- FP16 stability improvements
- Additional evaluation benchmarks
- Memory optimization techniques
- Documentation improvements

---

## 📞 Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Email: your.email@example.com
- Discord: [Your Server Link]

---

## 🙏 Acknowledgments

- **Hugging Face** - Transformers & Datasets libraries
- **PyTorch Team** - Deep learning framework
- **Sentence-Transformers** - MS MARCO dataset
- **MTEB** - Benchmark datasets
- **Google Colab** - Training infrastructure

---

## 📚 Additional Resources

- [ULTIMATE_GUIDE.md](ULTIMATE_GUIDE.md) - Detailed usage guide
- [DATASET_INFO.md](DATASET_INFO.md) - Dataset information
- [GOOGLE_DRIVE_SETUP.md](GOOGLE_DRIVE_SETUP.md) - Colab setup guide
- [colab_training.ipynb](colab_training.ipynb) - Training notebook

---

**Built with ❤️ for the NLP research community**

🔗 **Links**:

- Repository: [spectra-nsa](https://github.com/yourusername/spectra-nsa)
- Documentation: Coming Soon
- Paper: In preparation

---

*Last updated: November 15, 2025*
