# Changelog - Spectra-NSA

All notable changes to Spectra-NSA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha] - 2025-11-15 - **Spectra-NSA Launch**

### Added ✨

#### Architecture & Training
- **Spectral-Entropy Attention** - Novel attention mechanism with learnable fractal enrichment
- **Matryoshka Embeddings** - Multi-scale semantic representations (768→512→256)
- **Anomalous Detection** - Built-in OOD detection via learned anomalous basis
- **Multiple Model Sizes** - M456 (458M, default), M600 (600M), M700 (700M), M1B (1B params)
- **Component Toggles** - CLI flags for ablation studies (--no-spectral, --no-anchor, etc.)

#### Training & Optimization
- **Temperature Scheduling** - Cosine decay (0.07→0.05)
- **Gradient Accumulation** - Effective batch size 64 with batch_train=8
- **Mixed Precision Support** - FP16 ready (currently FP32 for debugging)
- **Auto-backup to Google Drive** - Every 6 checkpoints (3000 steps)
- **Integrated Monitoring** - Real-time health checks and SOTA comparison

#### Evaluation & Monitoring
- **Quick STS Evaluation** - 200-sample validation every 500 steps
- **Health Monitoring** - Gradient stats, embedding quality, anomaly detection
- **Training Monitor** - Real-time metrics dashboard with warmup phase (5000 steps)
- **Recall@K Evaluation** - Built-in evaluation metrics
- **Parameter Estimation** - Automatic parameter count validation

#### Data & Configuration
- **MS MARCO Dataset** - Sentence-transformers/msmarco-bm25 (triplet-hard split)
- **VRAM Optimization** - batch_train=8, max_length=160 for Colab Pro A100 40GB
- **Configurable Losses** - Multi-objective training (semantic, anchor, bridge, fast, ranking, matryoshka, spectral)

#### Debugging & Stability
- **NaN/Inf Detection** - Automatic detection and `torch.nan_to_num` recovery
- **Component-wise Loss Debugging** - Detailed logs for each loss component
- **Model Output Validation** - Pre-loss forward pass validation
- **FP32 Mode** - Forced precision for numerical stability

### Changed 🔄

- **Default Model** - M456 (458M params) optimized for Colab Pro
- **Early Stopping** - Disabled by default (patience=9999) for full training
- **STS Evaluation** - Uses `embedding_semantic[768]` instead of ranking head
- **Embedding Keys** - Unified naming convention (embedding_semantic, embedding_ranking, etc.)
- **Health Checks** - Added 5000-step warmup to avoid false warnings

### Fixed 🐛

- **Device Detection** - Fixed NaN detection in nested dict outputs
- **Gradient Clipping** - Moved inside gradient accumulation loop for Accelerate compatibility
- **Loss NaN Handling** - Graceful recovery with fallback to nan_to_num
- **Memory Alignment** - Optimized for 40GB A100 (max_length=160, batch=8)
- **Import Consistency** - Updated all embedding key references across files

### Documentation 📚

- Created comprehensive **README.md** with Apache 2.0 license badge
- Added **ULTIMATE_GUIDE.md** with detailed usage examples
- Created **CONTRIBUTING.md** for open-source collaboration
- Added **CODE_OF_CONDUCT.md** (Contributor Covenant 2.0)
- Updated **CHANGELOG.md** (this file)
- Created **LICENSE** (Apache 2.0 full text)
- Created **.gitignore** for Python/ML projects
- Created **requirements.txt** with all dependencies

### Infrastructure 🏗️

- Added **.github/CONTRIBUTING.md** - Contribution guidelines for Spectra-NSA
- Added **.github/CODE_OF_CONDUCT.md** - Community standards
- Created **requirements.txt** - Dependency management
- Created **.gitignore** - Git exclusion rules
- Project renamed to **Spectra-NSA** for clarity
- Prepared for GitHub publication

### Known Limitations ⚠️

- Mixed precision (FP16) disabled for debugging
- Multi-GPU training not fully optimized
- Gradient checkpointing not implemented
- Resume training not implemented
- Single-node training only

## Roadmap 🚀

### v0.2.0 (Q1 2025)
- [ ] Enable FP16 training with stability improvements
- [ ] Implement gradient checkpointing
- [ ] Add multi-GPU distributed training
- [ ] Support resume training from checkpoint
- [ ] Additional evaluation benchmarks

### v0.3.0 (Q2 2025)
- [ ] Flash Attention integration
- [ ] Quantization support (INT8, INT4)
- [ ] TorchScript export
- [ ] ONNX export
- [ ] Hugging Face Hub integration

### v1.0.0 (Q3 2025)
- [ ] Production-grade stability
- [ ] Comprehensive benchmarking
- [ ] Paper publication
- [ ] Official model releases
- [ ] Community contributions integration

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under Apache License 2.0 - see [LICENSE](LICENSE) file.
