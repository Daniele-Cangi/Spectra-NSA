#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anomalous Embedding — ULTIMATE FINAL
------------------------------------
Combined: Your granular toggles + Scientific optimizations

FEATURES:
• M456 default (Colab Pro perfect fit)
• Granular component toggles (5 separate flags)
• Learnable fractal enrichment + spectral fusion
• Temperature annealing (cosine 0.07→0.05)
• Scientific loss balancing
• Integrated monitoring + early stopping
• Post-training auto-eval
• Parameter estimation display

SIZES:
  - M456  ≈ 350M (H=1024, L=24, A=16, spectral=192) ← DEFAULT
  - M600  ≈ 480M (H=1280, L=24, A=20, spectral=256)
  - M700  ≈ 580M (H=1536, L=24, A=24, spectral=384)
  - M1B   ≈ 850M (H=2048, L=32, A=32, spectral=512)

USAGE:
  # Full training (all components)
  python anomalous_embedding_ultimate.py --mode train --epochs 3
  
  # Ablation: disable specific components
  python anomalous_embedding_ultimate.py --no-spectral --no-anchor64 --mode train
  
  # Evaluation with Recall@K + STS
  python anomalous_embedding_ultimate.py --mode eval --checkpoint checkpoints/best_sts.pt
  
  # Different sizes
  python anomalous_embedding_ultimate.py --size M700 --mode train --epochs 3
"""

from __future__ import annotations
import os, sys, math, json, random, warnings, argparse, time, shutil
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from scipy.stats import spearmanr
from training_monitor import TrainingMonitor  # Optional: live monitoring during training

warnings.filterwarnings("ignore")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# Config with Granular Component Control
# ============================================================

@dataclass
class Config:
    # Model size
    size: str = "M456"

    # Tokenizer
    tokenizer_name: str = "bert-base-uncased"
    max_length: int = 192       # meno token → meno SxS in attention
    max_positions: int = 512
    vocab_size: int = 30522

    # Architecture
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    ff_mult: int = 4
    dropout: float = 0.1
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory savings

    # Spectral attention
    use_spectral: bool = True
    spectral_dim: int = 192
    fractal_depth: int = 1      # metà memoria nella parte frattale
    learnable_fractal: bool = True
    learnable_spectral_fusion: bool = True

    # Component toggles (granular control)
    enable_anchor64: bool = True
    enable_bridge: bool = True
    enable_matryoshka: bool = True  # Matryoshka loss
    enable_matry_angular: bool = True  # Angular alignment

    # Heads
    sem_dims: List[int] = None
    ent_dims: List[int] = None
    anchor_dim: int = 64
    retrieval_dim: int = 128
    ranking_dim: int = 1536
    pooling: str = "mean"

    # Temperature annealing
    temperature_start: float = 0.07
    temperature_end: float = 0.05
    temperature_warmup_ratio: float = 0.2

    # Loss weights (scientifically balanced)
    loss_weight_sem: Dict[int, float] = None
    loss_weight_anchor: float = 0.15
    loss_weight_bridge: float = 0.02
    loss_weight_fast: float = 0.25
    loss_weight_ranking: float = 1.0
    loss_weight_matryoshka: float = 0.08
    spectral_reg_weight: float = 0.005

    # Data
    dataset_name: str = "sentence-transformers/msmarco-bm25"
    dataset_config: str = "triplet-hard"
    split: str = "train"
    max_train_samples: int = 100000
    val_ratio: float = 0.02

    # Training
    batch_train: int = 8        # VRAM safe
    batch_eval: int = 32
    epochs: int = 3
    lr: float = 2e-4
    weight_decay: float = 0.05
    warmup_ratio: float = 0.08
    grad_accum: int = 8         # effective batch 8×8 = 64
    fp16: bool = False          # Disabilitato per debug NaN
    max_grad_norm: float = 1.0

    # Monitoring
    use_wandb: int = 0
    project_name: str = "anomalous-embeddings-ultimate"
    run_name: str = "ultimate-final"
    save_dir: str = "checkpoints"
    save_every: int = 500
    eval_every: int = 500
    early_stop_patience: int = 9999  # Di fatto disabilita early stopping: durata determinata da epochs
    early_stop_delta: float = 0.001

    # Seed
    seed: int = 42

    def __post_init__(self):
        if self.sem_dims is None:
            self.sem_dims = [768, 512, 256]
        if self.ent_dims is None:
            self.ent_dims = [384, 192, 96]
        if self.loss_weight_sem is None:
            self.loss_weight_sem = {768: 0.35, 512: 0.30, 256: 0.25}


def apply_preset(cfg: Config, size: str):
    """Apply size preset"""
    size = (size or "M456").upper()
    cfg.size = size
    
    if size == "M456":
        cfg.hidden_size = 1024
        cfg.num_layers = 24
        cfg.num_heads = 16
        cfg.ff_mult = 4
        cfg.spectral_dim = 192
        cfg.max_length = 160      # VRAM safe (era 256→192→160)
        cfg.batch_train = 8       # VRAM safe
        cfg.grad_accum = 8        # effective 64
        cfg.lr = 2e-4
        cfg.fractal_depth = 1     # riduci profondità frattale
        
    elif size == "M600":
        cfg.hidden_size = 1280
        cfg.num_layers = 24
        cfg.num_heads = 20
        cfg.ff_mult = 4
        cfg.spectral_dim = 256
        cfg.max_length = 320
        cfg.batch_train = 28
        cfg.grad_accum = 5
        cfg.lr = 1.8e-4
        
    elif size == "M700":
        cfg.hidden_size = 1536
        cfg.num_layers = 24
        cfg.num_heads = 24
        cfg.ff_mult = 4
        cfg.spectral_dim = 384
        cfg.max_length = 384
        cfg.batch_train = 24
        cfg.grad_accum = 6
        cfg.lr = 1.5e-4
        
    elif size == "M1B":
        cfg.hidden_size = 2048
        cfg.num_layers = 32
        cfg.num_heads = 32
        cfg.ff_mult = 4
        cfg.spectral_dim = 512
        cfg.max_length = 512
        cfg.batch_train = 16
        cfg.grad_accum = 8
        cfg.lr = 1e-4
        
    else:
        raise ValueError(f"Unknown size: {size}. Use M456|M600|M700|M1B")


def estimate_params(vocab_size: int, hidden: int, num_layers: int, spectral_dim: int, max_pos: int) -> int:
    """Estimate total parameters"""
    # Embeddings
    embeddings = vocab_size * hidden + max_pos * hidden
    
    # Encoder layers
    per_layer = 4 * hidden * hidden + 2 * hidden * (4 * hidden)  # Attention + FFN
    per_layer += hidden * spectral_dim * 3  # Spectral branch
    encoder = num_layers * per_layer
    
    # Heads (approximate)
    heads = (hidden * 768 +    # semantic
             hidden * 384 +    # entity
             hidden * 64 +     # anchor
             hidden * 128 +    # retrieval
             hidden * 1536)    # ranking
    
    total = embeddings + encoder + heads
    return total


# ============================================================
# Spectral-Entropy Attention (Learnable)
# ============================================================

class SpectralEntropyAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, spectral_dim: int, fractal_depth: int,
                 dropout: float, learnable_fractal: bool = True, learnable_fusion: bool = True):
        super().__init__()
        self.h = hidden
        self.nh = heads
        self.hd = hidden // heads
        self.fractal_depth = fractal_depth
        
        # Standard attention
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.o = nn.Linear(hidden, hidden)
        self.do = nn.Dropout(dropout)
        
        # Spectral branch
        self.sq = nn.Linear(hidden, spectral_dim)
        self.sk = nn.Linear(hidden, spectral_dim)
        self.sv = nn.Linear(hidden, spectral_dim)
        self.st = nn.Linear(spectral_dim, hidden)
        
        # Learnable parameters
        if learnable_fractal:
            self.fractal_weights = nn.Parameter(torch.full((fractal_depth,), 0.02))
        else:
            self.register_buffer('fractal_weights', torch.full((fractal_depth,), 0.02))
        
        if learnable_fusion:
            self.spectral_fusion_weight = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer('spectral_fusion_weight', torch.tensor(0.1))
        
        self.ln = nn.LayerNorm(hidden)

    def _mh(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        return x.view(B, S, self.nh, self.hd).transpose(1, 2)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        B, S, H = x.shape
        q = self._mh(self.q(x))
        k = self._mh(self.k(x))
        v = self._mh(self.v(x))
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        if attn_mask is not None:
            attn = attn + attn_mask
        
        # Fractal enrichment with learnable weights
        for d in range(self.fractal_depth):
            patt = attn @ attn.transpose(-2, -1)
            patt = F.normalize(patt, dim=-1)
            attn = attn + self.fractal_weights[d] * patt
        
        p = torch.softmax(attn, dim=-1)
        y = p @ v
        y = y.transpose(1, 2).contiguous().view(B, S, H)

        # Spectral path
        qs = self.sq(x)
        ks = self.sk(x)
        vs = self.sv(x)
        ss = (qs @ ks.transpose(-2, -1)) / math.sqrt(qs.size(-1))
        sp = torch.sigmoid(ss) ** 2
        sy = sp @ vs
        sy = self.st(sy)

        # Learnable fusion (clamped)
        fusion_weight = torch.clamp(self.spectral_fusion_weight, 0.0, 0.5)
        z = self.o(self.do(y)) + fusion_weight * sy
        
        return self.ln(z + x)


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, mult: int, dropout: float):
        super().__init__()
        inner = hidden * mult
        self.w1 = nn.Linear(hidden, inner)
        self.w2 = nn.Linear(hidden, inner)
        self.w3 = nn.Linear(inner, hidden)
        self.do = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden)
        
    def forward(self, x):
        a = self.w1(x)
        b = self.w2(x)
        y = self.w3(F.silu(a) * b)
        y = self.do(y)
        return self.ln(x + y)


class EncoderBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = SpectralEntropyAttention(
            cfg.hidden_size, cfg.num_heads, cfg.spectral_dim, cfg.fractal_depth,
            cfg.dropout, cfg.learnable_fractal, cfg.learnable_spectral_fusion
        )
        self.ffn = SwiGLU(cfg.hidden_size, cfg.ff_mult, cfg.dropout)
        
    def forward(self, x, attn_mask):
        x = self.attn(x, attn_mask)
        x = self.ffn(x)
        return x


class CustomEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_positions, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln = nn.LayerNorm(cfg.hidden_size)
        self.pooling = cfg.pooling
        self.gradient_checkpointing = cfg.gradient_checkpointing

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        h = self.tok_emb(input_ids) + self.pos_emb(pos)
        h = self.drop(h)

        attn_mask_add = None
        if attention_mask is not None:
            attn_mask_add = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0

        # Use gradient checkpointing if enabled and in training mode
        if self.gradient_checkpointing and self.training:
            for blk in self.layers:
                h = torch.utils.checkpoint.checkpoint(
                    blk,
                    h,
                    attn_mask_add,
                    use_reentrant=False
                )
        else:
            for blk in self.layers:
                h = blk(h, attn_mask_add)

        h = self.ln(h)

        if self.pooling == "cls":
            pooled = h[:, 0]
        else:
            m = attention_mask.unsqueeze(-1)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

        return {"last_hidden_state": h, "pooled": pooled}


# ============================================================
# Dual-Head + Ranking
# ============================================================

class MatryoshkaHead(nn.Module):
    def __init__(self, in_dim: int, dims: List[int]):
        super().__init__()
        dims = sorted(list(dims), reverse=True)
        self.dims = dims
        self.proj = nn.Linear(in_dim, dims[0])
        self.norm = nn.LayerNorm(dims[0])
        
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        z = self.norm(self.proj(x))
        z = F.normalize(z, p=2, dim=-1)
        return {d: F.normalize(z[..., :d], p=2, dim=-1) for d in self.dims}


class AnomalousProjection(nn.Module):
    def __init__(self, hidden_size: int, ranking_dim: int, spectral_dim: int, anomalous_dim: int = 1024):
        super().__init__()
        self.to_anom = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), nn.GELU(), nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, anomalous_dim), nn.GELU(), nn.LayerNorm(anomalous_dim)
        )
        self.to_spec = nn.Sequential(
            nn.Linear(anomalous_dim, spectral_dim), nn.Tanh(),
            nn.Linear(spectral_dim, spectral_dim), nn.LayerNorm(spectral_dim)
        )
        self.anom_basis = nn.Parameter(torch.randn(16, anomalous_dim) * 0.02)
        self.out = nn.Sequential(
            nn.Linear(anomalous_dim + spectral_dim, ranking_dim),
            nn.LayerNorm(ranking_dim)
        )
        
    def forward(self, last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        m = attn_mask.unsqueeze(-1)
        x = (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
        anom = self.to_anom(x)
        spec = self.to_spec(anom)
        sim = anom @ self.anom_basis.t()
        score = 1.0 - torch.sigmoid(sim.abs().max(dim=-1).values)
        fused = torch.cat([anom * score.unsqueeze(-1), spec], dim=-1)
        rank = F.normalize(self.out(fused), p=2, dim=-1)
        return {
            "embedding_ranking": rank,
            "anomalous": anom,
            "spectral": spec,
            "anomaly_score": score
        }


class FullEmbedder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = CustomEncoder(cfg)
        H = cfg.hidden_size
        
        # Always create core heads
        self.sem_head = MatryoshkaHead(H, cfg.sem_dims)
        self.ent_head = MatryoshkaHead(H, cfg.ent_dims)
        self.retr = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, cfg.retrieval_dim))
        self.rank_head = AnomalousProjection(H, cfg.ranking_dim, cfg.spectral_dim, anomalous_dim=max(1024, 2 * H))
        
        # Optional: anchor64
        if cfg.enable_anchor64:
            self.anchor = nn.Sequential(nn.Linear(H, cfg.anchor_dim), nn.LayerNorm(cfg.anchor_dim))

    def forward(self, tok: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self.backbone(tok["input_ids"], tok.get("attention_mask"))
        pooled = out["pooled"]
        last = out["last_hidden_state"]
        
        sem = self.sem_head(pooled)
        ent = self.ent_head(pooled)
        r128 = F.normalize(self.retr(pooled), p=2, dim=-1)
        rank = self.rank_head(last, tok["attention_mask"])
        
        result = {
            "embedding_semantic": sem,
            "entity": ent,
            "retrieval128": r128,
            **rank,
            "pooled": pooled,
            "last_hidden_state": last
        }
        
        if self.cfg.enable_anchor64:
            result["anchor64"] = F.normalize(self.anchor(pooled), p=2, dim=-1)
        
        return result


# ============================================================
# Losses with Granular Control
# ============================================================

def info_nce(a: torch.Tensor, p: torch.Tensor, T: float = 0.05) -> torch.Tensor:
    logits = (a @ p.t()) / T
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)


def matryoshka_loss(a_dict: Dict[int, torch.Tensor], p_dict: Dict[int, torch.Tensor],
                   weights: Dict[int, float], T: float) -> torch.Tensor:
    loss = 0.0
    for d, w in weights.items():
        loss += w * 0.5 * (info_nce(a_dict[d], p_dict[d], T) + info_nce(p_dict[d], a_dict[d], T))
    return loss


def anchor64_loss(a64: torch.Tensor, p64: torch.Tensor, T: float) -> torch.Tensor:
    return 0.5 * (info_nce(a64, p64, T) + info_nce(p64, a64, T))


def bridge_sem_entity(a_small: torch.Tensor, a_ent: torch.Tensor,
                     p_small: torch.Tensor, p_ent: torch.Tensor, lam: float) -> torch.Tensor:
    mse = nn.MSELoss()
    return lam * 0.5 * (mse(a_small, a_ent) + mse(p_small, p_ent))


def matryoshka_angular(a_small: torch.Tensor, p_small: torch.Tensor, n_small: torch.Tensor,
                       a_large: torch.Tensor, p_large: torch.Tensor, n_large: torch.Tensor) -> torch.Tensor:
    def norm(x): return F.normalize(x, p=2, dim=-1)
    a_s, p_s, n_s = map(norm, (a_small, p_small, n_small))
    a_l, p_l, n_l = map(norm, (a_large, p_large, n_large))
    sim_pos_s = (a_s * p_s).sum(-1)
    sim_neg_s = (a_s * n_s).sum(-1)
    sim_pos_l = (a_l * p_l).sum(-1)
    sim_neg_l = (a_l * n_l).sum(-1)
    return 0.5 * (F.mse_loss(sim_pos_s, sim_pos_l.detach()) + F.mse_loss(sim_neg_s, sim_neg_l.detach()))


class LossStack(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.triplet = nn.TripletMarginLoss(margin=0.3, p=2)
        self.current_temperature = cfg.temperature_start
        
    def set_temperature(self, temperature: float):
        self.current_temperature = temperature
        
    def forward(self, outA: Dict, outP: Dict, outN: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        T = self.current_temperature
        
        # Get device from a tensor (not a nested dict)
        device = outA["pooled"].device if "pooled" in outA else outA["retrieval128"].device
        
        # Semantic (always enabled if matryoshka enabled)
        L_sem = torch.tensor(0.0, device=device)
        if self.cfg.enable_matryoshka:
            L_sem = matryoshka_loss(outA["embedding_semantic"], outP["embedding_semantic"], self.cfg.loss_weight_sem, T)
        
        # Anchor (optional)
        L_anchor = torch.tensor(0.0, device=device)
        if self.cfg.enable_anchor64 and "anchor64" in outA:
            L_anchor = anchor64_loss(outA["anchor64"], outP["anchor64"], T)
        
        # Bridge (optional)
        L_bridge = torch.tensor(0.0, device=device)
        if self.cfg.enable_bridge:
            a96 = F.normalize(outA["embedding_semantic"][256][..., :96], p=2, dim=-1)
            p96 = F.normalize(outP["embedding_semantic"][256][..., :96], p=2, dim=-1)
            L_bridge = bridge_sem_entity(a96, outA["entity"][96], p96, outP["entity"][96], self.cfg.loss_weight_bridge)
        
        # Fast retrieval (always enabled)
        L_fast = 0.5 * (info_nce(outA["retrieval128"], outP["retrieval128"], T) +
                       info_nce(outP["retrieval128"], outA["retrieval128"], T))
        L_fast += self.triplet(outA["retrieval128"], outP["retrieval128"], outN["retrieval128"])
        
        # Ranking (always enabled)
        L_rank = 0.5 * (info_nce(outA["embedding_ranking"], outP["embedding_ranking"], T) +
                       info_nce(outP["embedding_ranking"], outA["embedding_ranking"], T))
        L_rank += self.triplet(outA["embedding_ranking"], outP["embedding_ranking"], outN["embedding_ranking"])
        
        # Matryoshka angular (optional)
        L_matry_angular = torch.tensor(0.0, device=device)
        if self.cfg.enable_matry_angular:
            L_matry_angular = matryoshka_angular(
                outA["retrieval128"], outP["retrieval128"], outN["retrieval128"],
                outA["embedding_ranking"], outP["embedding_ranking"], outN["embedding_ranking"]
            )
        
        # Spectral regularization (optional)
        L_spec = torch.tensor(0.0, device=device)
        if self.cfg.use_spectral and "spectral" in outA:
            def ent(x):
                probs = F.softmax(x, dim=-1)
                return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
            L_spec = -ent(outA["spectral"])

        # ===== DEBUG BLOCCO NaN/Inf SUI TERMINI DI LOSS =====
        components = {
            "L_sem": L_sem,
            "L_anchor": L_anchor,
            "L_bridge": L_bridge,
            "L_fast": L_fast,
            "L_rank": L_rank,
            "L_matry_angular": L_matry_angular,
            "L_spec": L_spec,
        }
        for name, val in components.items():
            if not torch.isfinite(val):
                try:
                    v = float(val.detach().cpu())
                except Exception:
                    v = "tensor"
                print(f"[NaN DEBUG] {name} non finito (NaN/Inf). Valore: {v}")

        # Total loss
        total = (L_sem +
                 self.cfg.loss_weight_anchor * L_anchor +
                 L_bridge +
                 self.cfg.loss_weight_fast * L_fast +
                 self.cfg.loss_weight_ranking * L_rank +
                 self.cfg.loss_weight_matryoshka * L_matry_angular +
                 self.cfg.spectral_reg_weight * L_spec)

        # Se anche il totale non è finito, log e ripulisci per non far saltare subito tutto
        if not torch.isfinite(total):
            print("[NaN DEBUG] total non finito (NaN/Inf). Applico torch.nan_to_num per continuare il training.")
            total = torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=-1e4)

        logs = {
            "L_sem": float(L_sem.detach().cpu()),
            "L_anchor": float(L_anchor.detach().cpu()),
            "L_bridge": float(L_bridge.detach().cpu()),
            "L_fast": float(L_fast.detach().cpu()),
            "L_rank": float(L_rank.detach().cpu()),
            "L_matry_angular": float(L_matry_angular.detach().cpu()),
            "L_spec": float(L_spec.detach().cpu()),
            "total": float(total.detach().cpu()),
            "temperature": T
        }
        
        return total, logs


# ============================================================
# Temperature Scheduler
# ============================================================

class TemperatureScheduler:
    def __init__(self, T_start: float, T_end: float, warmup_steps: int, total_steps: int):
        self.T_start = T_start
        self.T_end = T_end
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
    def get_temperature(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.T_start
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return self.T_end + 0.5 * (self.T_start - self.T_end) * (1 + math.cos(math.pi * progress))


# ============================================================
# Quick STS Evaluation
# ============================================================

@torch.no_grad()
def quick_sts_eval(model, tokenizer, device: torch.device, n_samples: int = 1000) -> float:
    """Quick STS-B evaluation"""
    try:
        from datasets import load_dataset
        model.eval()
        
        ds = load_dataset("mteb/stsbenchmark-sts", split="test")
        ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
        
        sent1 = [ex["sentence1"] for ex in ds]
        sent2 = [ex["sentence2"] for ex in ds]
        scores = [ex["score"] for ex in ds]
        
        batch_size = 32
        
        def encode_batch(texts):
            tok = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            tok = {k: v.to(device) for k, v in tok.items()}
            out = model(tok)
            # Usa embedding_semantic per STS (più rappresentativo per task semantici)
            sem = out["embedding_semantic"]
            if isinstance(sem, dict):  # Matryoshka: prendi dimensione massima
                max_dim = max(sem.keys())
                return sem[max_dim].cpu().numpy()
            return sem.cpu().numpy()
        
        emb1 = np.vstack([encode_batch(sent1[i:i+batch_size]) for i in range(0, len(sent1), batch_size)])
        emb2 = np.vstack([encode_batch(sent2[i:i+batch_size]) for i in range(0, len(sent2), batch_size)])
        
        cos_sim = (emb1 * emb2).sum(axis=1) / (
            np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1) + 1e-8
        )
        
        corr, _ = spearmanr(scores, cos_sim)
        model.train()
        
        return float(corr)
    except Exception as e:
        print(f"[MONITOR] Quick STS eval failed: {e}")
        return 0.0


# ============================================================
# Recall@K Evaluation
# ============================================================

@torch.no_grad()
def evaluate_recall_at_k(model: nn.Module, val_loader: DataLoader, device: torch.device, k: int = 10) -> float:
    """Evaluate Recall@K"""
    model.eval()
    correct = 0
    total = 0
    
    for A, P, _ in val_loader:
        A = {k: v.to(device) for k, v in A.items()}
        P = {k: v.to(device) for k, v in P.items()}
        outA = model(A)
        outP = model(P)
        a = outA["embedding_ranking"]
        p = outP["embedding_ranking"]
        S = a @ p.t()
        topk = S.topk(k, dim=1).indices
        labels = torch.arange(S.size(0), device=S.device).unsqueeze(1)
        correct += (topk == labels).any(dim=1).sum().item()
        total += S.size(0)
    
    model.train()
    return correct / max(1, total)


# ============================================================
# Data Loading
# ============================================================

class MSMARCOTripletDataset(Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        ex = self.ds[idx]
        return ex["anchor"], ex["positive"], ex["negative"]


def prepare_embedding_dataset(cfg: Config, seed: int = 42):
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split)
    
    def _map(batch):
        A, P, N = [], [], []
        # msmarco-bm25 / triplet-hard: colonne "query", "positive", "negative"
        for q, pos, neg in zip(batch["query"], batch["positive"], batch["negative"]):
            # qui non ci sono liste, sono già stringhe singole
            if q and pos and neg:
                A.append(q)
                P.append(pos)
                N.append(neg)
        return {"anchor": A, "positive": P, "negative": N}
    
    ds = ds.map(_map, batched=True, remove_columns=ds.column_names, desc="Build triplets")
    ds = ds.shuffle(seed=seed)
    
    n_total = len(ds)
    n_val = max(2000, int(cfg.val_ratio * n_total))
    n_train = min(cfg.max_train_samples, n_total - n_val)
    
    ds_train = ds.select(range(0, n_train))
    ds_val = ds.select(range(n_total - n_val, n_total))
    
    return ds_train, ds_val


def triplet_collate(batch, tokenizer, max_length):
    A, P, N = zip(*batch)
    tokA = tokenizer(list(A), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tokP = tokenizer(list(P), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tokN = tokenizer(list(N), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return tokA, tokP, tokN


def build_dataloaders(ds_train, ds_val, tokenizer, cfg: Config):
    train_dl = DataLoader(
        MSMARCOTripletDataset(ds_train),
        batch_size=cfg.batch_train,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: triplet_collate(b, tokenizer, cfg.max_length)
    )
    val_dl = DataLoader(
        MSMARCOTripletDataset(ds_val),
        batch_size=cfg.batch_eval,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: triplet_collate(b, tokenizer, cfg.max_length)
    )
    return train_dl, val_dl


# ============================================================
# Trainer with Monitoring
# ============================================================

class Trainer:
    def __init__(self, cfg: Config, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, tokenizer):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        
        # Multi-GPU distributed training support
        # Accelerator automatically handles:
        # - Multiple GPUs via DataParallel or DistributedDataParallel
        # - Mixed precision training
        # - Gradient accumulation
        self.acc = Accelerator(
            mixed_precision='fp16' if cfg.fp16 else 'no',
            gradient_accumulation_steps=cfg.grad_accum,
            # split_batches=True,  # Uncomment for very large batch sizes
            # dispatch_batches=False,  # Uncomment for debugging
        )
        
        self.opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        total_steps = len(train_loader) * cfg.epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        self.lr_sched = get_cosine_schedule_with_warmup(
            self.opt,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        temp_warmup = int(total_steps * cfg.temperature_warmup_ratio)
        self.temp_sched = TemperatureScheduler(
            cfg.temperature_start,
            cfg.temperature_end,
            temp_warmup,
            total_steps
        )
        
        self.loss_stack = LossStack(cfg)
        
        self.model, self.opt, self.train_loader, self.val_loader = self.acc.prepare(
            self.model, self.opt, self.train_loader, self.val_loader
        )
        
        self.global_step = 0
        self.current_epoch = 0
        self.best_sts = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.checkpoint_counter = 0  # Per backup ogni 6 checkpoint
        
        # Path per backup su Google Drive (se esiste)
        self.drive_backup_path = "/content/drive/MyDrive/anomalous_checkpoints"
        self.enable_drive_backup = os.path.exists("/content/drive/MyDrive")
        
        os.makedirs(cfg.save_dir, exist_ok=True)
        
        self.use_wandb = False
        if cfg.use_wandb and self.acc.is_main_process:
            try:
                import wandb
                wandb.init(
                    project=cfg.project_name,
                    name=cfg.run_name,
                    config=asdict(cfg),
                    mode=os.environ.get("WANDB_MODE", "online")
                )
                self.use_wandb = True
            except Exception:
                pass
        
        # Initialize training monitor for live monitoring
        self.monitor = TrainingMonitor(
            model=self.acc.unwrap_model(self.model),
            tokenizer=self.tokenizer,
            device=self.acc.device,
            model_size=self.cfg.size,
            eval_every=self.cfg.eval_every,
            early_stop_patience=self.cfg.early_stop_patience,
        )
        
        if self.acc.is_main_process:
            self.print_config()
    
    def print_config(self):
        print("\n" + "="*80)
        print("ANOMALOUS EMBEDDING - ULTIMATE FINAL")
        print("="*80)
        print(f"Model: {self.cfg.size} | H={self.cfg.hidden_size} L={self.cfg.num_layers} A={self.cfg.num_heads}")

        n_params = sum(p.numel() for p in self.model.parameters())
        est_params = estimate_params(self.cfg.vocab_size, self.cfg.hidden_size, self.cfg.num_layers,
                                     self.cfg.spectral_dim, self.cfg.max_positions)
        print(f"Params: {n_params/1e6:.1f}M (estimated: {est_params/1e6:.1f}M)")

        # Multi-GPU info
        num_processes = self.acc.num_processes
        device_info = f"Device: {self.acc.device}"
        if num_processes > 1:
            device_info += f" | Multi-GPU: {num_processes} GPUs (Distributed)"
        print(f"\n{device_info}")

        print(f"\nComponents:")
        print(f"  Spectral:          {'✓' if self.cfg.use_spectral else '✗'}")
        print(f"  Anchor64:          {'✓' if self.cfg.enable_anchor64 else '✗'}")
        print(f"  Bridge:            {'✓' if self.cfg.enable_bridge else '✗'}")
        print(f"  Matryoshka:        {'✓' if self.cfg.enable_matryoshka else '✗'}")
        print(f"  Angular Alignment: {'✓' if self.cfg.enable_matry_angular else '✗'}")
        print(f"  Gradient Checkpoint: {'✓' if self.cfg.gradient_checkpointing else '✗'}")

        print(f"\nTraining:")
        print(f"  Epochs: {self.cfg.epochs}")
        print(f"  Batch: {self.cfg.batch_train} x {self.cfg.grad_accum} = {self.cfg.batch_train * self.cfg.grad_accum}")
        if num_processes > 1:
            print(f"  Effective Batch (Multi-GPU): {self.cfg.batch_train * self.cfg.grad_accum * num_processes}")
        print(f"  LR: {self.cfg.lr}")
        print(f"  Temperature: {self.cfg.temperature_start} → {self.cfg.temperature_end}")
        print(f"  Mixed Precision: {'FP16' if self.cfg.fp16 else 'FP32'}")
        print("="*80 + "\n")
    
    def _step(self, batch):
        A, P, N = batch
        A = {k: v.to(self.acc.device) for k, v in A.items()}
        P = {k: v.to(self.acc.device) for k, v in P.items()}
        N = {k: v.to(self.acc.device) for k, v in N.items()}
        
        with self.acc.autocast():
            outA = self.model(A)
            outP = self.model(P)
            outN = self.model(N)
            
            # ===== DEBUG NaN SUBITO DOPO FORWARD =====
            def _check_out(name, out):
                for k, v in out.items():
                    if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                        print(f"[NaN DEBUG MODEL] {name}.{k} non finito (NaN/Inf). "
                              f"min={v.min().item() if torch.isfinite(v).any() else 'nan'} "
                              f"max={v.max().item() if torch.isfinite(v).any() else 'nan'}")
                    elif isinstance(v, dict):  # Per semantic/entity che sono dict di tensori
                        for dim, tensor in v.items():
                            if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                                print(f"[NaN DEBUG MODEL] {name}.{k}[{dim}] non finito (NaN/Inf). "
                                      f"min={tensor.min().item() if torch.isfinite(tensor).any() else 'nan'} "
                                      f"max={tensor.max().item() if torch.isfinite(tensor).any() else 'nan'}")
            
            _check_out("A", outA)
            _check_out("P", outP)
            _check_out("N", outN)
            # ========================================
            
            loss, logs = self.loss_stack(outA, outP, outN)
        
        return loss, logs
    
    def train(self):
        self.model.train()

        start_epoch = self.current_epoch
        for ep in range(start_epoch, self.cfg.epochs):
            self.current_epoch = ep
            epoch_loss = 0.0
            t0 = time.time()
            
            for it, batch in enumerate(self.train_loader):
                current_temp = self.temp_sched.get_temperature(self.global_step)
                self.loss_stack.set_temperature(current_temp)
                
                loss, logs = self._step(batch)
                self.acc.backward(loss)
                
                if (it + 1) % self.cfg.grad_accum == 0:
                    if self.cfg.max_grad_norm > 0:
                        self.acc.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.opt.step()
                    self.lr_sched.step()
                    self.opt.zero_grad()
                
                epoch_loss += float(loss.detach().cpu())
                self.global_step += 1
                
                if self.acc.is_main_process and (self.global_step % 100 == 0):
                    avg_loss = epoch_loss / (it + 1)
                    print(f"[Epoch {ep+1}/{self.cfg.epochs}] Step {self.global_step} | Loss: {avg_loss:.4f} | Temp: {current_temp:.4f}")
                    
                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/temperature": current_temp,
                            "train/lr": self.lr_sched.get_last_lr()[0],
                            **{f"train/{k}": v for k, v in logs.items()},
                            "step": self.global_step
                        })
                
                if self.acc.is_main_process and (self.global_step % self.cfg.eval_every == 0) and self.global_step > 0:
                    print(f"\n[MONITOR] Running quick STS evaluation at step {self.global_step}...")
                    
                    # Run monitor check to get health metrics
                    health_metrics = self.monitor.check_step(
                        step=self.global_step,
                        loss_logs=logs,
                        val_loss=None  # val_loss computed at end of epoch
                    )
                    
                    # Get STS score
                    sts_score = quick_sts_eval(
                        self.acc.unwrap_model(self.model),
                        self.tokenizer,
                        self.acc.device,
                        n_samples=200
                    )
                    
                    # Display progress table from monitor
                    self.monitor.print_progress_table(self.global_step, health_metrics)
                    
                    targets = {"M456": 0.825, "M600": 0.835, "M700": 0.840, "M1B": 0.850}
                    target_sts = targets.get(self.cfg.size, 0.840)
                    progress = (sts_score / target_sts) * 100
                    
                    print(f"[MONITOR] STS-B: {sts_score:.4f} | Target: {target_sts:.4f} | Progress: {progress:.1f}%")
                    
                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            "eval/sts_quick": sts_score,
                            "eval/progress_vs_sota": progress,
                            "step": self.global_step
                        })
                    
                    if sts_score > self.best_sts + self.cfg.early_stop_delta:
                        self.best_sts = sts_score
                        self.patience_counter = 0
                        print(f"[MONITOR] ✓ New best STS: {sts_score:.4f}")
                        self.save_ckpt("best_sts")
                    else:
                        self.patience_counter += 1
                        print(f"[MONITOR] No improvement ({self.patience_counter}/{self.cfg.early_stop_patience})")
                        
                        if self.patience_counter >= self.cfg.early_stop_patience:
                            print(f"[MONITOR] ⚠️  EARLY STOPPING TRIGGERED")
                            if self.monitor.should_stop_early():
                                print("[TRAINER] Early stopping confirmed by monitor")
                            return
                
                if self.acc.is_main_process and (self.global_step % self.cfg.save_every == 0):
                    self.save_ckpt(f"step-{self.global_step}")
            
            if self.acc.is_main_process:
                print(f"\n[Epoch {ep+1}] Completed in {time.time()-t0:.1f}s | Mean loss: {epoch_loss/max(1,len(self.train_loader)):.4f}")
            
            val_loss = self.validate()
            if self.acc.is_main_process:
                print(f"[Validation] Loss: {val_loss:.4f}\n")
                
                if self.use_wandb:
                    import wandb
                    wandb.log({"val/loss": val_loss, "epoch": ep + 1})
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_ckpt("best_val")
        
        if self.acc.is_main_process:
            self.save_ckpt("final")
            print("\n" + "="*80)
            print(f"TRAINING COMPLETE")
            print(f"Best STS: {self.best_sts:.4f}")
            print(f"Best Val Loss: {self.best_val_loss:.4f}")
            print("="*80 + "\n")
    
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        
        for batch in self.val_loader:
            loss, _ = self._step(batch)
            total_loss += float(loss.detach().cpu())
        
        self.model.train()
        return total_loss / max(1, len(self.val_loader))
    
    def save_ckpt(self, name: str):
        if not self.acc.is_main_process:
            return

        path = os.path.join(self.cfg.save_dir, f"{name}.pt")
        payload = {
            "model": self.acc.unwrap_model(self.model).state_dict(),
            "opt": self.opt.state_dict(),
            "lr_sched": self.lr_sched.state_dict(),
            "cfg": asdict(self.cfg),
            "step": self.global_step,
            "best_sts": self.best_sts,
            "best_val_loss": self.best_val_loss,
            "epoch": self.current_epoch,
            "checkpoint_counter": self.checkpoint_counter,
            "patience_counter": self.patience_counter,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            }
        }
        torch.save(payload, path)
        print(f"[CHECKPOINT] Saved → {path}")

        # Backup automatico su Google Drive ogni 6 checkpoint
        self.checkpoint_counter += 1
        if self.enable_drive_backup and self.checkpoint_counter % 6 == 0:
            try:
                os.makedirs(self.drive_backup_path, exist_ok=True)
                backup_path = os.path.join(self.drive_backup_path, f"{name}.pt")
                shutil.copy2(path, backup_path)
                print(f"[DRIVE BACKUP] Checkpoint #{self.checkpoint_counter} → {backup_path}")
            except Exception as e:
                print(f"[DRIVE BACKUP] Failed: {e}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training state"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"[RESUME] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.acc.device, weights_only=False)

        # Load model state
        self.acc.unwrap_model(self.model).load_state_dict(ckpt["model"])
        print(f"[RESUME] ✓ Model state loaded")

        # Load optimizer and scheduler
        self.opt.load_state_dict(ckpt["opt"])
        self.lr_sched.load_state_dict(ckpt["lr_sched"])
        print(f"[RESUME] ✓ Optimizer and scheduler loaded")

        # Load training state
        self.global_step = ckpt.get("step", 0)
        self.best_sts = ckpt.get("best_sts", 0.0)
        self.best_val_loss = ckpt.get("best_val_loss", float('inf'))
        self.current_epoch = ckpt.get("epoch", 0)
        self.checkpoint_counter = ckpt.get("checkpoint_counter", 0)
        self.patience_counter = ckpt.get("patience_counter", 0)
        print(f"[RESUME] ✓ Training state: step={self.global_step}, epoch={self.current_epoch}, best_sts={self.best_sts:.4f}")

        # Restore RNG states for reproducibility
        if "rng_state" in ckpt:
            try:
                random.setstate(ckpt["rng_state"]["python"])
                np.random.set_state(ckpt["rng_state"]["numpy"])
                torch.set_rng_state(ckpt["rng_state"]["torch"])
                if torch.cuda.is_available() and ckpt["rng_state"]["torch_cuda"] is not None:
                    torch.cuda.set_rng_state_all(ckpt["rng_state"]["torch_cuda"])
                print(f"[RESUME] ✓ RNG states restored")
            except Exception as e:
                print(f"[RESUME] ⚠️  Could not restore RNG states: {e}")

        print(f"[RESUME] Resume training from epoch {self.current_epoch+1}, step {self.global_step}\n")


# ============================================================
# Inference Helpers
# ============================================================

@torch.no_grad()
def embed_semantic(texts: List[str], tokenizer, model: nn.Module, dim: int = 256,
                  max_length: int = 256, device: str = "cuda"):
    tok = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}
    out = model(tok)
    return out["semantic"][dim].detach().cpu().numpy()


@torch.no_grad()
def embed_anchor64(texts: List[str], tokenizer, model: nn.Module,
                   max_length: int = 256, device: str = "cuda"):
    tok = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}
    out = model(tok)
    return out.get("anchor64").detach().cpu().numpy() if "anchor64" in out else None


# ============================================================
# Main (CLI with Granular Toggles)
# ============================================================

def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Anomalous Embedding - Ultimate Final")
    p.add_argument("--size", choices=["M456", "M600", "M700", "M1B"], default="M456")
    p.add_argument("--mode", choices=["train", "eval", "extract", "param_estimate"], default="train")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--use_wandb", type=int, default=None)
    p.add_argument("--texts", nargs="*", default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint path")
    p.add_argument("--seed", type=int, default=None)

    # Memory optimization
    p.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing for memory savings")

    # Granular component toggles
    p.add_argument("--no-spectral", action="store_true", help="Disable spectral attention")
    p.add_argument("--no-anchor64", action="store_true", help="Disable anchor64 head")
    p.add_argument("--no-bridge", action="store_true", help="Disable bridge loss")
    p.add_argument("--no-matry", action="store_true", help="Disable matryoshka loss")
    p.add_argument("--no-angular", action="store_true", help="Disable angular alignment")
    
    args = p.parse_args(argv)
    
    # Setup config
    cfg = Config()
    apply_preset(cfg, args.size)
    set_seed(args.seed or cfg.seed)
    
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.use_wandb is not None:
        cfg.use_wandb = args.use_wandb
    
    # Apply toggles
    if args.no_spectral:
        cfg.use_spectral = False
    if args.no_anchor64:
        cfg.enable_anchor64 = False
    if args.no_bridge:
        cfg.enable_bridge = False
    if args.no_matry:
        cfg.enable_matryoshka = False
    if args.no_angular:
        cfg.enable_matry_angular = False

    # Apply memory optimizations
    if args.gradient_checkpointing:
        cfg.gradient_checkpointing = True
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
    cfg.vocab_size = tokenizer.vocab_size
    
    # Parameter estimation
    est = estimate_params(cfg.vocab_size, cfg.hidden_size, cfg.num_layers, cfg.spectral_dim, cfg.max_positions)
    print(f"[cfg] size={args.size} H={cfg.hidden_size} L={cfg.num_layers} A={cfg.num_heads} S={cfg.spectral_dim} | est params ≈ {est/1e6:.1f}M")
    print(f"[cfg] toggles: spectral={cfg.use_spectral}, anchor64={cfg.enable_anchor64}, bridge={cfg.enable_bridge}, matry={cfg.enable_matryoshka}, angular={cfg.enable_matry_angular}")
    
    # Param estimate mode (just show config and exit)
    if args.mode == "param_estimate":
        print(f"\n{'='*80}")
        print(f"PARAMETER ESTIMATION - {args.size}")
        print(f"{'='*80}")
        print(f"Hidden Size:      {cfg.hidden_size}")
        print(f"Num Layers:       {cfg.num_layers}")
        print(f"Num Heads:        {cfg.num_heads}")
        print(f"Spectral Dim:     {cfg.spectral_dim}")
        print(f"Max Positions:    {cfg.max_positions}")
        print(f"Vocab Size:       {cfg.vocab_size}")
        print(f"\nEstimated Params: {est/1e6:.1f}M ({est:,})")
        print(f"{'='*80}\n")
        return
    
    # Model
    model = FullEmbedder(cfg)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"[LOAD] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[LOAD] Loaded from step {ckpt.get('step', 0)}")
    
    model.to(dev)
    
    # Extract mode
    if args.mode == "extract":
        texts = args.texts or ["Quantum mechanics reveals hidden patterns.", "Simple sentence."]
        emb = embed_semantic(texts, tokenizer, model, dim=256, device=str(dev))
        print(f"Embeddings shape: {emb.shape}")
        return
    
    # Load data
    print(f"Loading dataset {cfg.dataset_name} / {cfg.dataset_config}…")
    ds_train, ds_val = prepare_embedding_dataset(cfg, seed=cfg.seed)
    train_loader, val_loader = build_dataloaders(ds_train, ds_val, tokenizer, cfg)
    
    # Eval mode
    if args.mode == "eval":
        print("[EVAL] Running evaluation...")
        r10 = evaluate_recall_at_k(model, val_loader, dev, k=10)
        sts = quick_sts_eval(model, tokenizer, dev, n_samples=1000)
        print(f"Recall@10: {r10:.3f} | STS (Spearman): {sts:.3f}")
        return
    
    # Training mode
    print("Starting training…")
    trainer = Trainer(cfg, model, train_loader, val_loader, tokenizer)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()
    
    # Post-training evaluation
    print("\n[POST-TRAIN] Running final evaluation...")
    sts = quick_sts_eval(model, tokenizer, dev, n_samples=1000)
    print(f"[POST] STS (Spearman): {sts:.3f}")


if __name__ == "__main__":
    main()
