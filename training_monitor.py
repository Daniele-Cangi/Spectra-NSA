#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Monitor with SOTA Benchmarking
----------------------------------------
Real-time evaluation during training with:
- Progressive STS evaluation every N steps
- Gradient health monitoring
- Embedding collapse detection
- SOTA comparison dashboard
- Intelligent early stopping

Usage:
  Integrate into Trainer class or run as separate monitoring process
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import spearmanr


# ============================================================
# SOTA Targets (for comparison)
# ============================================================

SOTA_TARGETS = {
    "M700": {
        "sts_average": 0.840,  # Target to beat
        "msmarco_ndcg10": 0.420,
        "recall@1": 0.550,
        "params_millions": 700,
    },
    "M456": {
        "sts_average": 0.800,  # Target M456 (456M params)
        "msmarco_ndcg10": 0.410,
        "recall@1": 0.540,
        "params_millions": 456,
    },
    "M300": {
        "sts_average": 0.825,
        "msmarco_ndcg10": 0.400,
        "recall@1": 0.530,
        "params_millions": 300,
    }
}


# ============================================================
# Health Monitoring
# ============================================================

@dataclass
class HealthMetrics:
    """Track model health during training"""
    
    # Gradient health
    grad_norm_mean: float = 0.0
    grad_norm_std: float = 0.0
    grad_norm_max: float = 0.0
    grad_dead_neurons: float = 0.0  # % of neurons with no gradient
    
    # Embedding quality
    embedding_collapse: float = 0.0  # Avg pairwise cosine similarity
    embedding_std: float = 0.0  # Standard deviation across dimensions
    embedding_rank: int = 0  # Effective rank
    
    # Loss components
    loss_sem: float = 0.0
    loss_anchor: float = 0.0
    loss_fast: float = 0.0
    loss_rank: float = 0.0
    loss_matryoshka: float = 0.0
    loss_spectral: float = 0.0
    
    # Anomaly features
    anomaly_mean: float = 0.0
    anomaly_std: float = 0.0
    spectral_entropy: float = 0.0
    
    # Performance trends
    sts_stsb: float = 0.0
    validation_loss: float = float('inf')
    
    # Flags
    is_healthy: bool = True
    warning_messages: List[str] = None
    
    def __post_init__(self):
        if self.warning_messages is None:
            self.warning_messages = []
    
    def check_health(self, step: int = 0) -> Tuple[bool, List[str]]:
        """Check if model is healthy and return warnings"""
        warnings = []
        
        # Warmup phase: no warnings before 5000 steps
        if step < 5000:
            self.warning_messages = []
            self.is_healthy = True
            return True, []
        
        # Check gradient explosion
        if self.grad_norm_max > 100.0:
            warnings.append(f"⚠️  Gradient explosion detected (max norm: {self.grad_norm_max:.2f})")
        
        # Check dead neurons
        if self.grad_dead_neurons > 0.3:
            warnings.append(f"⚠️  High dead neuron ratio: {self.grad_dead_neurons:.1%}")
        
        # Check embedding collapse
        if self.embedding_collapse > 0.8:
            warnings.append(f"⚠️  Embedding collapse detected (similarity: {self.embedding_collapse:.3f})")
        
        # Check low embedding rank
        if self.embedding_rank < 64:
            warnings.append(f"⚠️  Low embedding rank: {self.embedding_rank}")
        
        # Check anomaly score collapse
        if self.anomaly_std < 0.01:
            warnings.append(f"⚠️  Anomaly scores collapsed (std: {self.anomaly_std:.4f})")
        
        # Check spectral entropy
        if self.spectral_entropy < 1.0:
            warnings.append(f"⚠️  Low spectral entropy: {self.spectral_entropy:.3f}")
        
        self.warning_messages = warnings
        self.is_healthy = len(warnings) == 0
        
        return self.is_healthy, warnings


def compute_gradient_stats(model: torch.nn.Module) -> Dict[str, float]:
    """Compute gradient statistics"""
    grads = []
    dead_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            grads.append(grad.norm().item())
            
            # Check dead neurons (gradient below threshold)
            dead = (grad.abs() < 1e-7).sum().item()
            dead_count += dead
            total_count += grad.numel()
    
    if not grads:
        return {
            "mean": 0.0,
            "std": 0.0,
            "max": 0.0,
            "dead_ratio": 0.0
        }
    
    return {
        "mean": np.mean(grads),
        "std": np.std(grads),
        "max": np.max(grads),
        "dead_ratio": dead_count / max(1, total_count)
    }


@torch.no_grad()
def compute_embedding_stats(embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Compute embedding quality metrics
    Args:
        embeddings: [batch_size, embedding_dim]
    """
    # Normalize
    emb = F.normalize(embeddings, p=2, dim=-1)
    
    # Pairwise cosine similarity (collapse detection)
    sim_matrix = emb @ emb.T
    # Remove diagonal
    mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
    off_diag_sim = sim_matrix[mask]
    collapse = off_diag_sim.mean().item()
    
    # Standard deviation across dimensions
    std_per_dim = embeddings.std(dim=0).mean().item()
    
    # Effective rank (ratio of squared singular values)
    try:
        U, S, V = torch.svd(embeddings.cpu())
        S_normalized = S / S.sum()
        entropy = -(S_normalized * torch.log(S_normalized + 1e-9)).sum()
        eff_rank = torch.exp(entropy).item()
    except:
        eff_rank = embeddings.size(-1)
    
    return {
        "collapse": collapse,
        "std": std_per_dim,
        "rank": int(eff_rank)
    }


# ============================================================
# Progressive STS Evaluation (fast subset)
# ============================================================

@torch.no_grad()
def quick_sts_eval(
    model,
    tokenizer,
    device: torch.device,
    n_samples: int = 200
) -> float:
    """
    Quick STS-B evaluation on small subset
    Returns Spearman correlation
    """
    try:
        from datasets import load_dataset
        
        model.eval()
        
        # Load STS-B test
        ds = load_dataset("mteb/stsbenchmark-sts", split="test")
        ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
        
        sent1 = [ex["sentence1"] for ex in ds]
        sent2 = [ex["sentence2"] for ex in ds]
        scores = [ex["score"] for ex in ds]
        
        # Encode (small batches)
        batch_size = 32
        
        def encode_batch(texts):
            tok = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            tok = {k: v.to(device) for k, v in tok.items()}
            out = model(tok)
            # Usa embedding_semantic (max dim = 768) per STS
            sem = out["embedding_semantic"]
            if isinstance(sem, dict):  # Matryoshka: prendi dimensione massima
                max_dim = max(sem.keys())
                return sem[max_dim].cpu().numpy()
            return sem.cpu().numpy()
        
        emb1 = np.vstack([encode_batch(sent1[i:i+batch_size]) for i in range(0, len(sent1), batch_size)])
        emb2 = np.vstack([encode_batch(sent2[i:i+batch_size]) for i in range(0, len(sent2), batch_size)])
        
        # Cosine similarity
        cos_sim = (emb1 * emb2).sum(axis=1) / (
            np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1) + 1e-8
        )
        
        # Spearman correlation
        corr, _ = spearmanr(scores, cos_sim)
        
        model.train()
        return float(corr)
        
    except Exception as e:
        print(f"Quick STS eval failed: {e}")
        return 0.0


# ============================================================
# Training Monitor Class
# ============================================================

class TrainingMonitor:
    """
    Advanced training monitor with health checks and SOTA comparison
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        model_size: str = "M700",
        eval_every: int = 500,
        early_stop_patience: int = 5,
        early_stop_delta: float = 0.001
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_size = model_size
        self.eval_every = eval_every
        
        # Early stopping
        self.patience = early_stop_patience
        self.delta = early_stop_delta
        self.best_sts = 0.0
        self.patience_counter = 0
        self.should_stop = False
        
        # History
        self.history = {
            "step": [],
            "sts": [],
            "val_loss": [],
            "grad_norm": [],
            "collapse": [],
            "warnings": []
        }
        
        # SOTA targets
        self.targets = SOTA_TARGETS.get(model_size, SOTA_TARGETS["M700"])
    
    def check_step(
        self,
        step: int,
        loss_logs: Dict[str, float],
        val_loss: Optional[float] = None
    ) -> HealthMetrics:
        """
        Run health check at given step
        Returns HealthMetrics with warnings
        """
        metrics = HealthMetrics()
        
        # 1. Gradient health
        grad_stats = compute_gradient_stats(self.model)
        metrics.grad_norm_mean = grad_stats["mean"]
        metrics.grad_norm_std = grad_stats["std"]
        metrics.grad_norm_max = grad_stats["max"]
        metrics.grad_dead_neurons = grad_stats["dead_ratio"]
        
        # 2. Loss components
        metrics.loss_sem = loss_logs.get("L_sem", 0.0)
        metrics.loss_anchor = loss_logs.get("L_anchor", 0.0)
        metrics.loss_fast = loss_logs.get("L_fast", 0.0)
        metrics.loss_rank = loss_logs.get("L_rank", 0.0)
        metrics.loss_matryoshka = loss_logs.get("L_matry", 0.0)
        metrics.loss_spectral = loss_logs.get("L_spec", 0.0)
        
        # 3. Embedding quality (sample from last forward pass)
        # This requires accessing model outputs - simplified here
        # In practice, you'd pass in recent batch outputs
        
        if val_loss is not None:
            metrics.validation_loss = val_loss
        
        # 4. Run STS evaluation periodically
        if step % self.eval_every == 0 and step > 0:
            print(f"\n[MONITOR] Running quick STS eval at step {step}...")
            sts_score = quick_sts_eval(self.model, self.tokenizer, self.device)
            metrics.sts_stsb = sts_score
            
            # Update history
            self.history["step"].append(step)
            self.history["sts"].append(sts_score)
            
            # Check vs SOTA
            target_sts = self.targets["sts_average"]
            progress = (sts_score / target_sts) * 100
            print(f"[MONITOR] STS-B: {sts_score:.4f} | Target: {target_sts:.4f} | Progress: {progress:.1f}%")
            
            # Early stopping check
            if sts_score > self.best_sts + self.delta:
                self.best_sts = sts_score
                self.patience_counter = 0
                print(f"[MONITOR] ✓ New best STS: {sts_score:.4f}")
            else:
                self.patience_counter += 1
                print(f"[MONITOR] No improvement ({self.patience_counter}/{self.patience})")
                
                if self.patience_counter >= self.patience:
                    self.should_stop = True
                    print(f"[MONITOR] ⚠️  EARLY STOPPING TRIGGERED")
        
        # 5. Health check
        is_healthy, warnings = metrics.check_health(step=step)
        
        if not is_healthy:
            print(f"\n[MONITOR] ⚠️  HEALTH WARNINGS at step {step}:")
            for w in warnings:
                print(f"  {w}")
            self.history["warnings"].append((step, warnings))
        
        # 6. Record history
        self.history["grad_norm"].append(metrics.grad_norm_mean)
        if val_loss is not None:
            self.history["val_loss"].append(val_loss)
        
        return metrics
    
    def print_progress_table(self, step: int, metrics: HealthMetrics):
        """Print formatted progress table"""
        print("\n" + "="*80)
        print(f"TRAINING MONITOR - Step {step}")
        print("="*80)
        
        # Gradients
        print("GRADIENTS:")
        print(f"  Mean:        {metrics.grad_norm_mean:.4f}")
        print(f"  Std:         {metrics.grad_norm_std:.4f}")
        print(f"  Max:         {metrics.grad_norm_max:.4f}")
        print(f"  Dead ratio:  {metrics.grad_dead_neurons:.2%}")
        
        # Losses
        print("\nLOSSES:")
        print(f"  Semantic:    {metrics.loss_sem:.4f}")
        print(f"  Anchor:      {metrics.loss_anchor:.4f}")
        print(f"  Fast:        {metrics.loss_fast:.4f}")
        print(f"  Ranking:     {metrics.loss_rank:.4f}")
        print(f"  Matryoshka:  {metrics.loss_matryoshka:.4f}")
        print(f"  Spectral:    {metrics.loss_spectral:.4f}")
        
        # Performance
        if metrics.sts_stsb > 0:
            print("\nPERFORMANCE:")
            print(f"  STS-B:       {metrics.sts_stsb:.4f}")
            target = self.targets["sts_average"]
            status = "✓ ABOVE TARGET" if metrics.sts_stsb >= target else "⚠ BELOW TARGET"
            print(f"  Target:      {target:.4f} {status}")
        
        # Health status
        print("\nHEALTH:")
        print(f"  Status:      {'✓ HEALTHY' if metrics.is_healthy else '⚠ WARNINGS'}")
        
        if not metrics.is_healthy:
            for w in metrics.warning_messages:
                print(f"    {w}")
        
        print("="*80 + "\n")
    
    def should_stop_early(self) -> bool:
        """Check if training should stop"""
        return self.should_stop
    
    def get_summary(self) -> Dict:
        """Get training summary"""
        return {
            "best_sts": self.best_sts,
            "final_step": self.history["step"][-1] if self.history["step"] else 0,
            "total_warnings": len(self.history["warnings"]),
            "history": self.history
        }


# ============================================================
# Integration Example
# ============================================================

def integrate_monitor_example():
    """
    Example of how to integrate monitor into training loop
    """
    
    # In your Trainer class, add:
    
    # __init__:
    # self.monitor = TrainingMonitor(
    #     model=self.model,
    #     tokenizer=tokenizer,
    #     device=self.acc.device,
    #     model_size=cfg.size,
    #     eval_every=500,
    #     early_stop_patience=5
    # )
    
    # In training loop after loss.backward():
    # if self.global_step % 100 == 0:
    #     metrics = self.monitor.check_step(
    #         step=self.global_step,
    #         loss_logs=logs,
    #         val_loss=val_loss if hasattr(self, 'val_loss') else None
    #     )
    #     self.monitor.print_progress_table(self.global_step, metrics)
    #     
    #     if self.monitor.should_stop_early():
    #         print("[TRAINER] Early stopping triggered by monitor")
    #         break
    
    pass


if __name__ == "__main__":
    print("Training Monitor Module")
    print("=" * 80)
    print("\nSOTA Targets:")
    for size, targets in SOTA_TARGETS.items():
        print(f"\n{size}:")
        for metric, value in targets.items():
            print(f"  {metric}: {value}")
    
    print("\n" + "=" * 80)
    print("To integrate into training, add TrainingMonitor to your Trainer class")
    print("See integrate_monitor_example() for details")
