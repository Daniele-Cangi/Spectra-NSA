from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SpectraConfig


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class SemanticMixer(nn.Module):
    def __init__(self, cfg: SpectraConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.attn = nn.MultiheadAttention(
            cfg.hidden_size,
            cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        mixed, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=~mask,
            need_weights=False,
        )
        return x + self.dropout(mixed)


class FourierMixer(nn.Module):
    """Frequency-domain token mixing over each sample's valid prefix.

    Per-sample slicing prevents padded tokens from contaminating valid positions.
    This reference implementation prioritizes correctness and testability over
    throughput. A future vectorized implementation must preserve this contract.
    """

    def __init__(self, cfg: SpectraConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(cfg.hidden_size)
        freq_bins = cfg.max_positions // 2 + 1
        self.real_filter = nn.Parameter(torch.ones(freq_bins, cfg.hidden_size))
        self.imag_filter = nn.Parameter(torch.zeros(freq_bins, cfg.hidden_size))
        self.output = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        result = torch.zeros_like(normalized)

        for batch_idx in range(x.size(0)):
            length = int(mask[batch_idx].sum().item())
            if length == 0:
                continue
            valid = normalized[batch_idx, :length]
            spectrum = torch.fft.rfft(valid, dim=0, norm="ortho")
            bins = spectrum.size(0)
            learned_filter = torch.complex(
                self.real_filter[:bins],
                self.imag_filter[:bins],
            )
            mixed = torch.fft.irfft(
                spectrum * learned_filter,
                n=length,
                dim=0,
                norm="ortho",
            )
            result[batch_idx, :length] = mixed

        result = self.output(result)
        result = result * mask.unsqueeze(-1).to(dtype=result.dtype)
        return x + self.dropout(result)


class FeedForward(nn.Module):
    def __init__(self, cfg: SpectraConfig) -> None:
        super().__init__()
        inner = cfg.hidden_size * cfg.ff_multiplier
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.net = nn.Sequential(
            nn.Linear(cfg.hidden_size, inner * 2),
            nn.GLU(dim=-1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(inner, cfg.hidden_size),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class SpectraBlock(nn.Module):
    def __init__(self, cfg: SpectraConfig) -> None:
        super().__init__()
        self.mode = cfg.mixer_mode
        self.semantic = SemanticMixer(cfg) if self.mode in {"semantic", "dual"} else None
        self.spectral = FourierMixer(cfg) if self.mode in {"spectral", "dual"} else None
        self.gate = (
            nn.Sequential(
                nn.LayerNorm(cfg.hidden_size * 2),
                nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
                nn.GELU(),
                nn.Linear(cfg.hidden_size, 1),
            )
            if self.mode == "dual"
            else None
        )
        self.ffn = FeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        disagreement = None

        if self.mode == "semantic":
            x = self.semantic(x, mask)
        elif self.mode == "spectral":
            x = self.spectral(x, mask)
        else:
            semantic = self.semantic(x, mask)
            spectral = self.spectral(x, mask)
            pooled_semantic = masked_mean(semantic, mask)
            pooled_spectral = masked_mean(spectral, mask)
            disagreement = 1.0 - F.cosine_similarity(
                pooled_semantic,
                pooled_spectral,
                dim=-1,
            )
            gate = torch.sigmoid(self.gate(torch.cat([semantic, spectral], dim=-1)))
            x = gate * semantic + (1.0 - gate) * spectral

        x = self.ffn(x)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        return x, disagreement


class MatryoshkaHead(nn.Module):
    def __init__(self, cfg: SpectraConfig) -> None:
        super().__init__()
        self.dims = tuple(sorted(cfg.matryoshka_dims, reverse=True))
        self.projection = nn.Linear(cfg.hidden_size, self.dims[0])
        self.norm = nn.LayerNorm(self.dims[0])

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        full = self.norm(self.projection(x))
        return {
            dim: F.normalize(full[..., :dim], p=2, dim=-1)
            for dim in self.dims
        }


class SpectraEmbeddingModel(nn.Module):
    def __init__(self, cfg: SpectraConfig) -> None:
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.max_positions, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([SpectraBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.hidden_size)
        self.head = MatryoshkaHead(cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, object]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if input_ids.size(1) > self.config.max_positions:
            raise ValueError("sequence exceeds max_positions")

        mask = attention_mask.to(dtype=torch.bool)
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        disagreements = []
        for block in self.blocks:
            x, disagreement = block(x, mask)
            if disagreement is not None:
                disagreements.append(disagreement)

        x = self.final_norm(x)
        pooled = masked_mean(x, mask)
        embeddings = self.head(pooled)
        largest_dim = max(embeddings)

        diagnostics: Dict[str, torch.Tensor] = {}
        if disagreements:
            diagnostics["branch_disagreement"] = torch.stack(disagreements, dim=0).mean(dim=0)

        return {
            "embedding": embeddings[largest_dim],
            "matryoshka": embeddings,
            "diagnostics": diagnostics,
            "last_hidden_state": x,
        }
