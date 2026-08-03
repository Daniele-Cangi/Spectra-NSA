from dataclasses import dataclass
from typing import Literal, Tuple

MixerMode = Literal["semantic", "spectral", "dual"]


@dataclass(frozen=True)
class SpectraConfig:
    vocab_size: int
    hidden_size: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_multiplier: int = 4
    max_positions: int = 512
    dropout: float = 0.0
    mixer_mode: MixerMode = "dual"
    matryoshka_dims: Tuple[int, ...] = (128, 64, 32)

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")
        if not self.matryoshka_dims:
            raise ValueError("matryoshka_dims cannot be empty")
        if max(self.matryoshka_dims) > self.hidden_size:
            raise ValueError("matryoshka_dims cannot exceed hidden_size")
        if self.mixer_mode not in {"semantic", "spectral", "dual"}:
            raise ValueError(f"unsupported mixer_mode: {self.mixer_mode}")
