"""
Synthetic 1D Helmholtz (Poisson) dataset for Transolver++ demo.

PDE: u''(x) + (a*pi)^2 u(x) = 0 on [0,1], with Dirichlet BC u(0)=u(1)=0
Analytic family: u(x) = sin(a*pi*x), a in {1,2,...}

Each sample uses a fixed integer frequency a sampled per-sample, and returns:
- coords: (N, 1) uniform grid on [0,1]
- cond: (1,) the frequency a (as float)
- target: (N, 1) = sin(a*pi*x)
- bc_mask: (N,) boolean mask for boundary points (x=0 or x=1)

This dataset is small and meant for demonstrating the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import math
import torch
from torch.utils.data import Dataset


@dataclass
class SineHelmholtz1DConfig:
    num_samples: int = 1024
    n_points: int = 128
    freq_min: int = 1
    freq_max: int = 5
    dtype: torch.dtype = torch.float32


class SineHelmholtz1DDataset(Dataset):
    def __init__(self, cfg: SineHelmholtz1DConfig) -> None:
        super().__init__()
        assert cfg.freq_max >= cfg.freq_min >= 1
        assert cfg.n_points >= 3, "Need at least 3 points for finite differences"
        self.cfg = cfg
        # Pre-generate frequencies per sample for reproducibility
        self.freqs = torch.randint(cfg.freq_min, cfg.freq_max + 1, (cfg.num_samples,), dtype=torch.int64)
        # Fixed uniform grid and BC mask
        N = cfg.n_points
        x = torch.linspace(0.0, 1.0, steps=N, dtype=cfg.dtype).unsqueeze(-1)  # (N,1)
        bc_mask = torch.zeros(N, dtype=torch.bool)
        bc_mask[0] = True
        bc_mask[-1] = True
        self._coords = x
        self._bc_mask = bc_mask

    def __len__(self) -> int:
        return self.cfg.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        a = float(self.freqs[idx].item())  # frequency parameter (condition)
        x = self._coords.clone()  # (N,1)
        u = torch.sin(a * math.pi * x)  # (N,1)
        return {
            "coords": x,                   # (N,1)
            "cond": torch.tensor([a], dtype=self.cfg.dtype),  # (1,)
            "target": u.to(self.cfg.dtype),  # (N,1)
            "bc_mask": self._bc_mask.clone(),  # (N,)
        }
