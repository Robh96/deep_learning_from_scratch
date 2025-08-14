"""
Minimal training loop for Transolver++ on 1D Helmholtz demo.

Loss = data MSE + boundary MSE + interior finite-difference residual MSE.

Run:
    python -m transformers.transolver_demo.train_demo
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, random_split

from .dataset import SineHelmholtz1DDataset, SineHelmholtz1DConfig
from .transolver import TransolverPP, TransolverConfig



@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 10
    lr: float = 3e-3
    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Loss weights
    w_data: float = 1.0
    w_bc: float = 1.0
    w_res: float = 0.1


def fd_laplacian_1d(y: Tensor, h: float) -> Tensor:
    """Second derivative via central differences on uniform grid.
    y: (B,N,1)
    Returns: (B,N,1) with endpoints set to 0 (ignored by residual loss)
    """
    B, N, C = y.shape
    laplacian = torch.zeros_like(y)
    # interior i in [1, N-2]
    laplacian[:, 1:-1, 0] = (y[:, 2:, 0] - 2*y[:, 1:-1, 0] + y[:, :-2, 0]) / (h*h)
    return laplacian


def collate(batch):
    # All samples share same grid size; stack with shapes:
    # coords: (B,N,1), cond: (B,G), target: (B,N,1), bc_mask: (B,N)
    coords = torch.stack([b["coords"] for b in batch], dim=0)
    cond = torch.stack([b["cond"] for b in batch], dim=0)
    target = torch.stack([b["target"] for b in batch], dim=0)
    bc_mask = torch.stack([b["bc_mask"] for b in batch], dim=0)
    return {"coords": coords, "cond": cond, "target": target, "bc_mask": bc_mask}


def train_one_epoch(model: TransolverPP, loader: DataLoader, opt: torch.optim.Optimizer, cfg: TrainConfig) -> Tuple[float, float, float, float]:
    model.train()
    w_data, w_bc, w_res = cfg.w_data, cfg.w_bc, cfg.w_res
    mse = nn.MSELoss()
    total, loss_data_sum, loss_boundary_condition_sum, loss_residual_sum = 0.0, 0.0, 0.0, 0.0

    for batch in loader:
        coordinates = batch["coords"].to(cfg.device)  # (B,N,1)
        condition = batch["cond"].to(cfg.device)      # (B,1)
        target = batch["target"].to(cfg.device)       # (B,N,1)
        boundary_mask = batch["bc_mask"].to(cfg.device)  # (B,N)

        opt.zero_grad()
        prediction = model(coordinates, point_features=None, cond=condition)  # (B,N,1)

        # Data loss (all points)
        loss_data = mse(prediction, target)

        # Boundary loss (Dirichlet u=0 at ends)
        # Only mask boundary points
        if boundary_mask.any():
            prediction_bc = prediction[boundary_mask].view(-1)
            target_bc = target[boundary_mask].view(-1)
            loss_boundary_condition = mse(prediction_bc, target_bc)
        else:
            loss_boundary_condition = torch.tensor(0.0, device=prediction.device)

        # Residual loss (interior): u'' + (a*pi)^2 u = 0
        N = coordinates.shape[1]
        grid_spacing = (coordinates[:, 1, 0] - coordinates[:, 0, 0]).mean().item()
        laplacian = fd_laplacian_1d(prediction, grid_spacing)
        a = condition[:, 0].view(-1, 1, 1)  # (B,1,1)
        residual = (laplacian + (a * math.pi) ** 2 * prediction)
        # Mask interior only
        interior_mask = torch.ones_like(boundary_mask, dtype=torch.bool)
        interior_mask[:, 0] = False
        interior_mask[:, -1] = False
        loss_residual = mse(residual[interior_mask].view(-1), torch.zeros_like(residual[interior_mask].view(-1)))

        loss = w_data * loss_data + w_bc * loss_boundary_condition + w_res * loss_residual
        loss.backward()
        opt.step()

        batch_size = coordinates.shape[0]
        total += batch_size
        loss_data_sum += loss_data.item() * batch_size
        loss_boundary_condition_sum += loss_boundary_condition.item() * batch_size
        loss_residual_sum += loss_residual.item() * batch_size

    return (
    (loss_data_sum / total),
    (loss_boundary_condition_sum / total),
    (loss_residual_sum / total),
    (w_data * loss_data_sum + w_bc * loss_boundary_condition_sum + w_res * loss_residual_sum) / total,
    )


def evaluate(model: TransolverPP, loader: DataLoader, cfg: TrainConfig) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            coords = batch["coords"].to(cfg.device)
            cond = batch["cond"].to(cfg.device)
            target = batch["target"].to(cfg.device)
            pred = model(coords, cond=cond)
            total_loss += mse(pred, target).item()
            total_count += coords.shape[0]
    return total_loss / total_count


def main():
    # Data
    dataset_config = SineHelmholtz1DConfig(num_samples=1200, n_points=129)
    dataset = SineHelmholtz1DDataset(dataset_config)
    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    dataset_train, dataset_val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(dataset_train, batch_size=TrainConfig.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(dataset_val, batch_size=TrainConfig.batch_size, shuffle=False, collate_fn=collate)

    # Model
    model_config = TransolverConfig(coord_dim=1, point_feat_dim=0, cond_dim=1, d_model=128, n_heads=8, n_layers=6, dropout=0.1, use_knn=False)
    model = TransolverPP(model_config).to(TrainConfig.device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainConfig.lr, weight_decay=TrainConfig.weight_decay)

    # Train
    print(f"Device: {TrainConfig.device}")
    best_val_mse = float('inf')
    for epoch in range(1, TrainConfig.epochs + 1):
        t0 = time.time()
        loss_data, loss_boundary_condition, loss_residual, loss_total = train_one_epoch(model, train_loader, optimizer, TrainConfig)
        val_mse = evaluate(model, val_loader, TrainConfig)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d} | data {loss_data:.4e} | bc {loss_boundary_condition:.4e} | res {loss_residual:.4e} | val {val_mse:.4e} | {dt:.1f}s")
        best_val_mse = min(best_val_mse, val_mse)

    print(f"Best val MSE: {best_val_mse:.4e}")


if __name__ == "__main__":
    main()
