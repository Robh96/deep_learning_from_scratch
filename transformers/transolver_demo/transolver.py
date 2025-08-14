"""
Transolver++ (minimal, single-scale) implementation in PyTorch.

This module adapts a vanilla Transformer for PDE operator learning:
- Token = spatial sample (coordinate) with optional per-point features
- Fourier feature positional encoding for coordinates
- Geometry-aware attention: add bias from relative coordinates (distance MLP)
- Conditional normalization (AdaLayerNorm) from global condition vector
- SwiGLU feed-forward and pre-norm residual blocks
- Regression head to map tokens to field values u(x)

Notes
- This is a compact reference implementation intended for clarity.
- It supports optional kNN locality masks for attention.
- For physics-informed losses (residuals/BCs), compute them outside using autograd.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
from torch import nn, Tensor


# ------------------------------
# Encoders
# ------------------------------

class FourierFeatureEncoding(nn.Module):
	"""Sinusoidal positional encoding with multiple frequencies.

	Given coords x in R^{..., D}, returns [sin(2^k pi x), cos(2^k pi x)] over k.

	Args:
		in_dims: spatial coordinate dimension (e.g., 1, 2, or 3)
		num_frequencies: number of frequency bands (>= 1)
		include_input: whether to concatenate raw x to the embedding
		log_space: if True, use frequencies 2^k; else linearly spaced in [1, 2^K]
	"""

	def __init__(
		self,
		in_dims: int,
		num_frequencies: int = 6,
		include_input: bool = True,
		log_space: bool = True,
	) -> None:
		super().__init__()
		assert num_frequencies >= 1
		self.in_dims = in_dims
		self.num_frequencies = num_frequencies
		self.include_input = include_input
		self.log_space = log_space

		if log_space:
			freq_bands = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
		else:
			freq_bands = torch.linspace(1.0, 2.0 ** (num_frequencies - 1), steps=num_frequencies)
		# Register as buffer for device movement
		self.register_buffer("freq_bands", math.pi * freq_bands, persistent=False)

	@property
	def out_dim(self) -> int:
		base = (2 * self.num_frequencies) * self.in_dims
		return base + (self.in_dims if self.include_input else 0)

	def forward(self, x: Tensor) -> Tensor:
		"""x: (B, N, D) -> embedding: (B, N, out_dim)."""
		# Expand for broadcasting: (B, N, D, K)
		# x[..., None] * freq_bands[None, None, None, :]
		x_exp = x.unsqueeze(-1) * self.freq_bands.view(*([1] * (x.ndim - 1)), -1)
		sin = torch.sin(x_exp)
		cos = torch.cos(x_exp)
		pe = torch.cat([sin, cos], dim=-1)  # (B, N, D, 2K)
		pe = pe.flatten(-2)  # (B, N, D*2K)
		if self.include_input:
			pe = torch.cat([x, pe], dim=-1)
		return pe


class AdaLayerNorm(nn.Module):
	"""LayerNorm with per-sample affine from a conditioning vector.

	y = (1 + gamma(cond)) * LN(x) + beta(cond)

	If cond is None, behaves like standard LayerNorm.
	"""

	def __init__(self, d_model: int, cond_dim: int, hidden: int = 128, eps: float = 1e-5) -> None:
		super().__init__()
		self.ln = nn.LayerNorm(d_model, eps=eps)
		self.has_cond = cond_dim is not None and cond_dim > 0
		if self.has_cond:
			self.mlp = nn.Sequential(
				nn.Linear(cond_dim, hidden),
				nn.SiLU(),
				nn.Linear(hidden, 2 * d_model), # output is gamma (scale) and beta (shift)
			)

	def forward(self, x: Tensor, cond: Optional[Tensor]) -> Tensor:
		x_ln = self.ln(x)
		if not self.has_cond or cond is None:
			return x_ln
		# cond: (B, C) or (B, 1, C) -> (B, 1, C)
		if cond.dim() == 2:
			cond = cond.unsqueeze(1)
		gb = self.mlp(cond)  # (B, 1, 2D)
		# gb.chunk(2, dim=-1) splits the last dimension into 2 chunks, returning a tuple (gamma, beta) each with shape (B, 1, D).
		gamma, beta = gb.chunk(2, dim=-1)
		return (1 + gamma) * x_ln + beta


# ------------------------------
# Geometry-aware Multi-Head Attention
# ------------------------------

class GeoBias(nn.Module):
	"""Learned scalar bias from relative coordinates.

	psi(r_ij) -> scalar, broadcast to heads. r_ij can be distances or vectors.
	Here we use relative vector (x_i - x_j) passed through small MLP.
	"""

	def __init__(self, coord_dim: int, hidden: int = 64) -> None:
		super().__init__()
		in_dim = coord_dim
		self.net = nn.Sequential(
			nn.Linear(in_dim, hidden),
			nn.SiLU(),
			nn.Linear(hidden, 1),
		)

	def forward(self, coords: Tensor) -> Tensor:
		"""coords: (B, N, D) -> bias: (B, N, N)
		Computes psi(x_i - x_j).
		"""
		B, N, D = coords.shape
		# r_ij = x_i - x_j => (B, N, N, D)
		xi = coords.unsqueeze(2)
		xj = coords.unsqueeze(1)
		rel = xi - xj
		bias = self.net(rel).squeeze(-1)  # (B, N, N)
		return bias


class SwiGLU(nn.Module):
	def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
		super().__init__()
		self.w1 = nn.Linear(d_model, hidden)
		self.wg = nn.Linear(d_model, hidden)
		self.w2 = nn.Linear(hidden, d_model)
		self.act = nn.SiLU()
		self.dropout = nn.Dropout(dropout)

	def forward(self, x: Tensor) -> Tensor:
		a = self.act(self.w1(x))
		g = self.wg(x)
		y = a * g
		y = self.dropout(y)
		y = self.w2(y)
		y = self.dropout(y)
		return y


class GeoMultiheadAttention(nn.Module):
	def __init__(
		self,
		d_model: int,
		num_heads: int,
		dropout: float,
		coord_dim: int,
		bias: bool = True,
	) -> None:
		super().__init__()
		assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
		self.d_model = d_model
		self.num_heads = num_heads
		self.head_dim = d_model // num_heads
		self.scale = self.head_dim ** -0.5

		self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
		self.proj = nn.Linear(d_model, d_model, bias=bias)
		self.attn_drop = nn.Dropout(dropout)
		self.proj_drop = nn.Dropout(dropout)
		self.geobias = GeoBias(coord_dim)

	def forward(
		self,
		x: Tensor,           # (B, N, D)
		coords: Tensor,      # (B, N, C)
		mask: Optional[Tensor] = None,  # (B, N, N) 1=keep, 0=mask
	) -> Tensor:
		B, N, D = x.shape
		qkv = self.qkv(x)  # (B, N, 3D)
		qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
		qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, Hd)
		q, k, v = qkv[0], qkv[1], qkv[2]

		# scores: (B, H, N, N)
		scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
		# Add geometry bias (broadcast to heads)
		geo = self.geobias(coords).unsqueeze(1)  # (B, 1, N, N)
		scores = scores + geo

		if mask is not None:
			# mask: 1=keep, 0=block
			scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

		attn = torch.softmax(scores, dim=-1)
		attn = self.attn_drop(attn)
		out = torch.matmul(attn, v)  # (B, H, N, Hd)
		out = out.transpose(1, 2).contiguous().view(B, N, D)
		out = self.proj(out)
		out = self.proj_drop(out)
		return out


# ------------------------------
# Transformer Block
# ------------------------------

class TransolverBlock(nn.Module):
	def __init__(
		self,
		d_model: int,
		n_heads: int,
		dropout: float,
		coord_dim: int,
		cond_dim: int = 0,
		ffn_mult: float = 8.0 / 3.0,
	) -> None:
		super().__init__()
		hidden = int(round(ffn_mult * d_model))
		self.ada1 = AdaLayerNorm(d_model, cond_dim)
		self.attn = GeoMultiheadAttention(d_model, n_heads, dropout, coord_dim)
		self.ada2 = AdaLayerNorm(d_model, cond_dim)
		self.ffn = SwiGLU(d_model, hidden, dropout)

	def forward(self, x: Tensor, coords: Tensor, cond: Optional[Tensor], mask: Optional[Tensor]) -> Tensor:
		y = self.ada1(x, cond)
		y = self.attn(y, coords, mask)
		x = x + y
		y = self.ada2(x, cond)
		y = self.ffn(y)
		x = x + y
		return x


# ------------------------------
# Utility: kNN mask
# ------------------------------

def knn_mask(coords: Tensor, k: int) -> Tensor:
	"""Build a symmetric kNN attention mask.

	Args:
		coords: (B, N, C)
		k: number of neighbors to keep per point (include self)
	Returns:
		mask: (B, N, N) with 1 for allowed attention, 0 otherwise
	"""
	B, N, C = coords.shape
	# Compute pairwise distances
	# Using cdist is O(N^2), acceptable for moderate N
	dist = torch.cdist(coords, coords)  # (B, N, N)
	# Get top-k (smallest distances). Ensure self included by k=max(k,1)
	k = max(1, min(k, N))
	idx = dist.topk(k=k, largest=False).indices  # (B, N, k)
	mask = torch.zeros(B, N, N, device=coords.device, dtype=torch.bool)
	mask.scatter_(dim=-1, index=idx, value=True)
	# Symmetrize
	mask = mask | mask.transpose(1, 2)
	# Ensure self-attend
	eye = torch.eye(N, device=coords.device, dtype=torch.bool).unsqueeze(0)
	mask = mask | eye
	return mask.to(torch.float32)


# ------------------------------
# Main Model
# ------------------------------

@dataclass
class TransolverConfig:
	coord_dim: int = 2            # spatial coordinate dimension
	point_feat_dim: int = 0       # optional per-point features
	cond_dim: int = 0             # global condition vector dimension
	d_model: int = 128
	n_heads: int = 8
	n_layers: int = 6
	dropout: float = 0.1
	num_fourier: int = 6
	include_input_coords: bool = True
	use_knn: bool = False
	knn_k: int = 16
	out_dim: int = 1              # output field dimension per point


class TransolverPP(nn.Module):
	"""Transolver++ minimal single-scale encoder-to-field regressor.

	Inputs
	- coords: (B, N, C)
	- point_features: (B, N, F) or None
	- cond: (B, G) or None
	- attn_mask: (B, N, N) optional, 1=keep, 0=mask

	Output
	- y: (B, N, out_dim)
	"""

	def __init__(self, cfg: TransolverConfig) -> None:
		super().__init__()
		self.cfg = cfg

		self.coord_enc = FourierFeatureEncoding(
			in_dims=cfg.coord_dim,
			num_frequencies=cfg.num_fourier,
			include_input=cfg.include_input_coords,
		)

		enc_in_dim = self.coord_enc.out_dim + cfg.point_feat_dim
		self.token_proj = nn.Sequential(
			nn.Linear(enc_in_dim, cfg.d_model),
			nn.SiLU(),
			nn.Dropout(cfg.dropout),
		)

		blocks = []
		for _ in range(cfg.n_layers):
			blocks.append(
				TransolverBlock(
					d_model=cfg.d_model,
					n_heads=cfg.n_heads,
					dropout=cfg.dropout,
					coord_dim=cfg.coord_dim,
					cond_dim=cfg.cond_dim,
				)
			)
		self.blocks = nn.ModuleList(blocks)
		self.final_ln = nn.LayerNorm(cfg.d_model)
		self.head = nn.Sequential(
			nn.Linear(cfg.d_model, cfg.d_model),
			nn.SiLU(),
			nn.Linear(cfg.d_model, cfg.out_dim),
		)

	def forward(
		self,
		coords: Tensor,
		point_features: Optional[Tensor] = None,
		cond: Optional[Tensor] = None,
		attn_mask: Optional[Tensor] = None,
	) -> Tensor:
		B, N, C = coords.shape

		pe = self.coord_enc(coords)
		if point_features is not None:
			x_in = torch.cat([pe, point_features], dim=-1)
		else:
			x_in = pe
		x = self.token_proj(x_in)

		# Build kNN mask if requested and none provided
		if attn_mask is None and self.cfg.use_knn:
			attn_mask = knn_mask(coords, self.cfg.knn_k)

		for blk in self.blocks:
			x = blk(x, coords, cond, attn_mask)

		x = self.final_ln(x)
		y = self.head(x)
		return y


# ------------------------------
# Quick self-test
# ------------------------------

if __name__ == "__main__":
	torch.manual_seed(0)
	cfg = TransolverConfig(coord_dim=2, point_feat_dim=3, cond_dim=4, d_model=96, n_heads=6, n_layers=4, use_knn=True, knn_k=8)
	model = TransolverPP(cfg)
	B, N = 2, 64
	coords = torch.rand(B, N, cfg.coord_dim)
	pfeat = torch.rand(B, N, cfg.point_feat_dim)
	cond = torch.rand(B, cfg.cond_dim)
	out = model(coords, pfeat, cond)
	print("Output:", out.shape)
