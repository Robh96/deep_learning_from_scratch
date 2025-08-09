from __future__ import annotations

import os
import time
import math
import random
import logging
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from contextlib import contextmanager

__all__ = [
	"set_seed",
	"get_device",
	"count_parameters",
	"format_param_count",
	"save_checkpoint",
	"load_checkpoint",
	"AverageMeter",
	"MetricTracker",
	"accuracy",
	"EarlyStopping",
	"to_device",
	"detach_to_cpu",
	"grad_norm",
	"clip_grad",
	"configure_logging",
	"timer",
	"get_num_workers",
	"get_lr",
]


def set_seed(seed: int = 42, fully_deterministic: bool = False) -> None:
	"""
	Set seeds for reproducibility across Python, NumPy, and PyTorch.

	fully_deterministic:
		True  -> slower, but more reproducible (disables CuDNN autotuner, uses deterministic algorithms)
		False -> faster, allows some nondeterminism
	"""
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	# CuDNN settings
	try:
		torch.backends.cudnn.deterministic = fully_deterministic
		torch.backends.cudnn.benchmark = not fully_deterministic
	except Exception:
		pass

	# Deterministic algorithms (PyTorch 1.8+)
	try:
		torch.use_deterministic_algorithms(fully_deterministic)  # type: ignore[attr-defined]
	except Exception:
		pass


def get_device(prefer_gpu: bool = True) -> torch.device:
	"""Return the best available device."""
	if prefer_gpu and torch.cuda.is_available():
		return torch.device("cuda")
	# Apple Silicon (PyTorch MPS)
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
		return torch.device("mps")
	return torch.device("cpu")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
	"""Count model parameters."""
	if trainable_only:
		return sum(p.numel() for p in model.parameters() if p.requires_grad)
	return sum(p.numel() for p in model.parameters())


def format_param_count(n: int) -> str:
	"""Human-readable parameter count."""
	if n >= 1e9:
		return f"{n/1e9:.2f}B"
	if n >= 1e6:
		return f"{n/1e6:.2f}M"
	if n >= 1e3:
		return f"{n/1e3:.2f}K"
	return str(n)


def save_checkpoint(
	path: Union[str, os.PathLike],
	model: nn.Module,
	optimizer: Optional[Optimizer] = None,
	scheduler: Optional[Any] = None,
	epoch: Optional[int] = None,
	metrics: Optional[Dict[str, Any]] = None,
	extra: Optional[Dict[str, Any]] = None,
) -> str:
	"""
	Save a training checkpoint. Returns the path saved to.

	The checkpoint includes:
	- model state_dict
	- optimizer state_dict (if provided)
	- scheduler state_dict (if provided)
	- epoch, metrics, extra metadata (if provided)
	"""
	state: Dict[str, Any] = {
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict() if optimizer is not None else None,
		"scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
		"epoch": epoch,
		"metrics": metrics or {},
		"extra": extra or {},
		"created": time.time(),
		"device": str(next(model.parameters()).device) if any(True for _ in model.parameters()) else "cpu",
	}
	path = str(path)
	dirname = os.path.dirname(path)
	if dirname:
		os.makedirs(dirname, exist_ok=True)
	torch.save(state, path)
	return path


def _adjust_state_dict_for_dataparallel(
	model: nn.Module, state_dict: Mapping[str, Any]
) -> Mapping[str, Any]:
	"""Handle 'module.' prefix mismatch between DataParallel/DistributedDataParallel and single-GPU models."""
	model_keys = list(model.state_dict().keys())
	sd_keys = list(state_dict.keys())
	model_has_module = any(k.startswith("module.") for k in model_keys)
	sd_has_module = any(k.startswith("module.") for k in sd_keys)
	if model_has_module and not sd_has_module:
		return {f"module.{k}": v for k, v in state_dict.items()}
	if sd_has_module and not model_has_module:
		return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
	return state_dict


def load_checkpoint(
	path: Union[str, os.PathLike],
	model: Optional[nn.Module] = None,
	optimizer: Optional[Optimizer] = None,
	scheduler: Optional[Any] = None,
	map_location: Optional[Union[str, torch.device]] = None,
	strict: bool = True,
) -> Dict[str, Any]:
	"""
	Load a checkpoint from path. If model/optimizer/scheduler are provided, their states are restored.
	Returns the loaded checkpoint dict.
	"""
	ckpt = torch.load(path, map_location=map_location or "cpu")

	# If a raw state_dict was saved, wrap it.
	if isinstance(ckpt, Mapping) and "model" not in ckpt and all(isinstance(k, str) for k in ckpt.keys()):
		ckpt = {"model": ckpt}

	if model is not None and "model" in ckpt and isinstance(ckpt["model"], Mapping):
		state_dict = _adjust_state_dict_for_dataparallel(model, ckpt["model"])
		try:
			model.load_state_dict(state_dict, strict=strict)
		except RuntimeError:
			# Retry non-strict as a fallback
			model.load_state_dict(state_dict, strict=False)

	if optimizer is not None and ckpt.get("optimizer") is not None:
		try:
			optimizer.load_state_dict(ckpt["optimizer"])
		except Exception:
			pass

	if scheduler is not None and ckpt.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
		try:
			scheduler.load_state_dict(ckpt["scheduler"])
		except Exception:
			pass

	return ckpt


class AverageMeter:
	"""Track and compute the running average of a scalar metric."""

	def __init__(self) -> None:
		self.reset()

	def reset(self) -> None:
		self.val: float = 0.0
		self.sum: float = 0.0
		self.count: int = 0
		self.avg: float = 0.0

	def update(self, val: float, n: int = 1) -> None:
		self.val = float(val)
		self.sum += float(val) * n
		self.count += n
		self.avg = self.sum / max(self.count, 1)

	def __repr__(self) -> str:
		return f"AverageMeter(val={self.val:.4f}, avg={self.avg:.4f}, count={self.count})"


class MetricTracker:
	"""A thin wrapper around multiple AverageMeters keyed by metric name."""

	def __init__(self) -> None:
		self.meters: Dict[str, AverageMeter] = {}

	def update(self, metrics: Mapping[str, float], n: int = 1) -> None:
		for k, v in metrics.items():
			if k not in self.meters:
				self.meters[k] = AverageMeter()
			self.meters[k].update(float(v), n)

	def reset(self) -> None:
		for m in self.meters.values():
			m.reset()

	def averages(self) -> Dict[str, float]:
		return {k: m.avg for k, m in self.meters.items()}

	def __getitem__(self, key: str) -> AverageMeter:
		return self.meters[key]

	def __contains__(self, key: str) -> bool:
		return key in self.meters

	def __repr__(self) -> str:
		return " | ".join(f"{k}: {m.avg:.4f}" for k, m in self.meters.items())


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1,)) -> List[float]:
	"""
	Compute top-k accuracy for the specified values of k.
	- output: logits or probabilities of shape (N, C)
	- target: ground-truth class indices of shape (N,)
	Returns list of accuracies in percentages.
	"""
	if output.ndim != 2:
		raise ValueError("accuracy() expects output shape (N, C)")
	if target.ndim != 1:
		target = target.view(-1)
	maxk = max(topk)
	_, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
	pred = pred.t()  # (maxk, N)
	correct = pred.eq(target.view(1, -1).expand_as(pred))  # (maxk, N)
	accs: List[float] = []
	for k in topk:
		correct_k = correct[:k].reshape(-1).float().sum(0)
		acc = float(correct_k.mul_(100.0 / output.size(0)))
		accs.append(acc)
	return accs


class EarlyStopping:
	"""
	Early stopping utility.

	mode: "min" or "max"
	patience: epochs to wait for an improvement
	delta: minimum change to qualify as improvement
	"""

	def __init__(self, patience: int = 10, mode: str = "min", delta: float = 0.0) -> None:
		if mode not in {"min", "max"}:
			raise ValueError("mode must be 'min' or 'max'")
		self.patience = patience
		self.mode = mode
		self.delta = delta
		self.best: Optional[float] = None
		self.best_epoch: Optional[int] = None
		self.counter: int = 0
		self.should_stop: bool = False

	def _is_improvement(self, value: float) -> bool:
		if self.best is None:
			return True
		if self.mode == "min":
			return value < self.best - self.delta
		return value > self.best + self.delta

	def step(self, value: float, epoch: Optional[int] = None) -> bool:
		"""
		Update with the latest metric value.
		Returns True if training should stop.
		"""
		if self._is_improvement(value):
			self.best = value
			self.best_epoch = epoch
			self.counter = 0
			self.should_stop = False
		else:
			self.counter += 1
			self.should_stop = self.counter >= self.patience
		return self.should_stop

	def __call__(self, value: float, epoch: Optional[int] = None) -> bool:
		return self.step(value, epoch)


def to_device(
	obj: Any,
	device: Union[str, torch.device],
	non_blocking: bool = True,
) -> Any:
	"""
	Recursively move tensors to a device.
	Supports tensors, nn.Modules, mappings (dict), sequences (list/tuple), and sets.
	"""

	dev = torch.device(device)

	if isinstance(obj, torch.Tensor):
		return obj.to(dev, non_blocking=non_blocking)
	if isinstance(obj, nn.Module):
		return obj.to(dev)
	if isinstance(obj, Mapping):
		return {k: to_device(v, dev, non_blocking=non_blocking) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		seq = [to_device(v, dev, non_blocking=non_blocking) for v in obj]
		return type(obj)(seq)  # preserve list/tuple
	if isinstance(obj, set):
		return {to_device(v, dev, non_blocking=non_blocking) for v in obj}
	return obj


@torch.no_grad()
def detach_to_cpu(obj: Any) -> Any:
	"""
	Recursively detach tensors and move to CPU. Useful before converting to NumPy or logging.
	"""
	if isinstance(obj, torch.Tensor):
		return obj.detach().cpu()
	if isinstance(obj, Mapping):
		return {k: detach_to_cpu(v) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		seq = [detach_to_cpu(v) for v in obj]
		return type(obj)(seq)
	if isinstance(obj, set):
		return {detach_to_cpu(v) for v in obj}
	return obj


def grad_norm(parameters: Iterable[torch.Tensor], norm_type: float = 2.0) -> float:
	"""Compute gradient norm of an iterable of parameters."""
	params = [p for p in parameters if p.grad is not None]
	if not params:
		return 0.0
	norm_type = float(norm_type)
	if math.isinf(norm_type):
		total_norm = max(p.grad.detach().data.abs().max().item() for p in params)
	else:
		total_norm = 0.0
		for p in params:
			param_norm = p.grad.detach().data.norm(norm_type)
			total_norm += float(param_norm.item() ** norm_type)
		total_norm = total_norm ** (1.0 / norm_type)
	return float(total_norm)


def clip_grad(parameters: Iterable[torch.Tensor], max_norm: float, norm_type: float = 2.0) -> float:
	"""Clip gradients and return the total norm before clipping."""
	total_norm = grad_norm(parameters, norm_type)
	torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)
	return total_norm


def configure_logging(
	level: int = logging.INFO,
	log_file: Optional[str] = None,
	name: Optional[str] = None,
	force: bool = False,
) -> logging.Logger:
	"""
	Configure a simple logger. If log_file is provided, logs to both console and file.
	If force is True, existing handlers are removed.
	"""
	logger = logging.getLogger(name)
	if force:
		for h in list(logger.handlers):
			logger.removeHandler(h)
	if logger.handlers:
		logger.setLevel(level)
		return logger

	logger.setLevel(level)
	formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(level)
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)

	if log_file is not None:
		os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
		file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
		file_handler.setLevel(level)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

	return logger


@contextmanager
def timer(name: Optional[str] = None, logger: Optional[logging.Logger] = None):
	"""
	Context manager to time a code block.
	Usage:
		with timer("train-epoch", logger):
			... code ...
	"""
	start = time.perf_counter()
	yield
	elapsed = time.perf_counter() - start
	msg = f"{name} took {elapsed:.3f}s" if name else f"Took {elapsed:.3f}s"
	if logger:
		logger.info(msg)


def get_num_workers(fallback: int = 2) -> int:
	"""
	Simple heuristic for DataLoader num_workers that avoids oversubscription.
	"""
	try:
		cpus = os.cpu_count() or fallback
	except Exception:
		cpus = fallback
	# Keep modest default to work well on laptops/Windows
	return max(0, min(4, cpus - 1))


def get_lr(optimizer: Optimizer) -> List[float]:
	"""Get current learning rates from optimizer."""
	return [group.get("lr", 0.0) for group in optimizer.param_groups]
