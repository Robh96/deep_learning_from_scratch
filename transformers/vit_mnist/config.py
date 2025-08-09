from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DataConfig:
    root: str = "../datasets"
    batch_size: int = 64
    num_workers: int = 2
    target_size: int = 224
    make_rgb: bool = True

@dataclass
class ModelConfig:
    ch: int = 3
    img_size: int = 224
    patch_size: int = 16
    embedding_size: int = 768
    n_layers: int = 6
    num_heads: int = 8
    num_classes: int = 10
    dropout: float = 0.1

@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    seed: int = 42
    save_dir: str = "results"

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def save_path(self) -> Path:
        p = Path(self.train.save_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p