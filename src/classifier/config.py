from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ClassifierConfig:
    pretrained: str
    max_length: int
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    seed: int
    wandb_project: str
    num_labels: int
    class_weight_smoothing: float
    early_stopping_patience: int
    bf16: bool = True
    uncertainty_threshold: float = 0.7

    @classmethod
    def from_dict(cls, d: dict) -> ClassifierConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str | Path) -> ClassifierConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw["model"])
