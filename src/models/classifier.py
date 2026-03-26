from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import Dataset
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


def _remap_legacy_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename SciBERT-style LayerNorm keys: gamma→weight, beta→bias."""
    return {
        k.replace(".gamma", ".weight").replace(".beta", ".bias"): v
        for k, v in state_dict.items()
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device)
        )
        return (loss, outputs) if return_outputs else loss


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

    @classmethod
    def from_dict(cls, d: dict) -> ClassifierConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str | Path) -> ClassifierConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw["model"])


class TransformerClassifier(BaseModel):
    def __init__(self, config: ClassifierConfig) -> None:
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.pretrained, num_labels=config.num_labels
        )
        self.trainer: Trainer | None = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _tokenize(self, dataset: Dataset) -> Dataset:
        has_title = "title" in dataset.column_names
        return dataset.map(
            lambda batch: self.tokenizer(
                [
                    f"{t} {self.tokenizer.sep_token} {a}"
                    for t, a in zip(batch["title"], batch["abstract"])
                ]
                if has_title
                else batch["abstract"],
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
            ),
            batched=True,
        )

    def _compute_metrics(self, eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    def train(self, train_data: Dataset, val_data: Dataset) -> None:
        train_tok = self._tokenize(train_data)
        val_tok = self._tokenize(val_data)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=False,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=self.config.seed,
            bf16=self.config.bf16,
            report_to="wandb",
        )

        labels = np.array(train_tok["label"])
        balanced = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        s = self.config.class_weight_smoothing
        smoothed = s * balanced + (1 - s) * np.ones_like(balanced)
        class_weights = torch.tensor(smoothed, dtype=torch.float32)
        logger.info("Class weights (smoothing=%.1f): %s", s, class_weights)

        self.trainer = WeightedTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
                )
            ],
        )

        logger.info("Starting training — %d epochs", self.config.num_train_epochs)
        self.trainer.train()

        best_ckpt = self.trainer.state.best_model_checkpoint
        logger.info(
            "Best checkpoint: %s (F1=%.4f)",
            best_ckpt,
            self.trainer.state.best_metric,
        )
        # Reload best checkpoint with explicit gamma/beta → weight/bias
        # remapping (from_pretrained alone doesn't handle this reliably).
        if best_ckpt:
            ckpt_path = Path(best_ckpt) / "model.safetensors"
            state_dict = _remap_legacy_keys(load_safetensors(str(ckpt_path)))
            self.model.load_state_dict(state_dict)
            logger.info("Loaded best checkpoint weights from %s", ckpt_path)
        self.trainer.model = self.model
        self.best_metric = self.trainer.state.best_metric

    def test(self, test_data: Dataset) -> dict[str, float]:
        if self.trainer is None:
            raise RuntimeError("Model must be trained or loaded before testing.")
        test_tok = self._tokenize(test_data)
        return self.trainer.evaluate(test_tok, metric_key_prefix="test")

    def predict(self, dataset: Dataset) -> list[int]:
        if self.trainer is None:
            raise RuntimeError("Model must be trained before prediction.")
        tok = self._tokenize(dataset)
        output = self.trainer.predict(tok)
        return output.predictions.argmax(axis=-1).tolist()

    def save(self, path: str | Path) -> None:
        if self.trainer is None:
            raise RuntimeError("Model must be trained before saving.")
        self.trainer.save_model(str(path))

    def load(self, path: str | Path) -> None:
        """Load a saved checkpoint and set up a Trainer for evaluation."""
        ckpt_path = Path(path) / "model.safetensors"
        state_dict = _remap_legacy_keys(load_safetensors(str(ckpt_path)))
        self.model.load_state_dict(state_dict)
        logger.info("Loaded model weights from %s", ckpt_path)

        self.trainer = Trainer(
            model=self.model,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self._compute_metrics,
        )

    def push_to_hf_hub(self, repo_id: str) -> None:
        self.model.push_to_hub(repo_id)
        self.tokenizer.push_to_hub(repo_id)
