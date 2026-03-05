from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class SciBERTConfig:
    pretrained: str
    max_length: int
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    seed: int
    wandb_project: str
    num_labels: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> SciBERTConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str | Path) -> SciBERTConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw["model"])


class SciBERTClassifier(BaseModel):
    def __init__(self, config: SciBERTConfig) -> None:
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
        return dataset.map(
            lambda batch: self.tokenizer(
                batch["abstract"],
                truncation=True,
                padding="max_length",
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
            warmup_ratio=self.config.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=self.config.seed,
            report_to="wandb",
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self._compute_metrics,
        )

        logger.info("Starting training — %d epochs", self.config.num_train_epochs)
        self.trainer.train()

    def evaluate(self, test_data: Dataset) -> dict[str, float]:
        if self.trainer is None:
            raise RuntimeError("Model must be trained before evaluation.")
        test_tok = self._tokenize(test_data)
        return self.trainer.evaluate(test_tok)

    def predict(self, texts: list[str]) -> list[int]:
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        self.model.eval()
        outputs = self.model(**tokenized)
        return outputs.logits.argmax(dim=-1).tolist()

    def save(self, path: str | Path) -> None:
        if self.trainer is None:
            raise RuntimeError("Model must be trained before saving.")
        self.trainer.save_model(str(path))
        self.tokenizer.save_pretrained(str(path))

    def push_to_hf_hub(self, repo_id: str) -> None:
        self.model.push_to_hub(repo_id)
        self.tokenizer.push_to_hub(repo_id)
