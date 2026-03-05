from abc import ABC, abstractmethod

from pathlib import Path
from datasets import Dataset


class BaseModel(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def train(self, train_data: Dataset, val_data: Dataset) -> None:
        pass

    @abstractmethod
    def evaluate(self, test_data: Dataset) -> dict[str, float]:
        pass

    @abstractmethod
    def predict(self, texts: list[str]) -> list[int]:
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        pass

    @abstractmethod
    def push_to_hf_hub(self, repo_id: str) -> None:
        pass
