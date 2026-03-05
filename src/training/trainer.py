import logging

from pathlib import Path
from datasets import Dataset

from logging_config import setup_logging
from src.data.datasets import make_splits
from src.models.scibert import SciBERTClassifier, SciBERTConfig

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/base.yaml")
SPLITS_DIR = Path("data/splits")


def main() -> None:
    cfg = SciBERTConfig.from_yaml(CONFIG_PATH)

    make_splits()

    train_ds = Dataset.from_parquet(str(SPLITS_DIR / "train.parquet"))
    val_ds = Dataset.from_parquet(str(SPLITS_DIR / "val.parquet"))
    test_ds = Dataset.from_parquet(str(SPLITS_DIR / "test.parquet"))

    model = SciBERTClassifier(cfg)
    model.train(train_ds, val_ds)

    metrics = model.evaluate(test_ds)
    logger.info("Test metrics: %s", metrics)

    model.save(cfg.output_dir)
    logger.info("Model saved to %s", cfg.output_dir)


if __name__ == "__main__":
    setup_logging()
    main()
