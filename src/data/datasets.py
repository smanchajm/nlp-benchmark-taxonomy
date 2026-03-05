import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from logging_config import setup_logging

logger = logging.getLogger(__name__)

LABELED_PATH = Path("data/labeled/labels.parquet")
SPLITS_DIR = Path("data/splits")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42


def make_splits(force: bool = False) -> None:
    """Load the labeled parquet and write train/val/test splits."""
    split_files = {
        "train": SPLITS_DIR / "train.parquet",
        "val": SPLITS_DIR / "val.parquet",
        "test": SPLITS_DIR / "test.parquet",
    }

    if not force and all(p.exists() for p in split_files.values()):
        logger.info("Splits already exist in %s — skipping.", SPLITS_DIR)
        return

    df = pd.read_parquet(LABELED_PATH)
    logger.info("Loaded %d labeled examples from %s", len(df), LABELED_PATH)

    val_test_ratio = 1.0 - TRAIN_RATIO
    relative_val = VAL_RATIO / val_test_ratio

    train_df, val_test_df = train_test_split(
        df, test_size=val_test_ratio, stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=1.0 - relative_val,
        stratify=val_test_df["label"],
        random_state=SEED,
    )

    train_df.to_parquet(split_files["train"], index=False)
    val_df.to_parquet(split_files["val"], index=False)
    test_df.to_parquet(split_files["test"], index=False)

    logger.info(
        "Splits written — train: %d | val: %d | test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )


if __name__ == "__main__":
    setup_logging()
    make_splits()
