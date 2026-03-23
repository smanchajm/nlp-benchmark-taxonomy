import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import wandb
from datasets import Dataset

from logging_config import setup_logging
from src.models.classifier import TransformerClassifier, ClassifierConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("config/base.yaml")
SPLITS_DIR = Path("data/splits/ready")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transformer classifier")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to YAML config file (default: config/base.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ClassifierConfig.from_yaml(args.config)

    run_name = f"{cfg.pretrained.split('/')[-1]}_{datetime.now():%Y%m%d}"
    model_tag = cfg.pretrained.split("/")[-1]
    wandb.init(project=cfg.wandb_project, name=run_name, tags=[model_tag])

    train_ds = Dataset.from_parquet(str(SPLITS_DIR / "train.parquet"))
    val_ds = Dataset.from_parquet(str(SPLITS_DIR / "val.parquet"))
    test_ds = Dataset.from_parquet(str(SPLITS_DIR / "test.parquet"))

    model = TransformerClassifier(cfg)
    model.train(train_ds, val_ds)

    metrics = model.evaluate(test_ds)
    logger.info("Test metrics: %s", metrics)
    wandb.log({"test/" + k.removeprefix("eval_"): v for k, v in metrics.items()})

    preds = model.predict(test_ds)
    predictions_df = pd.DataFrame(
        {
            "bibkey": test_ds["bibkey"],
            "label": test_ds["label"],
            "predicted": preds,
        }
    )
    preds_path = Path(cfg.output_dir) / "test_predictions.parquet"
    predictions_df.to_parquet(preds_path, index=False)
    logger.info("Predictions saved to %s", preds_path)
    wandb.log({"test/predictions": wandb.Table(dataframe=predictions_df)})

    model.save(cfg.output_dir)
    logger.info("Model saved to %s", cfg.output_dir)

    wandb.finish()


if __name__ == "__main__":
    setup_logging()
    main()
