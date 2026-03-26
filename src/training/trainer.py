import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import wandb
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run as a W&B sweep agent (overrides config with wandb.config)",
    )
    parser.add_argument(
        "--test-only",
        type=Path,
        default=None,
        help="Skip training; evaluate a saved checkpoint on the test set",
    )
    return parser.parse_args()


def _apply_sweep_overrides(cfg: ClassifierConfig) -> ClassifierConfig:
    """Override config fields with values from wandb.config (set by sweep agent)."""
    sweep_params = dict(wandb.config)
    for key, value in sweep_params.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
            logger.info("Sweep override: %s = %s", key, value)
    return cfg


def train(cfg: ClassifierConfig) -> None:
    """Train + validate only. No test set evaluation."""
    train_ds = Dataset.from_parquet(str(SPLITS_DIR / "train.parquet"))
    val_ds = Dataset.from_parquet(str(SPLITS_DIR / "val.parquet"))

    model = TransformerClassifier(cfg)
    model.train(train_ds, val_ds)
    wandb.summary["best_val_f1"] = model.best_metric
    model.save(cfg.output_dir)
    logger.info("Model saved to %s", cfg.output_dir)


def test(cfg: ClassifierConfig) -> None:
    """Test a trained model on the held-out test set and log results."""
    test_ds = Dataset.from_parquet(str(SPLITS_DIR / "test.parquet"))

    model = TransformerClassifier(cfg)
    model.load(cfg.output_dir)

    metrics, preds, probas = model.test(test_ds)
    labels = test_ds["label"]
    logger.info("Test metrics: %s", metrics)
    wandb.log({"test/" + k: v for k, v in metrics.items()})
    confidence = probas.max(axis=1)
    predictions_df = pd.DataFrame(
        {
            "bibkey": test_ds["bibkey"],
            "label": labels,
            "predicted": preds,
            "confidence": confidence,
            "prob_positive": probas[:, 1],
        }
    )
    if "bucket" in test_ds.column_names:
        predictions_df["bucket"] = test_ds["bucket"]

    preds_path = Path(cfg.output_dir) / "test_predictions.parquet"
    predictions_df.to_parquet(preds_path, index=False)
    logger.info("Predictions saved to %s", preds_path)

    # Log uncertain predictions (confidence < 0.7)
    uncertain = predictions_df[
        predictions_df["confidence"] < cfg.uncertainty_threshold
    ].sort_values("confidence")
    wandb.summary["test/n_uncertain"] = len(uncertain)
    wandb.log({"test/uncertain": wandb.Table(dataframe=uncertain)})
    wandb.log({"test/predictions": wandb.Table(dataframe=predictions_df)})
    wandb.log(
        {
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=labels,
                preds=preds,
                class_names=["negative", "positive"],
            )
        }
    )

    # Per-bucket metrics
    if "bucket" in predictions_df.columns:
        for bucket, group in predictions_df.groupby("bucket"):
            p, r, f1, _ = precision_recall_fscore_support(
                group["label"], group["predicted"], average="binary"
            )
            acc = accuracy_score(group["label"], group["predicted"])
            bucket_metrics = {
                f"test/{bucket}/accuracy": acc,
                f"test/{bucket}/precision": p,
                f"test/{bucket}/recall": r,
                f"test/{bucket}/f1": f1,
                f"test/{bucket}/support": len(group),
            }
            wandb.log(bucket_metrics)
            wandb.log(
                {
                    f"test/{bucket}/confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=group["label"].tolist(),
                        preds=group["predicted"].tolist(),
                        class_names=["negative", "positive"],
                    )
                }
            )
            logger.info("Bucket %s metrics: %s", bucket, bucket_metrics)


def main() -> None:
    args = parse_args()
    cfg = ClassifierConfig.from_yaml(args.config)

    model_tag = cfg.pretrained.split("/")[-1]

    if args.sweep:
        wandb.init()
        cfg = _apply_sweep_overrides(cfg)
        cfg.output_dir = str(Path(cfg.output_dir) / wandb.run.id)
        train(cfg)
    elif args.test_only:
        cfg.output_dir = str(args.test_only)
        run_name = f"{model_tag}_test_{datetime.now():%Y%m%d_%H%M}"
        wandb.init(project=cfg.wandb_project, name=run_name, tags=[model_tag, "test"])
        test(cfg)
    else:
        run_name = f"{model_tag}_{datetime.now():%Y%m%d_%H%M}"
        wandb.init(project=cfg.wandb_project, name=run_name, tags=[model_tag])
        train(cfg)

    wandb.finish()


if __name__ == "__main__":
    setup_logging()
    main()
