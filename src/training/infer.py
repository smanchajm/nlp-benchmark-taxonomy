import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset

from logging_config import setup_logging
from src.models.classifier import ClassifierConfig, TransformerClassifier

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on unlabelled papers")
    parser.add_argument("input", type=Path, help="Path to input parquet file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--output", type=Path, default=None, help="Output parquet path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ClassifierConfig.from_yaml(args.config)
    cfg.output_dir = str(args.checkpoint)

    model = TransformerClassifier(cfg)
    model.load(args.checkpoint)

    ds = Dataset.from_parquet(str(args.input))
    preds, probas = model.predict(ds)

    df = pd.DataFrame(
        {
            "bibkey": ds["bibkey"],
            "predicted": preds,
            "confidence": probas.max(axis=1),
            "prob_positive": probas[:, 1],
        }
    )

    output_path = args.output or args.input.with_name(f"{args.input.stem}_predictions.parquet")
    df.to_parquet(output_path, index=False)
    logger.info("Predictions saved to %s (%d rows)", output_path, len(df))


if __name__ == "__main__":
    setup_logging()
    main()
