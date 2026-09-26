"""Sample Gold-Taxo and Gold-eligibility sets for resource evaluation."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from logging_config import setup_logging
from paths import DATA

logger = logging.getLogger(__name__)

EVAL_DIR = DATA / "evaluation"
DEFAULT_INPUT_PATH = DATA / "corpus" / "single_task_benchmark_paper.parquet"
DEFAULT_SIZE = 150
DEFAULT_SEED = 42

OUTPUT_COLUMNS = (
    "bibkey", "anthology_id", "title", "abstract", "pdf_url",
    "dataset_url", "language", "benchmark_languages", "thematic_domain",
    "paraphrase_task",
)


def _flatten_lang(val: object) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, np.ndarray):
        return ", ".join(str(v) for v in val.tolist())
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val)


def _validate_input(df: pd.DataFrame, size: int) -> int:
    missing = [col for col in OUTPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    n_ids = int(df["bibkey"].nunique())
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if size > n_ids:
        raise ValueError(f"size ({size}) exceeds unique bibkeys ({n_ids})")
    return n_ids


def generate_gold_taxo_set(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = EVAL_DIR,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
) -> Path:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    logger.info("Loading corpus from %s", input_path)
    input_df = pd.read_parquet(input_path)
    n_ids = _validate_input(input_df, size)

    ids = sorted(input_df["bibkey"].unique())
    logger.info("Sampling %d papers from %d candidates (seed=%d)", size, n_ids, seed)

    rng = np.random.default_rng(seed)
    picked_ids = rng.choice(ids, size, replace=False)

    sampled = (
        input_df[input_df["bibkey"].isin(picked_ids)][list(OUTPUT_COLUMNS)]
        .sort_values("bibkey")
        .reset_index(drop=True)
    )
    sampled["language"] = sampled["language"].apply(_flatten_lang)
    sampled["benchmark_languages"] = sampled["benchmark_languages"].apply(_flatten_lang)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"gold_taxo_seed{seed}_size{size}_{datetime.now().strftime('%Y%m%d')}.csv"
    )
    sampled.to_csv(output_path, index=False)
    logger.info("Wrote %d papers to %s", len(sampled), output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a random Gold-Taxo set for taxonomy evaluation.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=EVAL_DIR)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    generate_gold_taxo_set(args.input, args.output, args.size, args.seed)
