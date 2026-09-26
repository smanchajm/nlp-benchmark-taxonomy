"""Export Label Studio task JSON from the benchmark parquet.

Produces one JSON file per config:
  - taxo:       fields needed by label_studio_taxo_config.xml
  - validation: fields needed by label_studio_validation_config.xml (superset)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logging_config import setup_logging
from paths import DATA

logger = logging.getLogger(__name__)

PARQUET = DATA / "corpus" / "single_task_benchmark_paper.parquet"
OUT_DIR = DATA / "evaluation"

TAXO_FIELDS = ["bibkey", "anthology_id", "title", "abstract", "pdf_url", "paraphrase_task"]
VALIDATION_FIELDS = TAXO_FIELDS + ["dataset_url", "language", "thematic_domain", "benchmark_languages"]


def _flatten_lang(val: object) -> str:
    """Collapse numpy array / list of languages to a comma-separated string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, np.ndarray):
        return ", ".join(str(v) for v in val.tolist())
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val)


def _row_to_task(row: pd.Series, fields: list[str]) -> dict:
    data: dict = {}
    for f in fields:
        val = row.get(f, "")
        if f == "benchmark_languages":
            data[f] = _flatten_lang(val)
        elif val is None or (isinstance(val, float) and np.isnan(val)):
            data[f] = ""
        else:
            data[f] = str(val)
    return {"data": data}


def export(
    parquet: Path = PARQUET,
    out_dir: Path = OUT_DIR,
    mode: str = "validation",
) -> Path:
    df = pd.read_parquet(parquet)
    logger.info("Loaded %d rows from %s", len(df), parquet)

    fields = VALIDATION_FIELDS if mode == "validation" else TAXO_FIELDS
    tasks = [_row_to_task(row, fields) for _, row in df.iterrows()]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"label_studio_tasks_{mode}.json"
    out_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d tasks to %s", len(tasks), out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Label Studio tasks from parquet.")
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--mode", choices=["taxo", "validation"], default="validation",
                        help="taxo: minimal fields; validation: includes dataset_url/language/domain")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    export(args.parquet, args.out_dir, args.mode)
