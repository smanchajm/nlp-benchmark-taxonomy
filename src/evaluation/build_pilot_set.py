"""Build a small pilot set to onboard/test new annotators.

Samples N papers at random among those already annotated as classification
benchmarks (skip empty) in a Label Studio export, then emits:
  - a Label Studio task JSON ready to import (taxo config fields)
  - a reference CSV with the gold leaf/mother labels, for the organizer only
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
from export_label_studio_tasks import TAXO_FIELDS, _row_to_task
from logging_config import setup_logging
from paths import DATA

logger = logging.getLogger(__name__)

PARQUET = DATA / "corpus" / "single_task_benchmark_paper.parquet"
TAXONOMY = DATA / "taxonomy" / "manual_taxonomy_tree.json"
OUT_DIR = DATA / "evaluation"

DEFAULT_SIZE = 15
DEFAULT_SEED = 42


def build_leaf_to_mother(taxonomy_path: Path = TAXONOMY) -> dict[str, str]:
    """Map every taxonomy leaf label to its top-level mother label."""
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))

    def walk(node: dict, mother: str | None = None):
        for child in node.get("children") or []:
            if not isinstance(child, dict):
                continue
            kids = [k for k in (child.get("children") or []) if isinstance(k, dict)]
            if kids:
                yield from walk(child, mother or child["label"])
            else:
                yield child["label"], mother or child["label"]

    return dict(walk(taxonomy))


def parse_task_leaf(value: object) -> str | None:
    """Extract the selected leaf label from a Label Studio <Taxonomy> cell."""
    if pd.isna(value):
        return None
    try:
        return json.loads(str(value))[0]["taxonomy"][0][-1]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None


def build_pilot_set(
    export_path: Path,
    parquet: Path = PARQUET,
    out_dir: Path = OUT_DIR,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, Path]:
    ann = pd.read_csv(export_path)
    ann["gold_leaf"] = ann["task"].apply(parse_task_leaf)

    eligible = ann[ann["skip"].isna() & ann["gold_leaf"].notna()]
    logger.info(
        "Export: %d rows | skipped (not classif bench): %d | eligible: %d",
        len(ann), int(ann["skip"].notna().sum()), len(eligible),
    )
    if size > len(eligible):
        raise ValueError(f"size ({size}) exceeds eligible papers ({len(eligible)})")

    rng = np.random.default_rng(seed)
    picked = rng.choice(eligible["bibkey"].to_numpy(), size, replace=False)

    leaf_to_mother = build_leaf_to_mother()
    gold = (
        eligible[eligible["bibkey"].isin(picked)][["bibkey", "gold_leaf"]]
        .assign(gold_mother=lambda d: d["gold_leaf"].map(leaf_to_mother).fillna(d["gold_leaf"]))
        .sort_values("bibkey")
        .reset_index(drop=True)
    )

    df = pd.read_parquet(parquet)
    subset = df[df["bibkey"].isin(picked)]
    missing = set(picked) - set(subset["bibkey"])
    if missing:
        logger.warning("Not found in parquet, dropped: %s", sorted(missing))

    tasks = [_row_to_task(row, TAXO_FIELDS) for _, row in subset.iterrows()]

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = out_dir / f"pilot_set_seed{seed}_size{size}_tasks.json"
    gold_path = out_dir / f"pilot_set_seed{seed}_size{size}_gold.csv"
    tasks_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    gold.to_csv(gold_path, index=False)

    logger.info("Wrote %d tasks to %s", len(tasks), tasks_path)
    logger.info("Wrote gold reference to %s", gold_path)
    return tasks_path, gold_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a random pilot set from an annotated Label Studio export.",
    )
    parser.add_argument("export", type=Path, help="Label Studio CSV export with gold annotations")
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    build_pilot_set(args.export, args.parquet, args.out_dir, args.size, args.seed)
