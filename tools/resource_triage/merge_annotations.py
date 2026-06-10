"""Fusionne resource_annotations.csv dans le parquet corpus."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from logging_config import setup_logging
from paths import DATA

logger = logging.getLogger(__name__)

DEFAULT_BASE = DATA / "corpus" / "single_task_benchmark_paper.parquet"
DEFAULT_ANNOTATIONS = DATA / "corpus" / "resource_links" / "resource_annotations.csv"
DEFAULT_OUTPUT = DATA / "corpus" / "single_task_benchmark_paper_enriched_resources.parquet"

RESOURCE_COLUMNS = [
    "dataset_url",
    "no_dataset_url",
    "code_url",
    "resource_license",
    "resource_format",
    "resource_access",
    "resource_notes",
]


def merge_annotations(
    base_path: Path,
    annotations_path: Path,
    output_path: Path,
) -> None:
    base = pd.read_parquet(base_path)
    if "anthology_id" not in base.columns:
        raise ValueError(f"Colonne anthology_id absente dans {base_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations introuvables: {annotations_path}")

    ann = pd.read_csv(annotations_path)
    if "paper" not in ann.columns:
        raise ValueError("CSV annotations: colonne 'paper' requise (anthology_id).")

    ann = ann.rename(columns={"paper": "anthology_id"})
    ann["anthology_id"] = ann["anthology_id"].astype("string").str.strip()
    ann = ann.drop_duplicates(subset=["anthology_id"], keep="last")

    for col in RESOURCE_COLUMNS:
        if col not in ann.columns:
            ann[col] = pd.NA
        base[col] = pd.NA

    ann_indexed = ann.set_index("anthology_id")
    touched_papers: set[str] = set()
    for aid, row in ann_indexed.iterrows():
        mask = base["anthology_id"] == aid
        if not mask.any():
            logger.warning("Annotation ignorée (id inconnu): %s", aid)
            continue
        touched_papers.add(str(aid))
        for col in RESOURCE_COLUMNS:
            if col not in row.index:
                continue
            val = row[col]
            if pd.isna(val) or str(val).strip() == "":
                continue
            base.loc[mask, col] = val

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(output_path, index=False)
    logger.info(
        "Parquet enrichi: %d lignes, %d papiers annotes -> %s",
        len(base),
        len(touched_papers),
        output_path,
    )
    filled = base["dataset_url"].notna() & (base["dataset_url"].astype(str).str.strip() != "")
    logger.info("dataset_url renseigné: %d / %d", int(filled.sum()), len(base))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    merge_annotations(args.base, args.annotations, args.output)
