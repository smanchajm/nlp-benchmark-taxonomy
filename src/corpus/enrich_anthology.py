import argparse
import logging
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from logging_config import setup_logging
from paths import DATA

logger = logging.getLogger(__name__)

DEFAULT_ACL_OCL = DATA / "acl-ocl.parquet"
DEFAULT_ANTHOLOGY = DATA / "anthology_filtered.parquet"
DEFAULT_OUTPUT = DATA / "anthology_enriched.parquet"


def fetch_acl_ocl_dataset(output_path: Path = DEFAULT_ACL_OCL) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hf_path = hf_hub_download(
        repo_id="WINGNUS/ACL-OCL",
        filename="acl-publication-info.74k.v2.parquet",
        repo_type="dataset",
    )
    logger.info("Loading ACL-OCL from HF...")
    dataset = pd.read_parquet(hf_path)

    logger.info("Writing %d papers to %s...", len(dataset), output_path)
    dataset.to_parquet(output_path, index=False)
    logger.info("Done.")


def enrich_anthology(
    anthology_path: Path = DEFAULT_ANTHOLOGY,
    acl_ocl_path: Path = DEFAULT_ACL_OCL,
    output_path: Path = DEFAULT_OUTPUT,
) -> None:
    df_anthology = pd.read_parquet(anthology_path)
    df_acl_ocl = pd.read_parquet(acl_ocl_path)

    df_enriched = df_anthology.merge(
        df_acl_ocl[["acl_id", "abstract", "numcitedby"]],
        left_on="id",
        right_on="acl_id",
        how="left",
        suffixes=("", "_ocl"),
    )

    df_enriched["abstract"] = df_enriched["abstract"].fillna(
        df_enriched["abstract_ocl"]
    )
    df_enriched = df_enriched.drop(columns=["acl_id", "abstract_ocl"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_parquet(output_path, index=False)
    logger.info("Saved → %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge ACL Anthology with ACL-OCL")
    parser.add_argument("--anthology", type=Path, default=DEFAULT_ANTHOLOGY)
    parser.add_argument("--acl-ocl", type=Path, default=DEFAULT_ACL_OCL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fetch-acl-ocl",
        action="store_true",
        help="Download the ACL-OCL parquet before merging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    if args.fetch_acl_ocl:
        fetch_acl_ocl_dataset(args.acl_ocl)
    enrich_anthology(args.anthology, args.acl_ocl, args.output)
