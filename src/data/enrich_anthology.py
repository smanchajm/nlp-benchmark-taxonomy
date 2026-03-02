import logging
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from logging_config import ROOT, setup_logging


logger = logging.getLogger(__name__)

INPUT_ACL_OCL_PATH = ROOT / "data/acl-ocl.parquet"
INPUT_ACL_ANTHOLOGY_PATH = ROOT / "data/anthology_clean.parquet"


def fetch_acl_ocl_dataset(output_path: Path = Path("data/acl-ocl.parquet")) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hf_path = hf_hub_download(
        repo_id="WINGNUS/ACL-OCL",
        filename="acl-publication-info.74k.v2.parquet",
        repo_type="dataset",
    )
    logger.info("Loading ACL-OCL from HF...")
    dataset = pd.read_parquet(hf_path)

    logger.info("Writing %d papers to %s...", len(dataset), output_path)
    pd.DataFrame(dataset).to_parquet(output_path, index=False)
    logger.info("Done.")


def enrich_anthology() -> None:
    df_anthology = pd.read_parquet(INPUT_ACL_ANTHOLOGY_PATH)
    df_acl_ocl = pd.read_parquet(INPUT_ACL_OCL_PATH)

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

    OUTPUT_ENRICHED_PATH = ROOT / "data/anthology_enriched.parquet"
    OUTPUT_ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_parquet(OUTPUT_ENRICHED_PATH, index=False)
    logger.info("Saved → %s", OUTPUT_ENRICHED_PATH)


if __name__ == "__main__":
    setup_logging()
    # fetch_acl_ocl_dataset()
    enrich_anthology()
