import logging

import pandas as pd

from logging_config import ROOT, setup_logging

logger = logging.getLogger(__name__)

INPUT_PATH = ROOT / "data/anthology.parquet"
OUTPUT_PATH = ROOT / "data/anthology_clean.parquet"
OUTPUT_WITH_ABSTRACT_PATH = ROOT / "data/anthology_clean_with_abstract.parquet"

YEAR_FROM = 2010
YEAR_TO = 2025

# All is_toplevel venues from the ACL Anthology, minus journals and the generic
# workshop wrapper ("ws").
# Journals excluded: cl, ijclclp, jlcl, lilt, nejlt, tacl, tal.
VENUES: frozenset[str] = frozenset(
    {
        "aacl",
        "acl",
        "aimecon",
        "alta",
        "amta",
        "anlp",
        "arabicnlp",
        "ccl",
        "clicit",
        "coling",
        "conll",
        "eacl",
        "eamt",
        "emnlp",
        "findings",
        "hlt",
        "ijcnlp",
        "iwsds",
        "iwslt",
        "jeptalnrecital",
        "konvens",
        "lrec",
        "mtsummit",
        "muc",
        "naacl",
        "nodalida",
        "paclic",
        "ranlp",
        "rocling",
        "scil",
        "semeval",
        "starsem",
        "tinlap",
        "tipster",
        "wmt",
    }
)


def clean_anthology() -> None:
    logger.info("Reading %s...", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    n_initial = len(df)

    df = df[df["year"].astype(int).between(YEAR_FROM, YEAR_TO)]
    logger.info("After year filter [%d–%d]: %d papers", YEAR_FROM, YEAR_TO, len(df))

    df = df[df["venues"].apply(lambda vs: bool(set(vs) & VENUES))]
    logger.info(
        "After venue filter: %d papers (dropped %d)", len(df), n_initial - len(df)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Saved → %s", OUTPUT_PATH)

    df_abstracts = df[df["abstract"].notna() & (df["abstract"].str.strip() != "")]
    logger.info(
        "After abstract filter: %d papers (dropped %d)",
        len(df_abstracts),
        len(df) - len(df_abstracts),
    )
    OUTPUT_WITH_ABSTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_abstracts.to_parquet(OUTPUT_WITH_ABSTRACT_PATH, index=False)
    logger.info("Saved → %s", OUTPUT_WITH_ABSTRACT_PATH)


if __name__ == "__main__":
    setup_logging()
    clean_anthology()
