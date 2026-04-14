import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from logging_config import setup_logging
from paths import DATA

logger = logging.getLogger(__name__)

DEFAULT_INPUT = DATA / "anthology.parquet"
DEFAULT_OUTPUT = DATA / "anthology_filtered.parquet"
DEFAULT_OUTPUT_WITH_ABSTRACT = DATA / "anthology_filtered_with_abstract.parquet"

YEAR_FROM = 2013  # 2013 is the year of word2vec publication, often considered the start of the modern era of NLP.
YEAR_TO = datetime.now().year + 1

# All venue IDs from the ACL Anthology ("ws" generic wrapper excluded).
# Omitted — no papers post-2013: anlp, hlt, muc, tinlap, tipster.

ACL_VENUES: frozenset[str] = frozenset(
    {
        "aacl",
        "acl",
        "arabicnlp",
        "conll",
        "eacl",
        "emnlp",
        "findings",
        "iwslt",
        "naacl",
        "semeval",
        "starsem",
        "wmt",
    }
)

NON_ACL_VENUES: frozenset[str] = frozenset(
    {
        "aimecon",
        "alta",
        "amta",
        "ccl",
        "clicit",
        "coling",
        "eamt",
        "ijcnlp",
        "iwsds",
        "jeptalnrecital",
        "konvens",
        "lrec",
        "mtsummit",
        "nodalida",
        "paclic",
        "ranlp",
        "rocling",
        "scil",
    }
)

ACL_JOURNALS: frozenset[str] = frozenset({"cl", "tacl"})

NON_ACL_JOURNALS: frozenset[str] = frozenset(
    {"ijclclp", "jlcl", "lilt", "nejlt", "tal"}
)

VENUES: frozenset[str] = ACL_VENUES | NON_ACL_VENUES | ACL_JOURNALS

_VENUE_CATEGORY: dict[str, str] = {
    **{v: "acl_venue" for v in ACL_VENUES},
    **{v: "non_acl_venue" for v in NON_ACL_VENUES},
    **{v: "acl_journal" for v in ACL_JOURNALS},
    **{v: "non_acl_journal" for v in NON_ACL_JOURNALS},
}


def _get_venue_category(venues: list[str]) -> str:
    for v in venues:
        if v in _VENUE_CATEGORY:
            return _VENUE_CATEGORY[v]
    return "unknown"


def clean_anthology(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    output_with_abstract_path: Path = DEFAULT_OUTPUT_WITH_ABSTRACT,
) -> None:
    logger.info("Reading %s...", input_path)
    df = pd.read_parquet(input_path)
    n_initial = len(df)

    df = df[df["year"].astype(int).between(YEAR_FROM, YEAR_TO)]
    logger.info("After year filter [%d–%d]: %d papers", YEAR_FROM, YEAR_TO, len(df))

    df = df[df["venues"].apply(lambda vs: bool(set(vs) & VENUES))]
    logger.info(
        "After venue filter: %d papers (dropped %d)", len(df), n_initial - len(df)
    )

    df["venue_type"] = df["venues"].apply(_get_venue_category)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved → %s", output_path)

    df_abstracts = df[df["abstract"].notna() & (df["abstract"].str.strip() != "")]
    logger.info(
        "After abstract filter: %d papers (dropped %d)",
        len(df_abstracts),
        len(df) - len(df_abstracts),
    )
    output_with_abstract_path.parent.mkdir(parents=True, exist_ok=True)
    df_abstracts.to_parquet(output_with_abstract_path, index=False)
    logger.info("Saved → %s", output_with_abstract_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter the ACL Anthology corpus")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--output-with-abstract", type=Path, default=DEFAULT_OUTPUT_WITH_ABSTRACT
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    clean_anthology(args.input, args.output, args.output_with_abstract)
