import logging

import pandas as pd

from logging_config import ROOT, setup_logging

logger = logging.getLogger(__name__)

INPUT_PATH = ROOT / "data/anthology.parquet"
OUTPUT_PATH = ROOT / "data/anthology_filtered.parquet"
OUTPUT_WITH_ABSTRACT_PATH = ROOT / "data/anthology_filtered_with_abstract.parquet"

YEAR_FROM = 2013  # 2013 is the year of word2vec publication, often considered the start of the modern era of NLP.
YEAR_TO = 2026

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

    df["venue_type"] = df["venues"].apply(_get_venue_category)

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
