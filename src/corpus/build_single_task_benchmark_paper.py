"""Build single_task_benchmark_paper parquet (simple, no CLI).

This script always rebuilds the corpus from source parquets, then overwrites
the output parquet. If output already exists, it logs added/removed/updated
bibkeys before writing.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logging_config import setup_logging
from paths import DATA

logger = logging.getLogger(__name__)


# ---- Paths (edit here if needed) -------------------------------------------------

TAXONOMY_PATH = (
    DATA
    / "taxonomy"
    / "leaf_assignments"
    / "clustering_set_mistral_leaf_assignments.parquet"
)
EXCLUSIONS_PATH = DATA / "taxonomy" / "manual_exclusions.parquet"
THEMATIC_DOMAINS_PATH = (
    DATA
    / "taxonomy"
    / "domain_assignments"
    / "clustering_set_mistral_thematic_domains.parquet"
)
ANTHOLOGY_PATH = DATA / "corpus" / "anthology_enriched_with_bucket.parquet"
RESOURCE_LINKS_PATH = DATA / "extraction" / "resource_links.parquet"
LANGUAGE_EXTRACTIONS_PATH = DATA / "extraction" / "language_extractions.parquet"
OUTPUT_PATH = DATA / "corpus" / "single_task_benchmark_paper.parquet"


# ---- Columns --------------------------------------------------------------------

DOMAIN_COLUMNS = ("domain", "domain_raw")
RESOURCE_LINK_COLUMNS = (
    "dataset_url",
    "dataset_host_type",
    "code_url",
    "code_host_type",
)
LANGUAGE_COLUMNS = (
    "benchmark_languages",
    "language_evidence",
    "language_reasoning",
)
TAXONOMY_COLUMNS = (
    "bibkey",
    "mother_task",
    "task",
    "paraphrase_task",
    "paraphrase_domain",
    *DOMAIN_COLUMNS,
    "title",
    "abstract",
)
ANTHOLOGY_COLUMNS = (
    "bibkey",
    "id",
    "authors",
    "year",
    "venues",
    "doi",
    "url",
    "language",
    "venue_type",
    "numcitedby",
)


def _pdf_url_from_acl_url(url: str | None) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None
    return f"{url.rstrip('/')}.pdf"


def _require_columns(df: pd.DataFrame, needed: tuple[str, ...] | list[str], name: str) -> None:
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {name}: {sorted(missing)}")


def load_taxonomy(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    rename = {}
    if "mistral_mother" in df.columns:
        rename["mistral_mother"] = "mother_task"
    if "mistral_leaf" in df.columns:
        rename["mistral_leaf"] = "task"
    if rename:
        df = df.rename(columns=rename)
    _require_columns(df, TAXONOMY_COLUMNS, str(path))
    return df[list(TAXONOMY_COLUMNS)].drop_duplicates("bibkey")


def load_anthology(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    _require_columns(df, ANTHOLOGY_COLUMNS, str(path))
    out = (
        df[list(ANTHOLOGY_COLUMNS)]
        .drop_duplicates("bibkey")
        .rename(columns={"id": "anthology_id"})
    )
    out["pdf_url"] = out["url"].map(_pdf_url_from_acl_url)
    return out


def load_thematic_domains(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    source_col = (
        "mistral_thematic_domain"
        if "mistral_thematic_domain" in df.columns
        else "thematic_domain"
    )
    _require_columns(df, ["bibkey", source_col], str(path))
    return (
        df[["bibkey", source_col]]
        .rename(columns={source_col: "thematic_domain"})
        .drop_duplicates("bibkey")
    )


def load_resource_links(path: Path) -> pd.DataFrame:
    cols = ["anthology_id", *RESOURCE_LINK_COLUMNS]
    df = pd.read_parquet(path)
    _require_columns(df, cols, str(path))
    return df[cols].drop_duplicates("anthology_id")


def load_language_extractions(path: Path) -> pd.DataFrame:
    cols = ["bibkey", *LANGUAGE_COLUMNS]
    df = pd.read_parquet(path)
    _require_columns(df, cols, str(path))
    return df[cols].drop_duplicates("bibkey")


def load_exclusions(path: Path) -> set[str]:
    df = pd.read_parquet(path)
    _require_columns(df, ["bibkey"], str(path))
    return set(df["bibkey"].dropna().astype(str))


def build() -> pd.DataFrame:
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    anthology = load_anthology(ANTHOLOGY_PATH)

    df = taxonomy.merge(anthology, on="bibkey", how="left", validate="one_to_one")

    thematic = load_thematic_domains(THEMATIC_DOMAINS_PATH)
    df = df.merge(thematic, on="bibkey", how="left", validate="many_to_one")

    links = load_resource_links(RESOURCE_LINKS_PATH)
    df = df.merge(links, on="anthology_id", how="left", validate="many_to_one")

    languages = load_language_extractions(LANGUAGE_EXTRACTIONS_PATH)
    df = df.merge(languages, on="bibkey", how="left", validate="many_to_one")

    exclusions = load_exclusions(EXCLUSIONS_PATH)
    before = len(df)
    df = df.loc[~df["bibkey"].astype(str).isin(exclusions)].copy()
    logger.info(
        "Exclusions: %d lignes retirees (%d -> %d).",
        before - len(df),
        before,
        len(df),
    )

    column_order = [
        "bibkey",
        "anthology_id",
        "title",
        "abstract",
        "authors",
        "year",
        "venues",
        "doi",
        "url",
        "language",
        *LANGUAGE_COLUMNS,
        "venue_type",
        "numcitedby",
        "mother_task",
        "task",
        "paraphrase_task",
        "paraphrase_domain",
        *DOMAIN_COLUMNS,
        "thematic_domain",
        "pdf_url",
        *RESOURCE_LINK_COLUMNS,
    ]
    return df[column_order]


def log_diff_vs_existing(new_df: pd.DataFrame, output_path: Path) -> None:
    if not output_path.exists():
        logger.info("Nouveau fichier: %s", output_path)
        return

    old_df = pd.read_parquet(output_path)
    if "bibkey" not in old_df.columns:
        logger.warning("Ancien fichier sans bibkey, diff impossible.")
        return

    old = old_df.set_index("bibkey", drop=False)
    new = new_df.set_index("bibkey", drop=False)

    old_keys = set(old.index)
    new_keys = set(new.index)
    added = new_keys - old_keys
    removed = old_keys - new_keys

    shared = sorted(old_keys & new_keys)
    common_cols = [c for c in new.columns if c in old.columns and c != "bibkey"]
    changed = 0
    for key in shared:
        left = old.loc[key, common_cols]
        right = new.loc[key, common_cols]
        if not left.equals(right):
            changed += 1

    logger.info(
        "Diff vs existant - added: %d, removed: %d, updated: %d",
        len(added),
        len(removed),
        changed,
    )


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_diff_vs_existing(df, output_path)
    df.to_parquet(output_path, index=False)

    n_domain = int(df["domain"].notna().sum())
    n_thematic = int(df["thematic_domain"].notna().sum())
    n_dataset = int(df["dataset_url"].notna().sum())
    n_code = int(df["code_url"].notna().sum())
    n_benchmark_lang = int(df["benchmark_languages"].notna().sum())
    logger.info(
        "Ecrit %s (%d lignes, domain: %d, thematic_domain: %d, dataset_url: %d, "
        "code_url: %d, benchmark_languages: %d)",
        output_path,
        len(df),
        n_domain,
        n_thematic,
        n_dataset,
        n_code,
        n_benchmark_lang,
    )


def main() -> None:
    df = build()
    write_output(df, OUTPUT_PATH)


if __name__ == "__main__":
    setup_logging()
    main()
