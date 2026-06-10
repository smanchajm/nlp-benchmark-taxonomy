"""Build or refresh the single-task benchmark paper corpus parquet.

The clustering set (1357 papers) is defined in notebook 8: non-multitask,
single-record rows from ``mistral_extraction_full``. Task labels come from
``clustering_set_mistral_leaf_assignments.parquet``; ``thematic_domain`` from
``clustering_set_mistral_thematic_domains.parquet`` (batch thematic_domain_assignment).
Legacy extraction ``domain`` / ``domain_raw`` are kept for audit only.
``benchmark_languages`` / ``language_evidence`` come from batch ``language_assignment``
(notebook 9); ``language`` remains ACL paper metadata.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logging_config import setup_logging
from paths import DATA

logger = logging.getLogger(__name__)

DEFAULT_TAXONOMY = (
    DATA / "taxonomy" / "leaf_assignments" / "clustering_set_mistral_leaf_assignments.parquet"
)
DEFAULT_THEMATIC_DOMAINS = (
    DATA
    / "taxonomy"
    / "domain_assignments"
    / "clustering_set_mistral_thematic_domains.parquet"
)
DEFAULT_ANTHOLOGY = DATA / "corpus" / "anthology_enriched_with_bucket.parquet"
DEFAULT_OUTPUT = DATA / "corpus" / "single_task_benchmark_paper.parquet"
DEFAULT_RESOURCE_LINKS = DATA / "corpus" / "resource_links.parquet"
DEFAULT_LANGUAGE_EXTRACTIONS = DATA / "corpus" / "language_extractions.parquet"

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
    base = url.rstrip("/")
    return f"{base}.pdf" if base else None


def load_taxonomy(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    rename = {}
    if "mistral_mother" in df.columns:
        rename["mistral_mother"] = "mother_task"
    if "mistral_leaf" in df.columns:
        rename["mistral_leaf"] = "task"
    if rename:
        df = df.rename(columns=rename)
    missing = set(TAXONOMY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes taxonomie manquantes dans {path}: {sorted(missing)}")
    return df[list(TAXONOMY_COLUMNS)].drop_duplicates("bibkey")


def load_anthology(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = set(ANTHOLOGY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes anthology manquantes dans {path}: {sorted(missing)}")
    out = df[list(ANTHOLOGY_COLUMNS)].drop_duplicates("bibkey").rename(columns={"id": "anthology_id"})
    out["pdf_url"] = out["url"].map(_pdf_url_from_acl_url)
    return out


def build_corpus(taxonomy_path: Path, anthology_path: Path) -> pd.DataFrame:
    taxonomy = load_taxonomy(taxonomy_path)
    anthology = load_anthology(anthology_path)
    merged = taxonomy.merge(anthology, on="bibkey", how="left", validate="one_to_one")
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
        "venue_type",
        "numcitedby",
        "mother_task",
        "task",
        "paraphrase_task",
        "paraphrase_domain",
        *DOMAIN_COLUMNS,
        "pdf_url",
    ]
    return merged[column_order]


def add_domain_to_existing(corpus_path: Path, taxonomy_path: Path) -> pd.DataFrame:
    corpus = pd.read_parquet(corpus_path)
    if "bibkey" not in corpus.columns:
        raise ValueError(f"Colonne bibkey absente dans {corpus_path}")

    taxonomy = load_taxonomy(taxonomy_path)[["bibkey", *DOMAIN_COLUMNS]]
    base_cols = [c for c in corpus.columns if c not in DOMAIN_COLUMNS]
    enriched = corpus[base_cols].merge(
        taxonomy, on="bibkey", how="left", validate="many_to_one"
    )
    if len(enriched) != len(corpus):
        raise ValueError("Merge a duplique des lignes — verifier les bibkeys taxonomie.")

    if "paraphrase_domain" in base_cols:
        pos = base_cols.index("paraphrase_domain") + 1
        ordered = base_cols[:pos] + list(DOMAIN_COLUMNS) + base_cols[pos:]
        return enriched[ordered]
    return enriched


def load_thematic_domains(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "mistral_thematic_domain" in df.columns:
        source_col = "mistral_thematic_domain"
    elif "thematic_domain" in df.columns:
        source_col = "thematic_domain"
    else:
        raise ValueError(
            f"Colonne thematic_domain absente dans {path}: {sorted(df.columns)}"
        )
    return (
        df[["bibkey", source_col]]
        .rename(columns={source_col: "thematic_domain"})
        .drop_duplicates("bibkey")
    )


def merge_thematic_domain(df: pd.DataFrame, domains_path: Path) -> pd.DataFrame:
    if "bibkey" not in df.columns:
        raise ValueError("Colonne bibkey absente dans le corpus.")

    domains = load_thematic_domains(domains_path)
    base_cols = [c for c in df.columns if c != "thematic_domain"]
    enriched = df[base_cols].merge(
        domains, on="bibkey", how="left", validate="many_to_one"
    )
    if len(enriched) != len(df):
        raise ValueError("Merge a duplique des lignes — verifier les bibkeys domaines.")

    if "domain_raw" in base_cols:
        pos = base_cols.index("domain_raw") + 1
        ordered = base_cols[:pos] + ["thematic_domain"] + base_cols[pos:]
        return enriched[ordered]
    if "paraphrase_domain" in base_cols:
        pos = base_cols.index("paraphrase_domain") + 1
        ordered = base_cols[:pos] + ["thematic_domain"] + base_cols[pos:]
        return enriched[ordered]
    return enriched


def load_resource_links(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    cols = ["anthology_id", *RESOURCE_LINK_COLUMNS]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes resource_links manquantes dans {path}: {sorted(missing)}")
    return df[list(cols)].drop_duplicates("anthology_id")


def load_language_extractions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    cols = ["bibkey", *LANGUAGE_COLUMNS]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes language_extractions manquantes dans {path}: {sorted(missing)}"
        )
    return df[list(cols)].drop_duplicates("bibkey")


def merge_language_extractions(df: pd.DataFrame, languages_path: Path) -> pd.DataFrame:
    if "bibkey" not in df.columns:
        raise ValueError("Colonne bibkey absente dans le corpus.")

    languages = load_language_extractions(languages_path)
    base_cols = [c for c in df.columns if c not in LANGUAGE_COLUMNS]
    enriched = df[base_cols].merge(
        languages, on="bibkey", how="left", validate="many_to_one"
    )
    if len(enriched) != len(df):
        raise ValueError("Merge a duplique des lignes — verifier les bibkeys langues.")

    if "language" in base_cols:
        pos = base_cols.index("language") + 1
        ordered = base_cols[:pos] + list(LANGUAGE_COLUMNS) + base_cols[pos:]
        return enriched[ordered]
    return enriched


def merge_resource_links(df: pd.DataFrame, links_path: Path) -> pd.DataFrame:
    if "anthology_id" not in df.columns:
        raise ValueError("Colonne anthology_id absente dans le corpus.")

    links = load_resource_links(links_path)
    base_cols = [c for c in df.columns if c not in RESOURCE_LINK_COLUMNS]
    enriched = df[base_cols].merge(
        links, on="anthology_id", how="left", validate="many_to_one"
    )
    if len(enriched) != len(df):
        raise ValueError("Merge a duplique des lignes — verifier les anthology_id.")

    if "pdf_url" in base_cols:
        pos = base_cols.index("pdf_url") + 1
        ordered = base_cols[:pos] + list(RESOURCE_LINK_COLUMNS) + base_cols[pos:]
        return enriched[ordered]
    return enriched


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    n_domain = int(df["domain"].notna().sum()) if "domain" in df.columns else 0
    n_thematic = (
        int(df["thematic_domain"].notna().sum()) if "thematic_domain" in df.columns else 0
    )
    n_dataset = int(df["dataset_url"].notna().sum()) if "dataset_url" in df.columns else 0
    n_code = int(df["code_url"].notna().sum()) if "code_url" in df.columns else 0
    n_benchmark_lang = (
        int(df["benchmark_languages"].notna().sum())
        if "benchmark_languages" in df.columns
        else 0
    )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument(
        "--thematic-domains",
        type=Path,
        default=DEFAULT_THEMATIC_DOMAINS,
        help="Parquet Mistral thematic_domain_assignment (batch results).",
    )
    parser.add_argument(
        "--skip-thematic-domain",
        action="store_true",
        help="Ne pas fusionner thematic_domain dans le corpus.",
    )
    parser.add_argument(
        "--resource-links",
        type=Path,
        default=DEFAULT_RESOURCE_LINKS,
        help="Parquet link_arbiter (batch results) a fusionner dans le corpus.",
    )
    parser.add_argument(
        "--skip-resource-links",
        action="store_true",
        help="Ne pas fusionner dataset_url / code_url dans le corpus.",
    )
    parser.add_argument(
        "--language-extractions",
        type=Path,
        default=DEFAULT_LANGUAGE_EXTRACTIONS,
        help="Parquet language_assignment (batch results) a fusionner dans le corpus.",
    )
    parser.add_argument(
        "--skip-language-extractions",
        action="store_true",
        help="Ne pas fusionner benchmark_languages dans le corpus.",
    )
    parser.add_argument("--anthology", type=Path, default=DEFAULT_ANTHOLOGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstruire tout le corpus depuis taxonomie + anthology.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Parquet existant a enrichir (defaut: --output). Ignore si --rebuild.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebuild:
        df = build_corpus(args.taxonomy, args.anthology)
        if "thematic_domain" in df.columns:
            df = df.drop(columns=["thematic_domain"])
    else:
        input_path = args.input or args.output
        if not input_path.exists():
            raise FileNotFoundError(
                f"{input_path} introuvable. Utiliser --rebuild ou fournir --input."
            )
        df = add_domain_to_existing(input_path, args.taxonomy)

    if not args.skip_thematic_domain:
        if not args.thematic_domains.exists():
            raise FileNotFoundError(
                f"{args.thematic_domains} introuvable. "
                "Lancer le batch thematic_domain_assignment ou passer --skip-thematic-domain."
            )
        df = merge_thematic_domain(df, args.thematic_domains)

    if not args.skip_resource_links:
        if not args.resource_links.exists():
            raise FileNotFoundError(
                f"{args.resource_links} introuvable. "
                "Lancer le batch link_arbiter ou passer --skip-resource-links."
            )
        df = merge_resource_links(df, args.resource_links)

    if not args.skip_language_extractions:
        if not args.language_extractions.exists():
            raise FileNotFoundError(
                f"{args.language_extractions} introuvable. "
                "Lancer le batch language_assignment ou passer --skip-language-extractions."
            )
        df = merge_language_extractions(df, args.language_extractions)

    write_output(df, args.output)


if __name__ == "__main__":
    setup_logging()
    main()
