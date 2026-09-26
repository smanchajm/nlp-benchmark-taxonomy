"""Résolution ``bibkey -> arxiv_id``, entrée de la borne basse de couverture.

Deux canaux Semantic Scholar, complémentaires et unis. Mesuré sur les 1 312
papiers du corpus (cf. caches existants, aucune requête nécessaire) :

===================================  =======  ======
canal                                résolus  taux
===================================  =======  ======
clé exacte ``DOI:`` / ``ACL:``           336   25,6 %
recherche titre + année                  500   38,1 %
**union**                            **583**  **44,4 %**
===================================  =======  ======

Les deux canaux sont complémentaires parce qu'ils visent des enregistrements S2
différents : la clé exacte résout la *publication ACL*, dont les ``externalIds``
ne portent souvent aucun ``ArXiv`` ; la recherche par titre trouve le *préprint
arXiv jumeau*, enregistrement distinct qui lui porte l'identifiant. Sur les 253
papiers résolus par les deux canaux, les identifiants concordent à 100 %.

Remplace ``extract_benchmark_links.py`` (canal recherche seul) et
``s2_arxiv_resolution.py`` (canal exact seul). Le repli « API arXiv brute » de
ces implémentations a été retiré : 7 résolutions pour 1 404 requêtes.

Chaque canal est caché sur disque, donc une reprise ne requête que les papiers
absents du cache. Sans ``--query-missing``, le module travaille hors ligne.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ``norm_title`` vient de ``hub_coverage`` : une seule définition des clés de
# matching, car elle sert aussi au join PwC par titre.
from coverage.hub_coverage import norm_title
from logging_config import setup_logging
from paths import DATA

logger = logging.getLogger(__name__)

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

COVERAGE_DIR = DATA / "coverage"
DEFAULT_INPUT = DATA / "corpus" / "single_task_benchmark_paper.parquet"
DEFAULT_OUTPUT = COVERAGE_DIR / "arxiv_resolution.parquet"
EXACT_CACHE = COVERAGE_DIR / "s2_exact_cache.parquet"
SEARCH_CACHE = COVERAGE_DIR / "s2_search_cache.parquet"

EXACT_CACHE_COLUMNS = ("s2_query_id", "s2_found", "corpus_paper_id", "arxiv_id")
SEARCH_CACHE_COLUMNS = ("title_norm", "year", "arxiv_id")

# S2 tolère ~1 req/s sans clé ; le batch accepte 500 ids par appel.
MIN_REQUEST_INTERVAL = 1.1
BATCH_SIZE = 500
MAX_RETRIES = 6


# --- HTTP --------------------------------------------------------------------


def _request(
    method: str, url: str, *, api_key: str = "", **kwargs: object
) -> requests.Response | None:
    """Requête S2 avec backoff. ``None`` si l'appel échoue après tous les essais."""
    headers = {"x-api-key": api_key} if api_key else {}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.request(method, url, headers=headers, timeout=60, **kwargs)
        except requests.RequestException as err:
            logger.warning("S2 erreur réseau (%s) — essai %d", err, attempt + 1)
            resp = None
        else:
            if resp.status_code == 200:
                return resp
            logger.warning("S2 HTTP %s — essai %d", resp.status_code, attempt + 1)

        if attempt == MAX_RETRIES - 1:
            return None

        retry_after = resp.headers.get("Retry-After") if resp is not None else None
        if retry_after is not None:
            try:
                wait = max(5.0, float(retry_after))
            except ValueError:
                wait = 20.0 + 10.0 * attempt
        else:
            wait = min(60.0, 5.0 * (2**attempt))
        time.sleep(wait)
    return None


# --- cache -------------------------------------------------------------------


def load_cache(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


def save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Cache écrit: %s (%d entrées)", path, len(df))


# --- clés de requête ---------------------------------------------------------


def build_query_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute ``s2_query_id`` (``DOI:`` prioritaire, sinon ``ACL:``) et sa source."""
    out = df.copy()
    doi = out["doi"].where(out["doi"].notna() & out["doi"].astype(str).str.strip().ne(""))
    aid = out["anthology_id"].where(
        out["anthology_id"].notna() & out["anthology_id"].astype(str).str.strip().ne("")
    )
    out["s2_query_id"] = doi.radd("DOI:").fillna(aid.radd("ACL:"))
    out["s2_id_source"] = doi.notna().map({True: "DOI", False: "ACL"})
    out.loc[out["s2_query_id"].isna(), "s2_id_source"] = pd.NA
    out["title_norm"] = out["title"].map(norm_title)

    logger.info(
        "Clés construites: %d DOI + %d ACL, %d papiers sans clé",
        int((out["s2_id_source"] == "DOI").sum()),
        int((out["s2_id_source"] == "ACL").sum()),
        int(out["s2_query_id"].isna().sum()),
    )
    return out


# --- canal 1 : clé exacte ----------------------------------------------------


def query_exact(ids: list[str], api_key: str = "") -> pd.DataFrame:
    """Résout des clés ``DOI:``/``ACL:`` en ``arxiv_id`` via le batch S2."""
    rows: list[dict] = []
    for start in range(0, len(ids), BATCH_SIZE):
        chunk = ids[start : start + BATCH_SIZE]
        resp = _request(
            "POST",
            S2_BATCH_URL,
            api_key=api_key,
            params={"fields": "externalIds"},
            json={"ids": chunk},
        )
        if resp is None:
            logger.error("Batch %d abandonné après retries — clés non résolues.", start)
            continue
        for query_id, record in zip(chunk, resp.json(), strict=False):
            ext = (record or {}).get("externalIds") or {}
            rows.append(
                {
                    "s2_query_id": query_id,
                    "s2_found": record is not None,
                    "corpus_paper_id": ext.get("CorpusId"),
                    "arxiv_id": ext.get("ArXiv"),
                }
            )
        logger.info("Batch exact %d-%d résolu", start, start + len(chunk))
        time.sleep(MIN_REQUEST_INTERVAL)
    return pd.DataFrame(rows, columns=list(EXACT_CACHE_COLUMNS))


def resolve_exact(
    df: pd.DataFrame,
    *,
    api_key: str = "",
    cache_file: Path = EXACT_CACHE,
    query_missing: bool = False,
) -> pd.DataFrame:
    cache = load_cache(cache_file, EXACT_CACHE_COLUMNS)
    todo = sorted(set(df["s2_query_id"].dropna()) - set(cache["s2_query_id"]))
    logger.info("Canal exact: %d en cache, %d à requêter", len(cache), len(todo))

    if todo and query_missing:
        cache = pd.concat([cache, query_exact(todo, api_key)], ignore_index=True)
        cache = cache.drop_duplicates("s2_query_id", keep="last")
        save_cache(cache, cache_file)
    elif todo:
        logger.info("--query-missing absent: les %d clés restent non résolues.", len(todo))

    out = df.merge(
        cache.rename(columns={"arxiv_id": "arxiv_exact"}), on="s2_query_id", how="left"
    )
    out["s2_found"] = out["s2_found"].fillna(False)
    return out


# --- canal 2 : recherche titre + année ---------------------------------------


def query_search(title: str, year: int, api_key: str = "") -> str | None:
    """Cherche le préprint arXiv par titre ; exige titre normalisé égal et année ±1."""
    resp = _request(
        "GET",
        S2_SEARCH_URL,
        api_key=api_key,
        params={"query": title, "limit": 10, "fields": "externalIds,title,year"},
    )
    if resp is None:
        return None

    target = norm_title(title)
    for record in (resp.json() or {}).get("data") or []:
        rec_year = record.get("year")
        if norm_title(record.get("title")) != target:
            continue
        if not isinstance(rec_year, int) or abs(rec_year - int(year)) > 1:
            continue
        arxiv_id = ((record.get("externalIds") or {}).get("ArXiv")) or None
        if arxiv_id:
            return arxiv_id
    return None


def resolve_search(
    df: pd.DataFrame,
    *,
    api_key: str = "",
    cache_file: Path = SEARCH_CACHE,
    query_missing: bool = False,
    retry_misses: bool = False,
) -> pd.DataFrame:
    """Complète par recherche titre+année. Les échecs sont cachés (cf. ``retry_misses``)."""
    cache = load_cache(cache_file, SEARCH_CACHE_COLUMNS)
    candidates = (
        df.loc[df["title_norm"].notna() & df["year"].notna(), ["title", "title_norm", "year"]]
        .assign(year=lambda x: x["year"].astype(int))
        .drop_duplicates(["title_norm", "year"])
    )

    known = cache if retry_misses is False else cache[cache["arxiv_id"].notna()]
    seen = set(zip(known["title_norm"], known["year"], strict=False))
    todo = [c for c in candidates.itertuples(index=False) if (c.title_norm, c.year) not in seen]
    logger.info("Canal recherche: %d en cache, %d à requêter", len(cache), len(todo))

    if todo and query_missing:
        rows = []
        for i, cand in enumerate(todo, start=1):
            rows.append(
                {
                    "title_norm": cand.title_norm,
                    "year": cand.year,
                    "arxiv_id": query_search(cand.title, cand.year, api_key),
                }
            )
            if i % 50 == 0:
                logger.info("Recherche S2: %d/%d titres", i, len(todo))
            time.sleep(MIN_REQUEST_INTERVAL)
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache = cache.drop_duplicates(["title_norm", "year"], keep="last")
        save_cache(cache, cache_file)
    elif todo:
        logger.info("--query-missing absent: les %d titres restent non cherchés.", len(todo))

    lookup = cache.dropna(subset=["arxiv_id"]).set_index(["title_norm", "year"])["arxiv_id"]
    lookup = lookup.to_dict()
    out = df.copy()
    out["arxiv_search"] = [
        lookup.get((t, int(y))) if pd.notna(t) and pd.notna(y) else None
        for t, y in zip(out["title_norm"], out["year"], strict=False)
    ]
    return out


# --- union -------------------------------------------------------------------


def union_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Unit les deux canaux et trace la provenance dans ``arxiv_source``."""
    out = df.copy()
    exact = out["arxiv_exact"]
    search = out["arxiv_search"]

    disagree = int((exact.notna() & search.notna() & (exact.str.lower() != search.str.lower())).sum())
    if disagree:
        logger.warning("%d papiers où les deux canaux donnent un arxiv_id différent.", disagree)

    out["arxiv_id"] = exact.fillna(search)
    out["arxiv_source"] = pd.NA
    out.loc[search.notna(), "arxiv_source"] = "s2_search_title_year"
    out.loc[exact.notna(), "arxiv_source"] = "s2_exact_key"
    out.loc[exact.notna() & search.notna(), "arxiv_source"] = "both"

    n = len(out)
    logger.info(
        "Résolution: exact=%d (%.1f%%) | recherche=%d (%.1f%%) | union=%d (%.1f%%)",
        int(exact.notna().sum()), 100 * exact.notna().mean(),
        int(search.notna().sum()), 100 * search.notna().mean(),
        int(out["arxiv_id"].notna().sum()), 100 * out["arxiv_id"].notna().mean(),
    )
    return out.drop(columns=["arxiv_exact", "arxiv_search", "title_norm"])


# --- pipeline ----------------------------------------------------------------


def run_resolution(
    input_path: Path = DEFAULT_INPUT,
    *,
    api_key: str = "",
    query_missing: bool = False,
    retry_misses: bool = False,
) -> pd.DataFrame:
    """Charge le corpus, résout par les deux canaux, renvoie le corpus enrichi."""
    df = pd.read_parquet(input_path)
    logger.info("Corpus: %s (%d papiers)", input_path, len(df))

    df = build_query_keys(df)
    df = resolve_exact(df, api_key=api_key, query_missing=query_missing)
    df = resolve_search(
        df, api_key=api_key, query_missing=query_missing, retry_misses=retry_misses
    )
    return union_channels(df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Résout les arxiv_id du corpus (clé exacte S2 + recherche titre/année)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--query-missing",
        action="store_true",
        help="Interroger S2 pour ce qui manque au cache (sinon: hors ligne).",
    )
    parser.add_argument(
        "--retry-misses",
        action="store_true",
        help="Re-interroger aussi les titres déjà cherchés sans succès.",
    )
    parser.add_argument("--api-key", default="", help="Clé S2 (sinon mode non authentifié).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_resolution(
        args.input,
        api_key=args.api_key,
        query_missing=args.query_missing,
        retry_misses=args.retry_misses,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    logger.info("Corpus résolu -> %s (%d lignes)", args.output, len(df))


if __name__ == "__main__":
    setup_logging()
    main()
