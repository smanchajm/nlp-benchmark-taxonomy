"""Résolution ``bibkey -> arxiv_id`` (Semantic Scholar puis arXiv).

Déjà figée dans ``single_task_benchmark_paper_enriched_links.parquet`` ; on ne
la rejoue (``RUN_S2_ARXIV = True``) que pour la reconstruire. Étapes cachées sur
disque (résumables). Code de recherche : garde-fous minimaux.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

from coverage.hub_coverage import (
    ANTHOLOGY_ENRICHED_FILE,
    ENRICHED_LINKS_FILE,
    MERGED_FILE,
    PWC_DIR,
    norm_arxiv,
    norm_title,
)

logger = logging.getLogger(__name__)

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
S2_CACHE_FILE = PWC_DIR / "s2_resolution.parquet"
ARXIV_CACHE_FILE = PWC_DIR / "arxiv_resolution.parquet"


def build_s2_query_ids(merged_file: Path = MERGED_FILE, anthology_file: Path = ANTHOLOGY_ENRICHED_FILE) -> pd.DataFrame:
    """Corpus POSITIVE -> ``s2_query_id`` (``DOI:`` prioritaire, sinon ``ACL:``)."""
    benchmarks = pd.read_parquet(merged_file).query("majority_label == 'POSITIVE'")
    anthology = pd.read_parquet(anthology_file)[["bibkey", "id", "doi", "title", "year"]]
    df = benchmarks[["bibkey"]].merge(anthology, on="bibkey", how="left").rename(columns={"id": "anthology_id"})

    doi = df["doi"].where(df["doi"].astype(str).str.strip().ne(""))
    aid = df["anthology_id"].where(df["anthology_id"].astype(str).str.strip().ne(""))
    df["s2_query_id"] = doi.radd("DOI:").fillna(aid.radd("ACL:"))
    df["s2_id_source"] = doi.notna().map({True: "DOI", False: "ACL"})
    logger.info("s2_query_id construits: %d benchmarks POSITIVE", len(df))
    return df


def query_s2_batch(ids: list[str], api_key: str = "", batch_size: int = 500) -> pd.DataFrame:
    """Résout des clés S2 en ``arxiv_id`` + ``corpus_paper_id`` (1 retry sur erreur)."""
    headers = {"x-api-key": api_key} if api_key else {}
    rows: list[dict] = []
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        resp = requests.post(S2_BATCH_URL, params={"fields": "externalIds"}, json={"ids": chunk}, headers=headers, timeout=60)
        if resp.status_code != 200:
            time.sleep(10)
            resp = requests.post(S2_BATCH_URL, params={"fields": "externalIds"}, json={"ids": chunk}, headers=headers, timeout=60)
        for query_id, rec in zip(chunk, resp.json()):
            ext = (rec or {}).get("externalIds") or {}
            rows.append({"s2_query_id": query_id, "s2_found": rec is not None,
                         "corpus_paper_id": ext.get("CorpusId"), "arxiv_id": ext.get("ArXiv")})
        logger.info("S2 batch %d-%d ok", start, start + len(chunk))
        time.sleep(1.0 if api_key else 3.0)
    return pd.DataFrame(rows)


def resolve_s2(df: pd.DataFrame, api_key: str = "", cache_file: Path = S2_CACHE_FILE) -> pd.DataFrame:
    """Complète ``arxiv_id`` / ``corpus_paper_id`` / ``s2_found`` (caché)."""
    cols = ["s2_query_id", "s2_found", "corpus_paper_id", "arxiv_id"]
    s2 = pd.read_parquet(cache_file) if cache_file.exists() else pd.DataFrame(columns=cols)
    todo = sorted(set(df["s2_query_id"].dropna()) - set(s2["s2_query_id"]))
    logger.info("S2: en cache=%d | à requêter=%d", len(s2), len(todo))
    if todo:
        s2 = pd.concat([s2, query_s2_batch(todo, api_key)], ignore_index=True).drop_duplicates("s2_query_id")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        s2.to_parquet(cache_file, index=False)
    out = df.merge(s2, on="s2_query_id", how="left")
    out["s2_found"] = out["s2_found"].fillna(False)
    return out


def query_arxiv_by_title_year(title_norm: str, year: int) -> str | None:
    """arXiv : titre normalisé exact + année exacte -> arxiv id, sinon None."""
    try:
        resp = requests.get(ARXIV_API_URL, params={"search_query": f'ti:"{title_norm}"', "max_results": 8}, timeout=30)
        root = ET.fromstring(resp.text)
    except (requests.RequestException, ET.ParseError):
        return None
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", ns):
        title = entry.findtext("atom:title", "", ns)
        published = entry.findtext("atom:published", "", ns)
        url = entry.findtext("atom:id", "", ns)
        if norm_title(title) == title_norm and published[:4] == str(int(year)) and "/abs/" in url:
            return url.split("/abs/")[-1].split("?")[0] or None
    return None


def resolve_arxiv_titles(df: pd.DataFrame, cache_file: Path = ARXIV_CACHE_FILE, sleep: float = 0.2) -> pd.DataFrame:
    """Complète les ``arxiv_id`` manquants via arXiv (titre+année, caché)."""
    out = df.copy()
    out["title_norm"] = out["title"].map(norm_title)
    out["arxiv_source"] = out["arxiv_id"].notna().map({True: "s2", False: pd.NA})

    cols = ["title_norm", "year", "arxiv_id"]
    cache = pd.read_parquet(cache_file) if cache_file.exists() else pd.DataFrame(columns=cols)
    missing = out["arxiv_id"].isna()
    todo = (
        out.loc[missing, ["title_norm", "year"]].dropna().drop_duplicates().assign(year=lambda x: x["year"].astype(int))
    )
    known = set(zip(cache["title_norm"], cache["year"]))
    todo = [(t, y) for t, y in todo.itertuples(index=False) if (t, y) not in known]
    if todo:
        rows = [{"title_norm": t, "year": y, "arxiv_id": query_arxiv_by_title_year(t, y)} for t, y in _logged(todo, sleep)]
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["title_norm", "year"], keep="last")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(cache_file, index=False)

    lookup = cache.set_index(["title_norm", "year"])["arxiv_id"].to_dict()
    filled = out.loc[missing].apply(lambda r: lookup.get((r["title_norm"], int(r["year"]))) if pd.notna(r["year"]) else None, axis=1)
    out.loc[missing, "arxiv_id"] = filled
    out.loc[missing & out["arxiv_id"].notna(), "arxiv_source"] = "arxiv_api_title_year"
    logger.info("arXiv titre+année: +%d arxiv_id complétés", int(filled.notna().sum()))
    return out


def _logged(todo: list[tuple], sleep: float):
    """Itère en logguant la progression et en respectant ``sleep``."""
    for i, item in enumerate(todo, start=1):
        yield item
        if i % 50 == 0:
            logger.info("arXiv API: %d/%d titres", i, len(todo))
        time.sleep(sleep)


def run_s2_arxiv_resolution(api_key: str = "", save: bool = True, save_file: Path = ENRICHED_LINKS_FILE) -> pd.DataFrame:
    """Reconstruit le corpus enrichi (``bibkey -> arxiv_id``) from scratch."""
    df = resolve_arxiv_titles(resolve_s2(build_s2_query_ids(), api_key))
    df["arxiv_norm"] = df["arxiv_id"].map(norm_arxiv)
    if save:
        save_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_file, index=False)
        logger.info("Corpus enrichi réécrit: %d lignes -> %s", len(df), save_file)
    return df
