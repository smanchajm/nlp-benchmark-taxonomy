"""Matching du corpus de benchmarks dans les hubs (PwC, HF).

Borne basse de présence : on part du corpus et on interroge les hubs. PwC =
archive figée (join local, arxiv puis titre). HF = API datée, cascade
``direct -> arxiv -> name`` (``name`` = tier bruité, hors borne ``in_hf``).
Code de recherche : peu de garde-fous, on privilégie la lisibilité.

La résolution ``bibkey -> arxiv_id`` vit dans ``s2_arxiv_resolution``.
"""

from __future__ import annotations

import ast
import logging
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --- chemins ---
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PWC_DIR = DATA_DIR / "pwc_coverage"
OUT_DIR = DATA_DIR / "hub_coverage"

ENRICHED_LINKS_FILE = (
    DATA_DIR / "corpus" / "single_task_benchmark_paper_enriched_links.parquet"
)
RESOURCE_LINKS_FILE = DATA_DIR / "extraction" / "resource_links.parquet"
PWC_PAPERS_FILE = PWC_DIR / "pwc_papers.parquet"
PWC_DATASETS_FILE = PWC_DIR / "pwc_datasets.parquet"
HF_CACHE_FILE = OUT_DIR / "hf_resolution.parquet"
COVERAGE_OUT_FILE = OUT_DIR / "benchmark_hub_coverage.parquet"

# sources de `s2_arxiv_resolution`
MERGED_FILE = DATA_DIR / "taxonomy" / "merged.parquet"
ANTHOLOGY_ENRICHED_FILE = DATA_DIR / "raw" / "anthology_enriched.parquet"

PWC_ARCHIVE_DATE = "2025-07-28"
HF_COLS = [
    "bibkey",
    "in_hf",
    "hf_match_method",
    "hf_dataset_id",
    "hf_namesearch_candidates",
    "hf_query_date",
]


# --- normalisation ---


def norm_arxiv(x: object) -> str | None:
    """``2305.12345v3`` -> ``2305.12345`` ; sinon None."""
    return x.strip().lower().split("v")[0] if isinstance(x, str) and x.strip() else None


def norm_title(x: object) -> str | None:
    """Titre -> minuscules alphanumériques séparées par espaces (ASCII)."""
    if not isinstance(x, str):
        return None
    txt = re.sub(r"[^a-z0-9]+", " ", x.encode("ascii", "ignore").decode().lower())
    return " ".join(txt.split()) or None


def hf_dataset_id(urls: object) -> str | None:
    """Premier ``owner/name`` d'un lien ``huggingface.co/datasets/...`` trouvé."""
    if isinstance(urls, str):
        items: list = [urls]
    else:
        try:
            items = list(urls)  # array/list ; float NaN -> TypeError -> []
        except TypeError:
            items = []
    for u in items:
        m = re.search(r"huggingface\.co/datasets/([^/\s?#]+(?:/[^/\s?#]+)?)", str(u))
        if m:
            return m.group(1)
    return None


# --- Hub PwC (archive figée, offline) ---


def match_pwc(
    df: pd.DataFrame,
    papers_file: Path = PWC_PAPERS_FILE,
    datasets_file: Path = PWC_DATASETS_FILE,
) -> pd.DataFrame:
    """Match contre l'archive PwC : ``arxiv`` puis ``title`` en fallback.

    Ajoute ``in_pwc``, ``pwc_match_method`` (arxiv/title/none), ``pwc_paper_url``,
    ``in_pwc_dataset`` (le papier introduit un dataset PwC).
    """
    pwc = pd.read_parquet(papers_file)
    pwc["arxiv_norm"] = pwc["arxiv_id"].map(norm_arxiv)
    pwc["title_norm"] = pwc["title"].map(norm_title)
    pwc["paper_url"] = pwc["paper_url"].fillna(pwc["url_abs"])

    arxiv2url = (
        pwc.dropna(subset=["arxiv_norm"])
        .drop_duplicates("arxiv_norm")
        .set_index("arxiv_norm")["paper_url"]
    )
    title2url = (
        pwc.dropna(subset=["title_norm"])
        .drop_duplicates("title_norm")
        .set_index("title_norm")["paper_url"]
    )

    arxiv_hit = df["arxiv_norm"].isin(set(arxiv2url.index))
    title_hit = df["title_norm"].isin(set(title2url.index)) & ~arxiv_hit

    out = df.copy()
    out["in_pwc"] = arxiv_hit | title_hit
    out["pwc_match_method"] = "none"
    out.loc[title_hit, "pwc_match_method"] = "title"
    out.loc[arxiv_hit, "pwc_match_method"] = "arxiv"
    out["pwc_paper_url"] = df["arxiv_norm"].map(arxiv2url)
    out.loc[title_hit, "pwc_paper_url"] = df.loc[title_hit, "title_norm"].map(title2url)

    ds = pd.read_parquet(datasets_file)
    ds_paper_urls = set(ds["paper"].map(_paper_url).dropna())
    out["in_pwc_dataset"] = out["pwc_paper_url"].isin(ds_paper_urls)

    logger.info(
        "PwC: in_pwc=%d (arxiv=%d, title=%d) | dataset=%d / %d",
        int(out["in_pwc"].sum()),
        int(arxiv_hit.sum()),
        int(title_hit.sum()),
        int(out["in_pwc_dataset"].sum()),
        len(out),
    )
    return out


def _paper_url(s: object) -> str | None:
    """Champ ``paper`` (dict stringifié) -> son ``url``."""
    try:
        return ast.literal_eval(s).get("url") if isinstance(s, str) else None
    except (ValueError, SyntaxError):
        return None


# --- Hub HF (API datée, cascade résumable) ---


def hf_direct_ids(
    df: pd.DataFrame, resource_links_file: Path = RESOURCE_LINKS_FILE
) -> pd.Series:
    """Niveau ``direct`` (offline) : id HF tiré des liens de la couche 2."""
    rl = pd.read_parquet(resource_links_file)
    rl["hid"] = (
        rl["dataset_urls"]
        .map(hf_dataset_id)
        .fillna(rl["dataset_url"].map(hf_dataset_id))
    )
    aid2hf = (
        rl.dropna(subset=["hid"])
        .drop_duplicates("anthology_id")
        .set_index("anthology_id")["hid"]
    )
    return df["anthology_id"].map(aid2hf)


def _hf_datasets(api, **kwargs) -> list[str]:
    """``list_datasets`` -> liste d'ids (vide si erreur)."""
    try:
        return [d.id for d in api.list_datasets(**kwargs)]
    except Exception:  # noqa: BLE001 — code de recherche, on tolère un échec ponctuel
        return []


def match_hf(
    df: pd.DataFrame,
    resource_links_file: Path = RESOURCE_LINKS_FILE,
    cache_file: Path = HF_CACHE_FILE,
    token: str | None = None,
    do_name_search: bool = True,
    top_k: int = 5,
    sleep: float = 0.3,
) -> pd.DataFrame:
    """Cascade HF par benchmark : ``direct`` -> ``arxiv`` -> ``name``.

    ``in_hf`` = ``direct``/``arxiv`` (sûrs). ``name`` reste hors borne : ses
    candidats vont dans ``hf_namesearch_candidates``. Cache disque (clé
    ``bibkey``) => résumable.
    """
    from huggingface_hub import HfApi

    direct = hf_direct_ids(df, resource_links_file)
    cache = (
        pd.read_parquet(cache_file)
        if cache_file.exists()
        else pd.DataFrame(columns=HF_COLS)
    )
    todo = df[~df["bibkey"].isin(cache["bibkey"])]
    logger.info(
        "HF: en cache=%d | à interroger=%d (anonyme=%s)",
        len(cache),
        len(todo),
        token is None,
    )

    api = HfApi(token=token)
    today = date.today().isoformat()
    rows: list[dict] = []
    for n, (idx, r) in enumerate(todo.iterrows(), start=1):
        method, ds_id, cands = "none", None, []
        if isinstance(direct.get(idx), str):
            method, ds_id = "direct", direct[idx]
        elif isinstance(r["arxiv_norm"], str):
            hits = _hf_datasets(api, filter=f"arxiv:{r['arxiv_norm']}", limit=5)
            if hits:
                method, ds_id = "arxiv", hits[0]
            time.sleep(sleep)
        if method == "none" and do_name_search and isinstance(r["title_norm"], str):
            cands = _hf_datasets(api, search=r["title_norm"], limit=top_k)
            if cands:
                method = "name"
            time.sleep(sleep)

        rows.append(
            {
                "bibkey": r["bibkey"],
                "in_hf": method in ("direct", "arxiv"),
                "hf_match_method": method,
                "hf_dataset_id": ds_id,
                "hf_namesearch_candidates": cands,
                "hf_query_date": today,
            }
        )
        if n % 100 == 0:
            cache = _flush_cache(cache, rows, cache_file)
            rows = []
    cache = _flush_cache(cache, rows, cache_file)

    out = df.merge(cache, on="bibkey", how="left")
    out["in_hf"] = out["in_hf"].fillna(False)
    out["hf_match_method"] = out["hf_match_method"].fillna("none")
    logger.info(
        "HF: in_hf=%d (direct=%d, arxiv=%d) | name seul=%d / %d",
        int(out["in_hf"].sum()),
        int((out["hf_match_method"] == "direct").sum()),
        int((out["hf_match_method"] == "arxiv").sum()),
        int((out["hf_match_method"] == "name").sum()),
        len(out),
    )
    return out


def _flush_cache(
    cache: pd.DataFrame, rows: list[dict], cache_file: Path
) -> pd.DataFrame:
    """Ajoute ``rows`` au cache et persiste."""
    if not rows:
        return cache
    cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(
        "bibkey", keep="last"
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(cache_file, index=False)
    return cache
