import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import random
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from logging_config import setup_logging
from paths import DATA

logger = logging.getLogger(__name__)

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API_URL = "https://export.arxiv.org/api/query"

DEFAULT_INPUT = DATA / "corpus" / "single_task_benchmark_paper.parquet"
DEFAULT_OUTPUT = DATA / "corpus" / "single_task_benchmark_paper_enriched_links.parquet"
S2_MIN_INTERVAL_SECONDS = 1.1


def norm_title_strict(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = unicodedata.normalize("NFKD", value)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split()) or None


def s2_query_id(row: pd.Series) -> str | None:
    anthology_id = row.get("anthology_id")
    if isinstance(anthology_id, str) and anthology_id.strip():
        return f"ACL:{anthology_id.strip()}"
    return None


def s2_paper_url(corpus_paper_id) -> str | None:
    if pd.isna(corpus_paper_id):
        return None
    try:
        return f"https://www.semanticscholar.org/paper/{int(corpus_paper_id)}"
    except (TypeError, ValueError):
        return None


def extract_arxiv_id_from_url(url: str | None) -> str | None:
    if not isinstance(url, str) or "/abs/" not in url:
        return None
    return url.split("/abs/")[-1].split("?")[0].strip() or None


def query_s2_batch(ids: list[str], api_key: str = "") -> pd.DataFrame:
    headers = {"x-api-key": api_key} if api_key else {}
    rows: list[dict] = []
    batch_size = 500
    max_retries = 8
    last_request_at: float | None = None

    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        for attempt in range(max_retries):
            if last_request_at is not None:
                elapsed = time.monotonic() - last_request_at
                if elapsed < S2_MIN_INTERVAL_SECONDS:
                    time.sleep(S2_MIN_INTERVAL_SECONDS - elapsed)

            resp = requests.post(
                S2_BATCH_URL,
                params={
                    "fields": (
                        "externalIds,title,citationCount,influentialCitationCount,"
                        "fieldsOfStudy,openAccessPdf"
                    )
                },
                json={"ids": chunk},
                headers=headers,
                timeout=60,
            )
            last_request_at = time.monotonic()
            if resp.status_code == 200:
                break

            # 429 is common without API key: respect Retry-After when available,
            # otherwise use a conservative backoff.
            retry_after = resp.headers.get("Retry-After")
            if resp.status_code == 429:
                if retry_after is not None:
                    try:
                        wait_s = max(5.0, float(retry_after))
                    except ValueError:
                        wait_s = 20.0 + 10.0 * attempt
                else:
                    wait_s = 20.0 + 10.0 * attempt
            else:
                wait_s = min(60.0, (2**attempt) * 5.0)

            logger.warning(
                "S2 HTTP %s (chunk %d) -> retry dans %ds",
                resp.status_code,
                start,
                wait_s,
            )
            time.sleep(wait_s)
        else:
            raise RuntimeError(f"S2 batch echoue chunk {start}: {resp.status_code}")

        for query_id, record in zip(chunk, resp.json(), strict=False):
            ext = (record or {}).get("externalIds") or {}
            fields_of_study = (record or {}).get("fieldsOfStudy") or []
            if isinstance(fields_of_study, list):
                fields_of_study = "; ".join(
                    str(value.get("category", value))
                    if isinstance(value, dict)
                    else str(value)
                    for value in fields_of_study
                )
            else:
                fields_of_study = None

            open_access_pdf = (record or {}).get("openAccessPdf") or {}
            rows.append(
                {
                    "s2_query_id": query_id,
                    "s2_found": record is not None,
                    "corpus_paper_id": ext.get("CorpusId"),
                    "arxiv_id": ext.get("ArXiv"),
                    "s2_citation_count": (record or {}).get("citationCount"),
                    "s2_influential_citation_count": (record or {}).get(
                        "influentialCitationCount"
                    ),
                    "s2_fields_of_study": fields_of_study or None,
                    "s2_open_access_pdf_url": open_access_pdf.get("url"),
                }
            )
        logger.info("S2 batch %d-%d ok", start, start + len(chunk))

    return pd.DataFrame(rows)


def query_s2_search_title_year(
    title: str, year: int, api_key: str = "", max_retries: int = 6
) -> dict | None:
    headers = {"x-api-key": api_key} if api_key else {}
    params = {
        "query": title,
        "limit": 10,
        "fields": (
            "externalIds,title,year,citationCount,influentialCitationCount,"
            "fieldsOfStudy,openAccessPdf"
        ),
    }

    for attempt in range(max_retries):
        resp = requests.get(S2_SEARCH_URL, params=params, headers=headers, timeout=60)
        if resp.status_code == 200:
            break

        retry_after = resp.headers.get("Retry-After")
        if resp.status_code == 429:
            if retry_after is not None:
                try:
                    wait_s = max(5.0, float(retry_after))
                except ValueError:
                    wait_s = 20.0 + 10.0 * attempt
            else:
                wait_s = 20.0 + 10.0 * attempt
        else:
            wait_s = min(60.0, (2**attempt) * 5.0)

        if attempt == max_retries - 1:
            return None
        time.sleep(wait_s)
    else:
        return None

    payload = resp.json() if resp.text else {}
    candidates = payload.get("data") or []
    target_norm = norm_title_strict(title)
    for record in candidates:
        rec_title = record.get("title")
        rec_year = record.get("year")
        title_match = norm_title_strict(rec_title) == target_norm
        year_match = isinstance(rec_year, int) and abs(rec_year - int(year)) <= 1
        if title_match and year_match:
            return record
    return None


def query_arxiv_strict(title_norm: str, year: int) -> str | None:
    if not title_norm or pd.isna(year):
        return None

    response = None
    retryable_status = {429, 500, 502, 503, 504}
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.get(
                ARXIV_API_URL,
                params={
                    "search_query": f'ti:"{title_norm}"',
                    "start": 0,
                    "max_results": 8,
                    "sortBy": "relevance",
                },
                timeout=30,
            )
        except requests.RequestException:
            response = None

        if response is not None and response.status_code == 200:
            break

        if attempt == max_retries - 1:
            if response is not None:
                logger.warning(
                    "arXiv echec apres retries (status=%s) pour title=%r year=%s",
                    response.status_code,
                    title_norm,
                    year,
                )
            return None

        should_retry = response is None or response.status_code in retryable_status
        if not should_retry:
            return None

        wait_s = min(20.0, (2**attempt) + random.uniform(0.2, 0.8))
        time.sleep(wait_s)

    if response is None:
        return None

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        return None

    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        published_el = entry.find("atom:published", ns)
        id_el = entry.find("atom:id", ns)
        if title_el is None or published_el is None or id_el is None:
            continue

        title_match = norm_title_strict(title_el.text) == title_norm
        try:
            entry_year = int(published_el.text[:4])
            year_match = abs(entry_year - int(year)) <= 1
        except (TypeError, ValueError):
            year_match = False

        if title_match and year_match:
            return extract_arxiv_id_from_url(id_el.text)

    return None


def resolve_s2(df: pd.DataFrame, s2_api_key: str, query_missing: bool) -> pd.DataFrame:
    if not query_missing:
        logger.info("S2 API desactivee (--query-s2-missing absent).")
        out = df.copy()
        if "s2_found" not in out.columns:
            out["s2_found"] = False
        if "corpus_paper_id" not in out.columns:
            out["corpus_paper_id"] = pd.NA
        if "arxiv_id" not in out.columns:
            out["arxiv_id"] = pd.NA
        if "arxiv_source" not in out.columns:
            out["arxiv_source"] = pd.NA
        if "s2_citation_count" not in out.columns:
            out["s2_citation_count"] = pd.NA
        if "s2_influential_citation_count" not in out.columns:
            out["s2_influential_citation_count"] = pd.NA
        if "s2_fields_of_study" not in out.columns:
            out["s2_fields_of_study"] = pd.NA
        if "s2_open_access_pdf_url" not in out.columns:
            out["s2_open_access_pdf_url"] = pd.NA
        return out

    candidates = (
        df.loc[df["title"].notna() & df["year"].notna(), ["title", "year"]]
        .drop_duplicates()
        .astype({"year": "int64"})
    )
    logger.info("S2 search a requeter=%d", len(candidates))
    if candidates.empty:
        out = df.copy()
        out["s2_found"] = False
        if "corpus_paper_id" not in out.columns:
            out["corpus_paper_id"] = pd.NA
        if "arxiv_id" not in out.columns:
            out["arxiv_id"] = pd.NA
        if "arxiv_source" not in out.columns:
            out["arxiv_source"] = pd.NA
        if "s2_citation_count" not in out.columns:
            out["s2_citation_count"] = pd.NA
        if "s2_influential_citation_count" not in out.columns:
            out["s2_influential_citation_count"] = pd.NA
        if "s2_fields_of_study" not in out.columns:
            out["s2_fields_of_study"] = pd.NA
        if "s2_open_access_pdf_url" not in out.columns:
            out["s2_open_access_pdf_url"] = pd.NA
        return out

    rows: list[dict] = []
    iterator = tqdm(
        candidates.itertuples(index=False),
        total=len(candidates),
        desc="S2 search",
        unit="paper",
    )
    for candidate in iterator:
        record = query_s2_search_title_year(
            title=candidate.title, year=int(candidate.year), api_key=s2_api_key
        )
        ext = (record or {}).get("externalIds") or {}
        fields_of_study = (record or {}).get("fieldsOfStudy") or []
        if isinstance(fields_of_study, list):
            fields_of_study = "; ".join(
                str(value.get("category", value)) if isinstance(value, dict) else str(value)
                for value in fields_of_study
            )
        else:
            fields_of_study = None
        open_access_pdf = (record or {}).get("openAccessPdf") or {}

        rows.append(
            {
                "title": candidate.title,
                "year": int(candidate.year),
                "s2_found_match": record is not None,
                "corpus_paper_id_match": ext.get("CorpusId"),
                "arxiv_id_match": ext.get("ArXiv"),
                "s2_citation_count_match": (record or {}).get("citationCount"),
                "s2_influential_citation_count_match": (record or {}).get(
                    "influentialCitationCount"
                ),
                "s2_fields_of_study_match": fields_of_study or None,
                "s2_open_access_pdf_url_match": open_access_pdf.get("url"),
            }
        )

        time.sleep(S2_MIN_INTERVAL_SECONDS)

    matched = pd.DataFrame(rows)
    out = df.merge(matched, on=["title", "year"], how="left")
    out["s2_found"] = out["s2_found_match"].fillna(False)
    out["corpus_paper_id"] = out["corpus_paper_id_match"]
    if "arxiv_id" in out.columns:
        out["arxiv_id"] = out["arxiv_id"].fillna(out["arxiv_id_match"])
    else:
        out["arxiv_id"] = out["arxiv_id_match"]
    if "arxiv_source" not in out.columns:
        out["arxiv_source"] = pd.NA
    out["s2_citation_count"] = out["s2_citation_count_match"]
    out["s2_influential_citation_count"] = out["s2_influential_citation_count_match"]
    out["s2_fields_of_study"] = out["s2_fields_of_study_match"]
    out["s2_open_access_pdf_url"] = out["s2_open_access_pdf_url_match"]
    out.loc[out["arxiv_id"].notna(), "arxiv_source"] = "s2_search_title_year"

    return out.drop(
        columns=[
            "s2_found_match",
            "corpus_paper_id_match",
            "arxiv_id_match",
            "s2_citation_count_match",
            "s2_influential_citation_count_match",
            "s2_fields_of_study_match",
            "s2_open_access_pdf_url_match",
        ],
        errors="ignore",
    )


def resolve_arxiv(df: pd.DataFrame, query_missing: bool, arxiv_workers: int) -> pd.DataFrame:
    if not query_missing:
        logger.info("arXiv API desactivee (--query-arxiv-missing absent).")
        return df

    missing_before = df["arxiv_id"].isna()
    candidates = (
        df.loc[missing_before, ["title_norm", "year"]]
        .dropna()
        .drop_duplicates()
        .assign(year=lambda x: x["year"].astype(int))
    )
    to_query = [(row.title_norm, int(row.year)) for row in candidates.itertuples(index=False)]
    logger.info("arXiv a requeter=%d | workers=%d", len(to_query), arxiv_workers)

    if to_query:
        rows = []
        with ThreadPoolExecutor(max_workers=max(1, arxiv_workers)) as executor:
            futures = {
                executor.submit(query_arxiv_strict, title_norm, year): (title_norm, year)
                for title_norm, year in to_query
            }
            progress = tqdm(total=len(to_query), desc="arXiv search", unit="paper")
            for future in as_completed(futures):
                title_norm, year = futures[future]
                try:
                    arxiv_id = future.result()
                except Exception:
                    arxiv_id = None
                rows.append({"title_norm": title_norm, "year": year, "arxiv_id": arxiv_id})
                progress.update(1)
            progress.close()
        arxiv_lookup = pd.DataFrame(rows).set_index(["title_norm", "year"])["arxiv_id"].to_dict()
    else:
        arxiv_lookup = {}

    def lookup(row: pd.Series) -> str | None:
        if pd.isna(row["year"]):
            return None
        return arxiv_lookup.get((row["title_norm"], int(row["year"])))

    df.loc[missing_before, "arxiv_id"] = df.loc[missing_before].apply(lookup, axis=1)
    filled = int((missing_before & df["arxiv_id"].notna()).sum())
    df.loc[missing_before & df["arxiv_id"].notna(), "arxiv_source"] = "arxiv_api_title_year"
    logger.info("arXiv strict title+year: +%d arxiv_id completes", filled)
    return df


def step_load_input(input_path: Path) -> pd.DataFrame:
    logger.info("Chargement: %s", input_path)
    return pd.read_parquet(input_path)


def step_prepare_resolution_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["s2_query_id"] = df.apply(s2_query_id, axis=1)
    df["title_norm"] = df["title"].map(norm_title_strict)
    return df


def step_log_overview(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(
        "single_task=%d | resolvables=%d | avec doi=%d",
        len(df),
        int(df["s2_query_id"].notna().sum()),
        int(df["doi"].notna().sum()),
    )
    return df


def step_finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["s2_paper_url"] = df["corpus_paper_id"].map(s2_paper_url)
    if "pdf_url" in df.columns:
        df["acl_pdf_url"] = df["pdf_url"]
    else:
        df["acl_pdf_url"] = pd.NA
    return df.drop(columns=["title_norm"])


def step_save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Corpus enrichi -> %s (%d lignes)", output_path, len(df))


def run_pipeline(
    input_path: Path,
    output_path: Path,
    s2_api_key: str,
    query_s2_missing: bool,
    query_arxiv_missing: bool,
    arxiv_workers: int,
) -> None:
    df = step_load_input(input_path)
    df = step_prepare_resolution_columns(df)
    df = step_log_overview(df)
    df = resolve_s2(df, s2_api_key, query_s2_missing)
    df = resolve_arxiv(df, query_arxiv_missing, arxiv_workers)
    df = step_finalize_columns(df)
    step_save_output(df, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extraire des arxiv_id pour single_task_benchmark_paper (S2 + arXiv)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--query-s2-missing",
        action="store_true",
        help="Interroger Semantic Scholar pour les IDs absents du cache S2.",
    )
    parser.add_argument(
        "--query-arxiv-missing",
        action="store_true",
        help="Interroger arXiv (strict titre+annee) pour les arxiv_id manquants.",
    )
    parser.add_argument(
        "--arxiv-workers",
        type=int,
        default=8,
        help="Nombre de workers paralleles pour les requetes arXiv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    s2_api_key = os.getenv("S2_API_KEY", "").strip()
    if s2_api_key:
        logger.info("S2 API key chargee (env).")
    else:
        logger.info("S2 API key absente: mode non authentifie.")
    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        s2_api_key=s2_api_key,
        query_s2_missing=args.query_s2_missing,
        query_arxiv_missing=args.query_arxiv_missing,
        arxiv_workers=args.arxiv_workers,
    )
