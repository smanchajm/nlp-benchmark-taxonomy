"""Acquisition PDF + GROBID TEI pour le corpus ACL (couche partagée).

Télécharge le PDF de chaque papier (``aclanthology.org/{id}.pdf``) puis le re-parse
avec GROBID en TEI XML (footnotes conservées). Les deux étapes sont **cachées par
fichier** et reprenables. Réutilisable par n'importe quelle extraction TEI en aval
(liens ressource, citations, ...). L'extraction est faite par d'autres modules.

GROBID (port 8070), image **CRF** légère (~480 Mo) ::

    docker run -d --name grobid --init --ulimit core=0 -p 8070:8070 \\
      lfoppiano/grobid:0.9.0-crf
    curl -s http://localhost:8070/api/isalive   # -> true

Pré-remplir le cache (étapes lentes), depuis la racine du repo ::

    uv run python -m src.extraction.grobid_tei                      # PDF + TEI
    uv run python -m src.extraction.grobid_tei --download-only      # PDF seulement
    uv run python -m src.extraction.grobid_tei --skip-download      # TEI sur PDF déjà en cache
"""

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.logging_config import setup_logging
from src.paths import DATA

logger = logging.getLogger(__name__)

DEFAULT_INPUT = DATA / "corpus" / "single_task_benchmark_paper.parquet"
DEFAULT_WORK_DIR = DATA / "corpus" / "resource_links"

REQUEST_SLEEP = 1.0
GROBID_SLEEP = 0.5
HEADERS = {
    "User-Agent": (
        "RALI-benchmark-cartography/0.1 (research; contact: "
        f"{os.getenv('RESOURCE_LINKS_CONTACT_EMAIL', 'benchmark-taxonomy-research@iro.umontreal.ca')})"
    ),
}


def load_papers(input_path: Path, limit: int | None) -> pd.DataFrame:
    """Charge le parquet et retourne une ligne par ``anthology_id`` unique."""
    df = pd.read_parquet(input_path)
    for col in ("anthology_id", "pdf_url"):
        if col not in df.columns:
            raise ValueError(f"Colonne {col} absente dans {input_path}")
    df = df.dropna(subset=["anthology_id", "pdf_url"]).copy()
    df = df.astype({"anthology_id": "string", "pdf_url": "string"})
    df["anthology_id"] = df["anthology_id"].str.strip()
    df["pdf_url"] = df["pdf_url"].str.strip()
    df = df.drop_duplicates(subset=["anthology_id"])
    if limit is not None:
        df = df.head(limit)
    logger.info("%d papiers à traiter.", len(df))
    return df


@retry(
    wait=wait_exponential(min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(requests.HTTPError),
    reraise=True,
)
def _http_get(url: str) -> requests.Response:
    """GET avec retry exponentiel sur 429/503 (Anthology, serveur bénévole)."""
    response = requests.get(url, headers=HEADERS, timeout=60)
    if response.status_code in (429, 503):
        response.raise_for_status()
    return response


def _pdf_path(pdf_dir: Path, anthology_id: str) -> Path:
    return pdf_dir / f"{anthology_id}.pdf"


def _tei_path(tei_dir: Path, anthology_id: str) -> Path:
    return tei_dir / f"{anthology_id}.grobid.tei.xml"


def _load_cached_paths(
    ids: list[str], directory: Path, path_for_id
) -> dict[str, Path | None]:
    """Map anthology_id -> fichier en cache (ou None s'il manque)."""
    return {
        aid: path if (path := path_for_id(directory, aid)).exists() else None
        for aid in ids
    }


def _download_pdf(anthology_id: str, pdf_url: str, pdf_dir: Path) -> Path | None:
    """Un PDF : cache-hit, sinon download + validation. None (loggé) si échec/404."""
    out = _pdf_path(pdf_dir, anthology_id)
    if out.exists():
        return out
    try:
        response = _http_get(pdf_url)
    except requests.RequestException as exc:
        logger.warning("%s — téléchargement: %s", anthology_id, exc)
        return None
    if response.status_code != 200 or response.content[:4] != b"%PDF":
        logger.warning("%s — réponse invalide (%s).", anthology_id, response.status_code)
        return None
    out.write_bytes(response.content)
    time.sleep(REQUEST_SLEEP)
    return out


def step_download_pdfs(papers: pd.DataFrame, pdf_dir: Path) -> dict[str, Path | None]:
    """Télécharge les PDF ACL (cache local, ~1 req/s)."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    rows = list(papers.itertuples(index=False))
    paths: dict[str, Path | None] = {
        row.anthology_id: _download_pdf(row.anthology_id, row.pdf_url, pdf_dir)
        for row in tqdm(rows, desc="PDF download", unit="paper")
    }
    logger.info("PDFs: %d / %d", sum(p is not None for p in paths.values()), len(paths))
    return paths


def _check_grobid_alive(base: str) -> None:
    """Vérifie que le service GROBID répond avant d'envoyer le corpus (échec actionnable)."""
    try:
        alive = requests.get(f"{base}/api/isalive", timeout=10)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"GROBID injoignable ({base}). Démarrage: docker run -d --name grobid "
            "--init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.9.0-crf"
        ) from exc
    if alive.status_code != 200 or alive.text.strip().lower() != "true":
        raise RuntimeError(f"GROBID pas prêt ({base}, HTTP {alive.status_code!r}).")


def _grobid_tei(anthology_id: str, pdf_path: Path | None, tei_dir: Path, process_url: str) -> Path | None:
    """Un TEI : cache-hit, sinon POST GROBID. None (loggé) si pas de PDF ou échec."""
    out = _tei_path(tei_dir, anthology_id)
    if out.exists():
        return out
    if pdf_path is None:
        return None
    try:
        with pdf_path.open("rb") as handle:
            response = requests.post(
                process_url,
                files={"input": (pdf_path.name, handle, "application/pdf")},
                timeout=300,
            )
    except requests.RequestException as exc:
        logger.warning("%s — GROBID: %s", anthology_id, exc)
        return None
    if response.status_code != 200:
        logger.warning("%s — GROBID HTTP %s", anthology_id, response.status_code)
        return None
    out.write_bytes(response.content)
    time.sleep(GROBID_SLEEP)
    return out


def step_grobid_tei(
    pdf_paths: dict[str, Path | None], tei_dir: Path, grobid_url: str
) -> dict[str, Path | None]:
    """Envoie chaque PDF à GROBID ; un TEI XML par papier (footnotes conservées)."""
    tei_dir.mkdir(parents=True, exist_ok=True)
    base = grobid_url.rstrip("/")
    _check_grobid_alive(base)
    process_url = f"{base}/api/processFulltextDocument"
    tei_paths: dict[str, Path | None] = {
        aid: _grobid_tei(aid, pdf_path, tei_dir, process_url)
        for aid, pdf_path in tqdm(list(pdf_paths.items()), desc="GROBID TEI", unit="paper")
    }
    logger.info(
        "TEI: %d / %d", sum(p is not None for p in tei_paths.values()), len(tei_paths)
    )
    return tei_paths


def acquire(
    papers: pd.DataFrame,
    work_dir: Path,
    grobid_url: str,
    skip_download: bool = False,
    skip_grobid: bool = False,
    download_only: bool = False,
) -> tuple[dict[str, Path | None], dict[str, Path | None]]:
    """Cache PDF puis TEI -> (pdf_paths, tei_paths). ``skip_*`` réutilise le cache existant."""
    ids = papers["anthology_id"].tolist()
    pdf_dir, tei_dir = work_dir / "pdfs", work_dir / "tei"

    if skip_download:
        pdf_paths = _load_cached_paths(ids, pdf_dir, _pdf_path)
    else:
        pdf_paths = step_download_pdfs(papers, pdf_dir)
    if download_only:
        return pdf_paths, {}

    if skip_grobid:
        tei_paths = _load_cached_paths(ids, tei_dir, _tei_path)
    else:
        tei_paths = step_grobid_tei(pdf_paths, tei_dir, grobid_url)
    return pdf_paths, tei_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pré-remplir le cache PDF + GROBID TEI. Voir docstring."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--grobid-url", default=os.getenv("GROBID_URL", "http://localhost:8070")
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--download-only", action="store_true", help="S'arrêter après le PDF."
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-grobid", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    papers = load_papers(args.input, args.limit)
    acquire(
        papers,
        work_dir=args.work_dir,
        grobid_url=args.grobid_url,
        skip_download=args.skip_download,
        skip_grobid=args.skip_grobid,
        download_only=args.download_only,
    )
