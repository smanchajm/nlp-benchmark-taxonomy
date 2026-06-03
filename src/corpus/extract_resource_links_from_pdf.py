"""Extract dataset/code link candidates from ACL PDFs (GROBID + regex).

GROBID (port 8070) — deux modes :

1. Script seul (pilote, reprise facile) — image **CRF** (~400 Mo, suffisant pour fulltext/footnotes) ::
       docker run --rm -p 8070:8070 grobid/grobid:0.8.1-crf
       uv run python src/corpus/extract_resource_links_from_pdf.py --limit 10

   (Image ``grobid/grobid:0.8.1`` complète ≈ 12 Go ; inutile ici sauf besoin DL max.)

2. Batch CLI (gros volume) — puis ``--skip-grobid`` pour regex + CSV seulement.

Préparer tout le corpus sans regex (PDF + TEI en cache) ::

    uv run python src/corpus/extract_resource_links_from_pdf.py --through grobid

Puis plus tard (regex + CSV, GROBID optionnel si TEI complets) ::

    uv run python src/corpus/extract_resource_links_from_pdf.py --skip-download --skip-grobid

Extraction large (toute URL ``https?://``), deny-list pour le bruit stable,
score de confiance pour ordonner le triage (pas pour exclure).

Texte TEI : une seule passe ``normalize_grobid_text()`` (artefacts GROBID/PDF connus).

Triage : colonne ``decision`` dans ``resource_links_triage.csv``
(dataset | code | reject), lignes triées par ``score`` décroissant.
"""

import argparse
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from lxml import etree
from tqdm import tqdm
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from logging_config import setup_logging
from paths import DATA

logger = logging.getLogger(__name__)

DEFAULT_INPUT = DATA / "corpus" / "single_task_benchmark_paper.parquet"
DEFAULT_WORK_DIR = DATA / "corpus" / "resource_links"
DEFAULT_TRIAGE_CSV = DEFAULT_WORK_DIR / "resource_links_triage.csv"
DEFAULT_SUMMARY_CSV = DEFAULT_WORK_DIR / "resource_links_summary.csv"

TEI_NS = {"t": "http://www.tei-c.org/ns/1.0"}
TEI_ZONE_XPATHS: list[tuple[str, str]] = [
    ("abstract", ".//t:profileDesc//t:abstract//t:p"),
    ("body", ".//t:text/t:body//t:p"),
    ("footnote", ".//t:note[@place='foot']"),
    ("back", ".//t:text/t:back//t:div"),
]

REQUEST_SLEEP = 1.0
GROBID_SLEEP = 0.5
HEADERS = {
    "User-Agent": (
        "RALI-benchmark-cartography/0.1 (research; contact: "
        f"{os.getenv('RESOURCE_LINKS_CONTACT_EMAIL', 'benchmark-taxonomy-research@iro.umontreal.ca')})"
    ),
}

URL_RE = re.compile(r"(?i)\bhttps?://[^\s)\]}>\",;]+")
AVAILABILITY_RE = re.compile(
    r"(?i)\b(available|released?|provided|download(?:able|ed)?|"
    r"can be downloaded|we (?:release|introduce))\b"
)

# Sous-chaînes dans l'URL normalisée — bruit stable, pas des ressources benchmark.
DENY_URL = (
    # DOI du papier ACL / proceedings (pas le dataset)
    "doi.org/10.18653",
    "doi.org/10.3115",
    "dx.doi.org/10.18653",
    "dx.doi.org/10.3115",
    "doi.org/10.1145/",  # ACM proceedings (idno header)
    "doi.org/10.14288/",  # Sockeye / compute ack
    # Légal, standards, éditeur
    "creativecommons.org",
    "w3.org",
    "gdpr-info",
    "eur-lex.europa.eu",
    "aclanthology.org",
    "aclweb.org",
    # API / plateformes (pas les données du papier)
    "t.co/",
    "developer.twitter.com",
    # Libs / outils / embeddings cités
    "keras.io",
    "pytorch.org",
    "pypi.org",
    "fasttext.cc",
    "nominatim.openstreetmap.org",
    "sentiwordnet",
    "code.google.com/archive",
    "radimrehurek.com",
    # Infra recherche (remerciements)
    "computecanada.ca",
    "westgrid.ca",
    # Citations stats / presse / orgs
    "ethnologue.com",
    "internetlivestats.com",
    "who.int",
    "unctad.org",
    "arabsocialmediareport.com",
    "business-standard.com",
    "www.google.com",
)

KNOWN_HOST = {
    "github": "github",
    "githubusercontent": "github",
    "github.io": "github_pages",
    "huggingface": "hf",
    "hf.co": "hf",
    "zenodo": "zenodo",
    "figshare": "figshare",
    "osf": "osf",
    "codalab": "codalab",
    "kaggle": "kaggle",
    "gitlab": "gitlab",
    "drive.google": "gdrive",
    "dropbox": "dropbox",
    "ldc": "ldc",
    "dataverse": "dataverse",
    "doi.org": "doi",
    "mendeley": "mendeley",
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
    """GET avec retry sur 429/503 (Anthology)."""
    response = requests.get(url, headers=HEADERS, timeout=60)
    if response.status_code in (429, 503):
        response.raise_for_status()
    return response


def _pdf_path(pdf_dir: Path, anthology_id: str) -> Path:
    return pdf_dir / f"{anthology_id}.pdf"


def _tei_path(tei_dir: Path, anthology_id: str) -> Path:
    return tei_dir / f"{anthology_id}.grobid.tei.xml"


def step_download_pdfs(papers: pd.DataFrame, pdf_dir: Path) -> dict[str, Path | None]:
    """Télécharge les PDF ACL (cache local, ~1 req/s)."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path | None] = {}
    rows = list(papers.itertuples(index=False))
    for row in tqdm(rows, desc="PDF download", unit="paper"):
        anthology_id = row.anthology_id
        out = _pdf_path(pdf_dir, anthology_id)
        if out.exists():
            paths[anthology_id] = out
            continue
        try:
            response = _http_get(row.pdf_url)
        except requests.RequestException as exc:
            logger.warning("%s — téléchargement: %s", anthology_id, exc)
            paths[anthology_id] = None
            continue
        if response.status_code == 404:
            logger.warning("%s — PDF 404.", anthology_id)
            paths[anthology_id] = None
            continue
        if response.status_code != 200 or response.content[:4] != b"%PDF":
            logger.warning("%s — réponse invalide (%s).", anthology_id, response.status_code)
            paths[anthology_id] = None
            continue
        out.write_bytes(response.content)
        paths[anthology_id] = out
        time.sleep(REQUEST_SLEEP)
    logger.info("PDFs: %d / %d", sum(p is not None for p in paths.values()), len(paths))
    return paths


def step_grobid_tei(
    pdf_paths: dict[str, Path | None],
    tei_dir: Path,
    grobid_url: str,
) -> dict[str, Path | None]:
    """Envoie chaque PDF à GROBID ; produit un TEI XML par papier (footnotes conservées)."""
    tei_dir.mkdir(parents=True, exist_ok=True)
    base = grobid_url.rstrip("/")
    alive = requests.get(f"{base}/api/isalive", timeout=10)
    if alive.status_code != 200 or alive.text.strip().lower() != "true":
        raise RuntimeError(
            f"GROBID injoignable ({base}). "
            "Lancez: docker run --rm -p 8070:8070 grobid/grobid:0.8.1-crf"
        )

    process_url = f"{base}/api/processFulltextDocument"
    tei_paths: dict[str, Path | None] = {}
    items = list(pdf_paths.items())
    for anthology_id, pdf_path in tqdm(items, desc="GROBID TEI", unit="paper"):
        tei_out = _tei_path(tei_dir, anthology_id)
        if tei_out.exists():
            tei_paths[anthology_id] = tei_out
            continue
        if pdf_path is None:
            tei_paths[anthology_id] = None
            continue
        try:
            with pdf_path.open("rb") as handle:
                response = requests.post(
                    process_url,
                    files={"input": (pdf_path.name, handle, "application/pdf")},
                    timeout=300,
                )
        except requests.RequestException as exc:
            logger.warning("%s — GROBID: %s", anthology_id, exc)
            tei_paths[anthology_id] = None
            continue
        if response.status_code != 200:
            logger.warning("%s — GROBID HTTP %s", anthology_id, response.status_code)
            tei_paths[anthology_id] = None
            continue
        tei_out.write_bytes(response.content)
        tei_paths[anthology_id] = tei_out
        time.sleep(GROBID_SLEEP)
    logger.info("TEI: %d / %d", sum(p is not None for p in tei_paths.values()), len(tei_paths))
    return tei_paths


# --- Motifs pour normalize_grobid_text (ordre des passes fixe, ne pas réordonner à la légère) ---
_DOI_PATH = r"10\.\d+/[\w./-]+"
_RE_DX_DOI_SPLIT = re.compile(rf"https?://dx\.?\s*doi\.org/({_DOI_PATH})", re.I)
_RE_BARE_DOI = re.compile(rf"(?<![\w/])doi\.org/({_DOI_PATH})", re.I)
_RE_SCHEME_GAP = re.compile(r"https?:\s*/\s*/")
_RE_HYPHEN_BREAK = re.compile(r"-\s*\n\s*")
# Espaces à l'intérieur d'une URL (ex. "https://www.iitp. ac.in/ ~path")
_RE_SPACED_URL = re.compile(r"https?://(?:[^\s]|\s(?=[a-z0-9~/~._#$-]))+", re.I)
_RE_MULTI_SPACE = re.compile(r"\s+")


def normalize_grobid_text(text: str) -> str:
    """Normalise le texte issu du TEI GROBID avant extraction d'URLs.

    Contrat — artefacts connus (l'ordre des passes est intentionnel) :

    1. **Césure PDF** — ``-\\n`` en fin de ligne (mot coupé).
    2. **DOI dx coupé** — ``http(s)://dx. doi.org/10.xxx`` → ``https://doi.org/10.xxx``.
       Doit passer *avant* la recollage de schéma, sinon ``http://dx`` devient ``https://dx``
       et le DOI ne matche plus (ex. LREC 2020.lrec-1.624).
    3. **Schéma recollé** — ``https: //host`` → ``https://host``.
    4. **DOI sans schéma** — ``doi.org/10.xxx`` nu → ``https://doi.org/10.xxx``.
    5. **Caractères de lien** — tilde PDF (U+02DC, ˜) → ``~`` ; fragment GROBID ``$#$`` → ``#``.
    6. **Espaces intra-URL** — espaces dans le host/path (ex. ``iitp. ac.in``) supprimés
       à l'intérieur du span ``https?://…`` (ex. footnote 2020.lrec-1.621).
    7. **Espaces multiples** — compaction du reste du texte.

    À appeler sur chaque zone TEI (et blocs ``ref``/``idno``) avant ``URL_RE``.
    Ne filtre pas le bruit éditeur (deny-list) : uniquement la forme du texte.
    """
    text = _RE_HYPHEN_BREAK.sub("", text)
    text = _RE_DX_DOI_SPLIT.sub(r"https://doi.org/\1", text)
    text = _RE_SCHEME_GAP.sub("https://", text)
    text = _RE_BARE_DOI.sub(r"https://doi.org/\1", text)
    text = text.replace("\u02dc", "~").replace("˜", "~").replace("$#$", "#")
    text = _RE_SPACED_URL.sub(lambda m: _RE_MULTI_SPACE.sub("", m.group(0)), text)
    return _RE_MULTI_SPACE.sub(" ", text)


def tei_zones(tei_path: Path) -> list[tuple[str, str]]:
    """Découpe le TEI en zones taguées (abstract, body, footnote, back)."""
    root = etree.parse(str(tei_path)).getroot()
    zones: list[tuple[str, str]] = []
    for zone_name, xpath in TEI_ZONE_XPATHS:
        for node in root.findall(xpath, TEI_NS):
            if zone_name == "body" and node.xpath(
                "boolean(ancestor::t:note)", namespaces=TEI_NS
            ):
                continue
            zones.append((zone_name, "".join(node.itertext())))
    return zones


def tei_structured_candidates(tei_path: Path) -> list[dict]:
    """Candidats depuis idno DOI et ref[@type='url'] (cibles GROBID tronquées)."""
    root = etree.parse(str(tei_path)).getroot()
    out: list[dict] = []
    seen: set[str] = set()

    for idno in root.findall(".//t:idno[@type='DOI']", TEI_NS):
        doi = (idno.text or "").strip()
        if not doi:
            continue
        url = _normalize_url(f"https://doi.org/{doi}")
        if len(url) < 15 or _is_denied(url) or url in seen:
            continue
        seen.add(url)
        context = f"TEI idno DOI {doi}"
        out.append(
            {
                "url": url,
                "host_type": _host_type(url),
                "zone": "metadata",
                "context": context,
                "score": confidence_signal(url, context, "metadata"),
            }
        )

    for ref in root.findall(".//t:ref[@type='url']", TEI_NS):
        target = (ref.get("target") or "").strip()
        parent = ref.getparent()
        block = "".join(parent.itertext()) if parent is not None else ""
        blob = normalize_grobid_text(f"{target} {block}")
        for match in URL_RE.finditer(blob):
            url = _normalize_url(match.group(0))
            if len(url) < 15 or _is_denied(url) or url in seen:
                continue
            seen.add(url)
            pos = blob.find(match.group(0))
            start = max(0, pos - 120)
            end = min(len(blob), pos + len(match.group(0)) + 120)
            context = blob[start:end]
            zone = "footnote" if ref.xpath("ancestor::t:note[@place='foot']", namespaces=TEI_NS) else "ref"
            out.append(
                {
                    "url": url,
                    "host_type": _host_type(url),
                    "zone": zone,
                    "context": context,
                    "score": confidence_signal(url, context, zone),
                }
            )
    return out


def _is_denied(url: str) -> bool:
    lower = url.lower()
    return any(fragment in lower for fragment in DENY_URL)


def confidence_signal(url: str, context: str, zone: str) -> float:
    """Score pour ordonner le triage (plus haut = revue en priorité)."""
    score = 0.0
    if _host_type(url) != "other":
        score += 1.0
    if AVAILABILITY_RE.search(context):
        score += 1.0
    if zone == "footnote":
        score += 0.5
    return score


def extract_candidates(
    zones: list[tuple[str, str]],
    context_window: int = 120,
) -> list[dict]:
    """Toute URL https ; deny-list ; score de confiance (n'exclut pas)."""
    candidates: list[dict] = []
    seen: set[str] = set()
    for zone, raw_text in zones:
        text = normalize_grobid_text(raw_text)
        for match in URL_RE.finditer(text):
            chunk = match.group(0)
            offset = match.start()
            for piece in re.split(r"(?=https?://)", chunk):
                if not piece:
                    continue
                url = _normalize_url(piece)
                if len(url) < 15 or _is_denied(url) or url in seen:
                    continue
                seen.add(url)
                pos = offset + chunk.find(piece)
                start = max(0, pos - context_window)
                end = min(len(text), pos + len(piece) + context_window)
                context = text[start:end]
                candidates.append(
                    {
                        "url": url,
                        "host_type": _host_type(url),
                        "zone": zone,
                        "context": context,
                        "score": confidence_signal(url, context, zone),
                    }
                )
    return candidates


def _normalize_url(url: str) -> str:
    """Ajoute https si besoin ; minuscule sur l'hôte seulement (paths sensibles à la casse)."""
    url = url.strip().rstrip(".,;:)]}>'\"")
    if not url.lower().startswith("http"):
        url = "https://" + url
    match = re.match(r"(https?://)([^/]+)(/.*)?$", url, re.I)
    if match:
        url = match.group(1).lower() + match.group(2).lower() + (match.group(3) or "")
    # Ne pas tronquer un fragment (#anchor) ni un path se terminant par /resource.
    if "#" in url or url.endswith((".html", ".htm", ".php", ".json", ".xml")):
        return url
    return url.rstrip("/")


def _host_type(url: str) -> str:
    lower = url.lower()
    for key, label in KNOWN_HOST.items():
        if key in lower:
            return label
    return "other"


def dedup_candidates(cands: list[dict]) -> list[dict]:
    """Une entrée par URL (garde le score le plus élevé)."""
    best: dict[str, dict] = {}
    for c in cands:
        url = c["url"]
        if url not in best or c["score"] > best[url]["score"]:
            best[url] = c
    return list(best.values())


def dedup_by_prefix(cands: list[dict]) -> list[dict]:
    """Supprime les URLs parentes si une URL plus profonde existe (ex. /geopy vs /geopy/geopy)."""
    urls = {c["url"] for c in cands}
    keep_urls = {
        u
        for u in urls
        if not any(u != v and v.startswith(u) for v in urls)
    }
    return [c for c in cands if c["url"] in keep_urls]


def _sort_candidates(cands: list[dict]) -> list[dict]:
    return sorted(cands, key=lambda c: (-c["score"], c["url"]))


def step_extract_candidates(tei_paths: dict[str, Path | None]) -> dict[str, list[dict]]:
    """TEI → candidats URL par ``anthology_id``."""
    by_paper: dict[str, list[dict]] = {}
    for anthology_id, tei_path in tei_paths.items():
        if tei_path is None:
            by_paper[anthology_id] = []
            continue
        try:
            raw = extract_candidates(tei_zones(tei_path)) + tei_structured_candidates(
                tei_path
            )
            by_paper[anthology_id] = _sort_candidates(
                dedup_by_prefix(dedup_candidates(raw))
            )
        except etree.XMLSyntaxError as exc:
            logger.warning("%s — TEI invalide: %s", anthology_id, exc)
            by_paper[anthology_id] = []
    n_with = sum(1 for c in by_paper.values() if c)
    logger.info("Papiers avec >=1 candidat: %d / %d", n_with, len(by_paper))
    return by_paper


def step_write_triage_csv(
    candidates_by_paper: dict[str, list[dict]],
    triage_csv: Path,
) -> None:
    """CSV de tri : ``decision`` vide, lignes ordonnées par ``score`` décroissant."""
    rows: list[dict] = []
    for anthology_id, candidates in candidates_by_paper.items():
        n = len(candidates)
        for c in candidates:
            rows.append(
                {
                    "paper": anthology_id,
                    "n_cand": n,
                    "decision": "",
                    "score": c["score"],
                    "url": c["url"],
                    "host_type": c["host_type"],
                    "zone": c["zone"],
                    "context": c["context"],
                }
            )
    triage_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["paper", "score"], ascending=[True, False])
    df.to_csv(triage_csv, index=False)
    logger.info("Triage CSV (%d lignes): %s", len(df), triage_csv)


def step_write_summary_csv(
    candidates_by_paper: dict[str, list[dict]],
    pdf_paths: dict[str, Path | None],
    tei_paths: dict[str, Path | None],
    summary_csv: Path,
) -> None:
    """Une ligne par papier (y compris n_cand=0) avec état PDF/TEI."""
    rows = [
        {
            "paper": anthology_id,
            "n_cand": len(candidates),
            "max_score": max((c["score"] for c in candidates), default=0.0),
            "has_pdf": pdf_paths.get(anthology_id) is not None,
            "has_tei": tei_paths.get(anthology_id) is not None,
        }
        for anthology_id, candidates in sorted(candidates_by_paper.items())
    ]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    logger.info("Résumé (%d papiers): %s", len(rows), summary_csv)


def _load_cached_paths(
    anthology_ids: list[str],
    directory: Path,
    path_for_id,
) -> dict[str, Path | None]:
    """Map anthology_id → fichier en cache, ou None."""
    return {
        aid: path if (path := path_for_id(directory, aid)).exists() else None
        for aid in anthology_ids
    }


def run_pipeline(
    input_path: Path,
    work_dir: Path,
    triage_csv: Path,
    summary_csv: Path,
    grobid_url: str,
    limit: int | None,
    skip_download: bool,
    skip_grobid: bool,
    through: str,
) -> None:
    """Pipeline par étapes : download → grobid → extract (défaut = tout)."""
    papers = load_papers(input_path, limit)
    ids = papers["anthology_id"].tolist()
    pdf_dir = work_dir / "pdfs"
    tei_dir = work_dir / "tei"

    if skip_download:
        logger.info("Étape 0 ignorée (--skip-download).")
        pdf_paths = _load_cached_paths(ids, pdf_dir, _pdf_path)
    else:
        pdf_paths = step_download_pdfs(papers, pdf_dir)

    if through == "download":
        logger.info("Arrêt après téléchargement PDF (--through download).")
        return

    if skip_grobid:
        logger.info("Étape 1 ignorée (--skip-grobid).")
        tei_paths = _load_cached_paths(ids, tei_dir, _tei_path)
    else:
        tei_paths = step_grobid_tei(pdf_paths, tei_dir, grobid_url)

    if through == "grobid":
        logger.info("Arrêt après GROBID (--through grobid). TEI dans %s", tei_dir)
        return

    candidates_by_paper = step_extract_candidates(tei_paths)
    step_write_triage_csv(candidates_by_paper, triage_csv)
    step_write_summary_csv(candidates_by_paper, pdf_paths, tei_paths, summary_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDF ACL → GROBID → regex → CSV de tri. Voir docstring du module.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--triage-csv", type=Path, default=DEFAULT_TRIAGE_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument(
        "--grobid-url",
        default=os.getenv("GROBID_URL", "http://localhost:8070"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--through",
        choices=("download", "grobid", "extract"),
        default="extract",
        help="Arrêt après cette étape (grobid = PDF + TEI seulement, sans regex/CSV).",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-grobid", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    run_pipeline(
        input_path=args.input,
        work_dir=args.work_dir,
        triage_csv=args.triage_csv,
        summary_csv=args.summary_csv,
        grobid_url=args.grobid_url,
        limit=args.limit,
        skip_download=args.skip_download,
        skip_grobid=args.skip_grobid,
        through=args.through,
    )
