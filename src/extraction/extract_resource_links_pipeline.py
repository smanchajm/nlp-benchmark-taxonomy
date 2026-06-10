"""CLI batch Mistral pour l'arbitre de liens ressource (couche 2).

    prepare  acquisition (grobid_tei) -> liens (extract_resource_links) -> parquet d'items

Acquisition PDF/TEI : ``grobid_tei.py`` (cache partagé, réutilisable).
Extraction des liens : ``extract_resource_links.py``.
Infra batch + schéma de sortie (``LinkArbitration``) : ``taxonomy/`` — réutilisés tels quels.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.extraction.extract_resource_links import extract_links
from src.extraction.grobid_tei import (
    DEFAULT_INPUT,
    DEFAULT_WORK_DIR,
    acquire,
    load_papers,
)
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_BATCH_INPUT = DEFAULT_WORK_DIR / "link_arbiter_batch.parquet"


# --------------------------------------------------------------------------- #
# prepare — parquet d'items pour le batch                                      #
# --------------------------------------------------------------------------- #


def preselect_candidates(cands: list[dict], top_k: int) -> list[dict]:
    """Top-k par score décroissant (borne ce qu'on envoie au LLM, sans rien décider)."""
    return sorted(cands, key=lambda c: -c.get("score", 0))[:top_k]


def render_paper_text(title: str, abstract: str, candidates: list[dict]) -> str:
    """Le prompt rendu d'un papier : titre + abstract + candidats ordonnés."""
    lines = [f"TITRE: {title}", f"ABSTRACT: {abstract or ''}", "", "CANDIDATS:"]
    for i, c in enumerate(candidates, 1):
        lines.append(f"{i}. [{c['zone']}] {c['url']}")
        if c.get("context"):
            lines.append(f'   contexte: "{c["context"][:300]}"')
    return "\n".join(lines)


def write_batch_input(
    papers: pd.DataFrame,
    candidates_by_paper: dict[str, list[dict]],
    output_path: Path,
    top_k: int,
) -> None:
    """Un parquet, une ligne par papier (``paper_text`` = prompt rendu pour ``link_arbiter``)."""
    meta = papers.set_index("anthology_id")
    rows: list[dict] = []
    for aid in papers["anthology_id"]:
        raw = candidates_by_paper.get(aid, [])
        selected = preselect_candidates(raw, top_k)
        row = meta.loc[aid]
        title, abstract = (
            str(row.get("title", "") or ""),
            str(row.get("abstract", "") or ""),
        )
        rows.append(
            {
                "anthology_id": aid,
                "n_candidates_raw": len(raw),
                "candidates_json": json.dumps(selected, ensure_ascii=False),
                "paper_text": render_paper_text(title, abstract, selected),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    logger.info("Batch input (%d papiers) -> %s", len(rows), output_path)


def cmd_prepare(args: argparse.Namespace) -> None:
    papers = load_papers(args.input, args.limit)
    pdf_paths, tei_paths = acquire(
        papers,
        work_dir=args.work_dir,
        grobid_url=args.grobid_url,
        skip_download=args.skip_download,
        skip_grobid=args.skip_grobid,
        download_only=args.through == "download",
    )
    if args.through != "candidates":
        return
    candidates = extract_links(tei_paths, pdf_paths)
    write_batch_input(papers, candidates, args.batch, args.top_k)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="acquisition + candidats -> parquet batch."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH_INPUT)
    parser.add_argument("--grobid-url", default="http://localhost:8070")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--through", choices=("download", "grobid", "candidates"), default="candidates"
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-grobid", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    cmd_prepare(args)
