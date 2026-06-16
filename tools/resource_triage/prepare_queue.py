"""Fusionne le corpus benchmark + triage CSV → papers_queue.json pour l'UI."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from logging_config import setup_logging
from paths import DATA

logger = logging.getLogger(__name__)

DEFAULT_PAPERS = DATA / "corpus" / "single_task_benchmark_paper.parquet"
DEFAULT_TRIAGE = DATA / "corpus" / "papers" / "resource_links_triage.csv"
DEFAULT_SUMMARY = DATA / "corpus" / "papers" / "resource_links_summary.csv"
DEFAULT_OUTPUT = DATA / "corpus" / "papers" / "papers_queue.json"

ANTHOLOGY_BASE = "https://aclanthology.org/"


def _anthology_url(anthology_id: str) -> str:
    return f"{ANTHOLOGY_BASE}{anthology_id}"


def load_candidates(triage_path: Path) -> dict[str, list[dict]]:
    if not triage_path.exists():
        logger.warning("Pas de triage CSV (%s) — candidats vides.", triage_path)
        return {}
    df = pd.read_csv(triage_path)
    required = {"paper", "url", "score", "host_type", "zone", "context"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {triage_path}: {sorted(missing)}")
    by_paper: dict[str, list[dict]] = {}
    for paper, group in df.groupby("paper", sort=False):
        rows = group.sort_values("score", ascending=False)
        by_paper[str(paper)] = [
            {
                "url": str(r.url),
                "score": float(r.score),
                "host_type": str(r.host_type),
                "zone": str(r.zone),
                "context": str(r.context)[:500],
            }
            for r in rows.itertuples(index=False)
        ]
    return by_paper


def load_summary(summary_path: Path) -> dict[str, dict]:
    if not summary_path.exists():
        return {}
    df = pd.read_csv(summary_path)
    out: dict[str, dict] = {}
    for r in df.itertuples(index=False):
        out[str(r.paper)] = {
            "n_cand": int(r.n_cand),
            "max_score": float(r.max_score),
            "has_pdf": bool(r.has_pdf),
            "has_tei": bool(r.has_tei),
        }
    return out


def build_queue(
    papers_path: Path,
    triage_path: Path,
    summary_path: Path,
    output_path: Path,
    limit: int | None,
) -> None:
    df = pd.read_parquet(papers_path)
    if "anthology_id" not in df.columns:
        raise ValueError(f"Colonne anthology_id absente dans {papers_path}")
    df = df.drop_duplicates(subset=["anthology_id"])
    if limit is not None:
        df = df.head(limit)

    candidates_by_paper = load_candidates(triage_path)
    summary_by_paper = load_summary(summary_path)

    papers: list[dict] = []
    for row in df.itertuples(index=False):
        aid = str(row.anthology_id)
        meta = summary_by_paper.get(aid, {})
        papers.append(
            {
                "anthology_id": aid,
                "title": str(getattr(row, "title", "") or ""),
                "year": int(row.year) if pd.notna(getattr(row, "year", None)) else None,
                "mother_task": str(getattr(row, "mother_task", "") or ""),
                "task": str(getattr(row, "task", "") or ""),
                "abstract": str(getattr(row, "abstract", "") or ""),
                "anthology_url": _anthology_url(aid),
                "pdf_url": str(getattr(row, "pdf_url", "") or ""),
                "n_cand": meta.get("n_cand", len(candidates_by_paper.get(aid, []))),
                "max_score": meta.get("max_score", 0.0),
                "has_pdf": meta.get("has_pdf", False),
                "has_tei": meta.get("has_tei", False),
                "candidates": candidates_by_paper.get(aid, []),
            }
        )

    papers.sort(key=lambda p: (-p["max_score"], -p["n_cand"], p["anthology_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    n_with = sum(1 for p in papers if p["n_cand"] > 0)
    logger.info(
        "Ecrit %d papiers (%d avec candidats) -> %s",
        len(papers),
        n_with,
        output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=Path, default=DEFAULT_PAPERS)
    parser.add_argument("--triage", type=Path, default=DEFAULT_TRIAGE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    build_queue(args.papers, args.triage, args.summary, args.output, args.limit)
