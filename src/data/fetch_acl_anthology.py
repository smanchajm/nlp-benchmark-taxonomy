import logging
from pathlib import Path

from acl_anthology import Anthology
from acl_anthology.collections.paper import Paper
import pandas as pd

from logging_config import setup_logging

logger = logging.getLogger(__name__)


def _paper_to_record(paper: Paper) -> dict:
    return {
        "id": paper.full_id,
        "bibkey": paper.bibkey,
        "title": paper.title.as_text(),
        "abstract": paper.abstract.as_text() if paper.abstract else None,
        "authors": [a.name.as_first_last() for a in paper.authors],
        "year": paper.year,
        "venues": paper.venue_ids,
        "doi": paper.doi,
        "url": paper.web_url,
        "language": paper.language,
    }


def fetch_anthology(output_path: Path = Path("data/anthology.parquet")) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ACL Anthology from repo...")
    anthology = Anthology.from_repo()

    logger.info("Extracting papers...")
    records = [
        _paper_to_record(paper)
        for paper in anthology.papers()
        if not paper.is_deleted and not paper.is_frontmatter
    ]

    logger.info("Writing %d papers to %s...", len(records), output_path)
    pd.DataFrame(records).to_parquet(output_path, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    setup_logging()
    fetch_anthology()
