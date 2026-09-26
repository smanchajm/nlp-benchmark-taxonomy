"""Patch existing Label Studio tasks to add missing fields (e.g. paraphrase_task).

Usage:
    uv run python src/evaluation/patch_label_studio_tasks.py \
        --url https://labelstudio.samuelmanchajm.fr \
        --token <API_TOKEN> \
        --project <PROJECT_ID> \
        --fields paraphrase_task
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
from logging_config import setup_logging
from paths import DATA

logger = logging.getLogger(__name__)

PARQUET = DATA / "corpus" / "single_task_benchmark_paper.parquet"

PATCHABLE_FIELDS = [
    "paraphrase_task", "paraphrase_domain", "thematic_domain",
    "benchmark_languages", "dataset_url", "language",
]


def get_all_tasks(base_url: str, token: str, project_id: int) -> list[dict]:
    headers = {"Authorization": f"Token {token}"}
    tasks, page = [], 1
    while True:
        r = requests.get(
            f"{base_url}/api/tasks",
            params={"project": project_id, "page": page, "page_size": 100},
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        body = r.json()
        batch = body.get("tasks", body) if isinstance(body, dict) else body
        if not batch:
            break
        tasks.extend(batch)
        if isinstance(body, dict) and not body.get("next"):
            break
        page += 1
    return tasks


def patch_tasks(
    base_url: str,
    token: str,
    project_id: int,
    fields: list[str],
    parquet: Path = PARQUET,
    dry_run: bool = False,
) -> None:
    df = pd.read_parquet(parquet)
    lookup = df.set_index("bibkey")[fields].to_dict(orient="index")

    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    tasks = get_all_tasks(base_url, token, project_id)
    logger.info("Found %d tasks in project %d", len(tasks), project_id)

    patched = skipped = errors = 0
    for task in tasks:
        bibkey = task.get("data", {}).get("bibkey")
        if not bibkey or bibkey not in lookup:
            skipped += 1
            continue

        patch_data = {}
        for field in fields:
            val = lookup[bibkey].get(field)
            if pd.isna(val) if not isinstance(val, str) else not val:
                patch_data[field] = ""
            elif hasattr(val, "tolist"):   # numpy array
                patch_data[field] = ", ".join(str(v) for v in val.tolist())
            else:
                patch_data[field] = str(val)

        if dry_run:
            logger.info("[DRY RUN] would patch task %d (%s): %s", task["id"], bibkey, patch_data)
            patched += 1
            continue

        r = requests.patch(
            f"{base_url}/api/tasks/{task['id']}",
            json={"data": {**task["data"], **patch_data}},
            headers=headers,
            timeout=30,
        )
        if r.ok:
            patched += 1
        else:
            logger.warning("Task %d failed: %s %s", task["id"], r.status_code, r.text[:200])
            errors += 1
        time.sleep(0.05)   # avoid hammering the API

    logger.info("Done — patched=%d  skipped=%d  errors=%d", patched, skipped, errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch Label Studio task data fields.")
    parser.add_argument("--url", required=True, help="Label Studio base URL")
    parser.add_argument("--token", required=True, help="API token (Settings > Account)")
    parser.add_argument("--project", type=int, required=True, help="Project ID")
    parser.add_argument(
        "--fields", nargs="+", default=["paraphrase_task"],
        choices=PATCHABLE_FIELDS, help="Fields to add/update"
    )
    parser.add_argument("--dry-run", action="store_true", help="Log what would be patched without writing")
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    patch_tasks(args.url, args.token, args.project, args.fields, args.parquet, args.dry_run)
