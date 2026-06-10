"""Build readable assignment views with leaf label/description.

Reads:
- assignments.csv  (leaf_id,node_id,date)
- leaves.taxobuilder.json (id,label,description,...)

Writes:
- assignments_enriched.csv  (one row per assignment)
- assignments_by_node.csv   (one row per node with concatenated leaves)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_leaves(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("leaves file must be a JSON list.")
    rows = []
    for item in data:
        rows.append(
            {
                "leaf_id": str(item.get("id", "")),
                "leaf_label": str(item.get("label", "")),
                "leaf_description": str(item.get("description", "")),
            }
        )
    return pd.DataFrame(rows)


def _group_with_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.sort_values(["node_id", "leaf_id"])
        .groupby("node_id", dropna=False)
        .agg(
            n_leaves=("leaf_id", "count"),
            leaf_ids=("leaf_id", lambda s: " | ".join(s.astype(str))),
            leaf_labels=("leaf_label", lambda s: " | ".join(s.astype(str))),
            leaf_descriptions=("leaf_description", lambda s: " || ".join(s.astype(str))),
            first_assignment_date=("date", "min"),
            last_assignment_date=("date", "max"),
        )
        .reset_index()
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect taxonomy builder assignments with leaf labels/descriptions."
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        required=True,
        help="Path to assignments.csv exported by taxonomy builder.",
    )
    parser.add_argument(
        "--leaves",
        type=Path,
        required=True,
        help="Path to leaves.taxobuilder.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="assignments",
        help="Output filename prefix (default: assignments).",
    )
    args = parser.parse_args()

    assignments = pd.read_csv(args.assignments, dtype=str).fillna("")
    required_cols = {"leaf_id", "node_id", "date"}
    missing = required_cols - set(assignments.columns)
    if missing:
        raise ValueError(f"assignments CSV missing required columns: {sorted(missing)}")

    leaves = _read_leaves(args.leaves)
    enriched = assignments.merge(leaves, how="left", on="leaf_id")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched_path = out_dir / f"{args.prefix}_enriched.csv"
    grouped_path = out_dir / f"{args.prefix}_by_node.csv"

    enriched.to_csv(enriched_path, index=False, encoding="utf-8")
    _group_with_descriptions(enriched).to_csv(grouped_path, index=False, encoding="utf-8")

    print(f"Wrote: {enriched_path}")
    print(f"Wrote: {grouped_path}")


if __name__ == "__main__":
    main()
