from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _normalize_label(value: object, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else fallback


def build_hierarchy(df: pd.DataFrame) -> dict:
    paper_id_col = None
    for candidate in ("anthology_id", "bibkey"):
        if candidate in df.columns:
            paper_id_col = candidate
            break

    if paper_id_col is None:
        df = df.copy()
        df["__paper_id"] = df.index.astype(str)
        paper_id_col = "__paper_id"

    required = {"mother_task", "task", paper_id_col}
    missing = required - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in parquet: {missing_str}")

    data = (
        df[[paper_id_col, "mother_task", "task"]]
        .dropna(subset=["mother_task", "task"])
        .copy()
    )
    data["mother_task"] = data["mother_task"].map(lambda x: _normalize_label(x, "Unknown Mother Task"))
    data["task"] = data["task"].map(lambda x: _normalize_label(x, "Unknown Task"))

    grouped = (
        data.groupby(["mother_task", "task"], dropna=False)[paper_id_col]
        .nunique()
        .reset_index(name="value")
    )

    mother_task_nodes = []
    for mother_task, frame in grouped.groupby("mother_task", sort=False):
        direct_paper_count = int(
            frame.loc[frame["task"] == mother_task, "value"].sum()
        )

        task_rows = frame.loc[frame["task"] != mother_task]
        children = [
            {
                "id": f"task::{mother_task}::{row.task}",
                "label": row.task,
                "value": int(row.value),
                "paper_count": int(row.value),
                "children": [],
            }
            for row in task_rows.sort_values("value", ascending=False).itertuples(index=False)
        ]

        if direct_paper_count > 0 and children:
            children.append(
                {
                    "id": f"task::{mother_task}::__direct",
                    "label": f"{mother_task} (direct)",
                    "is_direct": True,
                    "value": direct_paper_count,
                    "paper_count": direct_paper_count,
                    "children": [],
                }
            )

        mother_total = int(sum(child["value"] for child in children))
        mother_is_leaf = not children and direct_paper_count > 0
        mother_task_nodes.append(
            {
                "id": f"mother::{mother_task}",
                "label": mother_task,
                # Internal nodes keep value=0 to avoid double-counting in D3 sum.
                # Exception: mother tasks with only direct assignments are encoded as leaves.
                "value": direct_paper_count if mother_is_leaf else 0,
                "paper_count": direct_paper_count if mother_is_leaf else mother_total,
                "children": children,
            }
        )

    mother_task_nodes.sort(key=lambda node: node["paper_count"], reverse=True)
    total = int(sum(node["paper_count"] for node in mother_task_nodes))

    return {
        "id": "root",
        "label": "Tasks",
        "value": 0,
        "paper_count": total,
        "children": mother_task_nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build hierarchical JSON for task sunburst from single_task_benchmark_paper.parquet.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/corpus/single_task_benchmark_paper.parquet"),
        help="Input parquet path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/taxonomy_builder/task_sunburst_data.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    hierarchy = build_hierarchy(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(hierarchy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote sunburst data to: {args.output}")
    print(f"Total mapped papers: {hierarchy['paper_count']}")
    print(f"Mother tasks: {len(hierarchy['children'])}")


if __name__ == "__main__":
    main()
