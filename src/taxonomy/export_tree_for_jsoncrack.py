"""Convert a flat LLM-merge tree JSON to a nested JSON for graph viewers.

The flat ``{"nodes": {...}, "roots": [...]}`` payload produced by
:func:`src.taxonomy.llm_merge.save_llm_tree` is great for editing and
post-processing, but visualizers like JSON Crack expect a self-nested
structure to draw a proper graph. This script materializes that view.

Usage:
    uv run python -m src.taxonomy.export_tree_for_jsoncrack INPUT [-o OUTPUT]
                                                            [--no-paraphrases]
                                                            [--keep-none]

Examples:
    # Default: writes data/taxonomy/llm_tree_postprocessed/tree_final.nested.json
    uv run python -m src.taxonomy.export_tree_for_jsoncrack \\
        data/taxonomy/llm_tree_postprocessed/tree_final.json

    # Drop paraphrases for a cleaner graph
    uv run python -m src.taxonomy.export_tree_for_jsoncrack \\
        data/taxonomy/llm_tree_postprocessed/tree_final.json --no-paraphrases
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.logging_config import setup_logging
from src.taxonomy.llm_merge import save_nested_tree

logger = logging.getLogger(__name__)


def _default_output_path(input_path: Path) -> Path:
    """``tree_final.json`` -> ``tree_final.nested.json`` next to the source."""
    return input_path.with_name(f"{input_path.stem}.nested.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a flat LLM-merge tree.json into a nested JSON suitable "
            "for graph viewers (JSON Crack, jsoncrack.com, ...)."
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the flat tree JSON (e.g. tree_final.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path. Defaults to '<input_stem>.nested.json' next to the "
            "input file."
        ),
    )
    parser.add_argument(
        "--no-paraphrases",
        action="store_true",
        help="Strip the paraphrase lists from each node for a cleaner view.",
    )
    parser.add_argument(
        "--keep-none",
        action="store_true",
        help="Keep metadata fields whose value is None (default: drop them).",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help=(
            "Preserve the original child order. By default, children "
            "(including the forest roots) are sorted by descending size."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path: Path = args.input
    if not input_path.is_file():
        raise FileNotFoundError(f"Input tree not found: {input_path}")

    output_path: Path = args.output or _default_output_path(input_path)

    with input_path.open(encoding="utf-8") as f:
        tree = json.load(f)

    if "nodes" not in tree or "roots" not in tree:
        raise ValueError(
            f"{input_path} does not look like a flat tree payload "
            "(expected keys 'nodes' and 'roots')."
        )

    n_nodes = len(tree["nodes"])
    n_roots = len(tree["roots"])
    logger.info("Loaded flat tree: %d nodes, %d roots", n_nodes, n_roots)

    save_nested_tree(
        tree,
        out_path=output_path,
        include_paraphrases=not args.no_paraphrases,
        drop_none=not args.keep_none,
        sort_by_size=not args.no_sort,
    )
    logger.info("Done -> %s", output_path)


if __name__ == "__main__":
    setup_logging()
    main()
