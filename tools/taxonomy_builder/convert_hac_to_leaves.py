"""Convert HAC/LLM leaf artifacts to taxonomy_builder leaves.json format.

Supported input patterns:
1) leaves from llm_merge save format:
   [{"id": "...", "label": "...", "description": "...", "paraphrases": [...]}, ...]
2) generic leaves:
   [{"id": "...", "label": "...", "description": "...", "freq": 1, "contexts": [...]}, ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _choose_label(row: dict) -> str:
    label = row.get("label")
    if label:
        return str(label)
    if row.get("paraphrase_task"):
        return str(row["paraphrase_task"])
    paraphrases = row.get("paraphrases") or []
    if paraphrases:
        return str(paraphrases[0])
    return str(row.get("id", ""))


def _choose_description(row: dict) -> str:
    description = row.get("description")
    if description:
        return str(description)
    paraphrases = row.get("paraphrases") or []
    if paraphrases:
        return str(paraphrases[0])
    return ""


def _choose_freq(row: dict) -> int:
    if "freq" in row:
        return int(row["freq"])
    if "size" in row:
        return int(row["size"])
    if isinstance(row.get("member_indices"), list):
        return len(row["member_indices"])
    return 1


def convert_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        node_id = str(row["id"])
        contexts = row.get("contexts")
        if not isinstance(contexts, list):
            contexts = row.get("paraphrases") if isinstance(row.get("paraphrases"), list) else []
        out.append(
            {
                "id": node_id,
                "label": _choose_label(row),
                "description": _choose_description(row),
                "freq": _choose_freq(row),
                "contexts": [str(x) for x in contexts],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HAC leaves artifact to taxonomy_builder leaves.json."
    )
    parser.add_argument("input", type=Path, help="Input JSON file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output leaves.json path (default: <input>.taxobuilder.json)",
    )
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of leaf objects.")

    converted = convert_rows(data)
    output = args.output or args.input.with_name(f"{args.input.stem}.taxobuilder.json")
    output.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(converted)} leaves -> {output}")


if __name__ == "__main__":
    main()
