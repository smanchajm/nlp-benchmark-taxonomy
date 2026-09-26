"""Generate Label Studio labeling config from manual taxonomy using <Taxonomy> tag.

Layout:
- Left column: paper content (bibkey, PDF link, title, abstract)
- Right column: annotation controls
  1. Quick "Is benchmark?" Choices — non-benchmarks skipped without touching taxonomy
  2. <Taxonomy> with leafOnly + filter — shown only when benchmark=yes
  3. Notes textarea
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import xml.sax.saxutils as saxutils
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logging_config import setup_logging
from paths import DATA

logger = logging.getLogger(__name__)

DEFAULT_TAXONOMY_PATH = DATA / "taxonomy" / "manual_taxonomy_tree.json"
DEFAULT_OUTPUT_PATH = DATA / "evaluation" / "label_studio_taxo_config.xml"

_INDENT = "  "


def _esc(value: str) -> str:
    return saxutils.escape(value, {"'": "&apos;", '"': "&quot;"})


def _render_choice(node: dict, depth: int) -> list[str]:
    """Recursively render a taxonomy node as nested <Choice> elements."""
    pad = _INDENT * depth
    label = _esc(str(node["label"]))
    desc = node.get("description")
    hint = f' hint="{_esc(str(desc))}"' if desc else ""
    children = [c for c in (node.get("children") or []) if isinstance(c, dict)]

    if not children:
        return [f"{pad}<Choice value=\"{label}\"{hint}/>"]

    lines = [f"{pad}<Choice value=\"{label}\"{hint}>"]
    for child in children:
        lines.extend(_render_choice(child, depth + 1))
    lines.append(f"{pad}</Choice>")
    return lines


def build_label_studio_config(taxonomy: dict[str, object]) -> str:
    mothers = taxonomy.get("children")
    if not isinstance(mothers, list):
        raise ValueError("Taxonomy root must have a 'children' list.")

    lines: list[str] = [
        "<View>",
        "",
        '  <Style>',
        "    .paper-col{overflow-y:auto;padding:16px;border-right:2px solid #e8e8e8}",
        "    .annot-col{overflow-y:auto;padding:16px;background:#fafafa}",
        "  </Style>",
        "",
        '  <View style="display:flex;height:calc(100vh - 120px);">',
        "",
        "    <!-- LEFT: paper content -->",
        '    <View className="paper-col" style="flex:1;min-width:0;">',
        '      <Header value="$bibkey"/>',
        '      <HyperText name="pdf_link" value="&lt;a href=&quot;$pdf_url&quot; '
        'target=&quot;_blank&quot;&gt;Open PDF&lt;/a&gt;"/>',
        '      <Header value="Title"/>',
        '      <Text name="title" value="$title"/>',
        '      <Header value="Abstract"/>',
        '      <Text name="abstract" value="$abstract"/>',
        '      <Header value="Task description (auto-extracted)"/>',
        '      <Text name="paraphrase_task" value="$paraphrase_task"/>',
        "    </View>",
        "",
        "    <!-- RIGHT: annotation controls -->",
        '    <View className="annot-col" style="width:420px;flex-shrink:0;">',
        "",
        '      <Header value="Task category"/>',
        '      <Taxonomy name="task" toName="abstract"',
        '                placeholder="Type to filter or expand categories..."',
        '                leafOnly="true" filter="true" maxUsages="1">',
    ]

    for mother in mothers:
        if not isinstance(mother, dict):
            continue
        lines.extend(_render_choice(mother, depth=5))

    lines.extend([
        '          <Choice value="other" hint="No taxonomy category fits"/>',
        "        </Taxonomy>",
        "",
        "        <!-- Skip option: check this instead of filling the taxonomy -->",
        '        <Choices name="skip" toName="abstract" choice="single" layout="vertical">',
        '          <Choice value="not_classif_bench"',
        '                  hint="Not a classification benchmark — taxonomy not applicable"/>',
        "        </Choices>",
        "",
        '      <Header value="Notes (optional)"/>',
        '      <TextArea name="notes" toName="abstract" rows="4"',
        '                placeholder="Edge case justification, ambiguities..."/>',
        "    </View>",
        "  </View>",
        "",
        "</View>",
    ])

    return "\n".join(lines) + "\n"


def generate_config(
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    if not taxonomy_path.is_file():
        raise FileNotFoundError(f"Taxonomy not found: {taxonomy_path}")

    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    config = build_label_studio_config(taxonomy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(config, encoding="utf-8")
    logger.info("Wrote Label Studio config to %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Label Studio <Taxonomy> labeling config from taxonomy JSON.",
    )
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    generate_config(args.taxonomy, args.output)
