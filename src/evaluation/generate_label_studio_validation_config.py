"""Generate Label Studio validation config for personal expert annotation.

Fields validated per paper:
  - Task category (same <Taxonomy> as the inter-annotator config)
  - Dataset link quality (valid / broken / wrong target / none)
  - Language: dropdown to confirm or correct detected value (shown always — many are "und")
  - Domain: correct/incorrect + free-text correction when wrong
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
DEFAULT_OUTPUT_PATH = DATA / "evaluation" / "label_studio_validation_config.xml"

_INDENT = "  "

# ISO 639-3 codes present in data/corpus/single_task_benchmark_paper.parquet,
# sorted by descending frequency. Display name follows the code.
_LANGUAGES: list[tuple[str, str]] = [
    ("und", "Undetermined"),
    ("eng", "English"),
    ("deu", "German"),
    ("fra", "French"),
    ("cmn", "Mandarin Chinese"),
    ("mul", "Multilingual"),
    ("arb", "Arabic (Standard)"),
    ("spa", "Spanish"),
    ("kor", "Korean"),
    ("ita", "Italian"),
    ("hin", "Hindi"),
    ("ben", "Bengali"),
    ("jpn", "Japanese"),
    ("por", "Portuguese"),
    ("rus", "Russian"),
    ("ron", "Romanian"),
    ("zho", "Chinese (generic)"),
    ("fas", "Persian/Farsi"),
    ("tur", "Turkish"),
    ("dan", "Danish"),
    ("ind", "Indonesian"),
    ("nld", "Dutch"),
    ("bul", "Bulgarian"),
    ("ces", "Czech"),
    ("tam", "Tamil"),
    ("tel", "Telugu"),
    ("pol", "Polish"),
    ("urd", "Urdu"),
    ("heb", "Hebrew"),
    ("ell", "Greek"),
    ("fin", "Finnish"),
    ("kan", "Kannada"),
    ("ary", "Moroccan Arabic"),
    ("swe", "Swedish"),
    ("hrv", "Croatian"),
    ("est", "Estonian"),
    ("guj", "Gujarati"),
    ("mal", "Malayalam"),
    ("mar", "Marathi"),
    ("arz", "Egyptian Arabic"),
    ("amh", "Amharic"),
    ("slv", "Slovenian"),
    ("swh", "Swahili"),
    ("yor", "Yoruba"),
    ("pan", "Punjabi"),
    ("nor", "Norwegian"),
    ("srp", "Serbian"),
    ("eus", "Basque"),
    ("cat", "Catalan"),
    ("lit", "Lithuanian"),
    ("ukr", "Ukrainian"),
    ("hun", "Hungarian"),
    ("lav", "Latvian"),
    ("vie", "Vietnamese"),
    ("tha", "Thai"),
    ("yue", "Cantonese"),
    ("hau", "Hausa"),
    ("ibo", "Igbo"),
    ("other", "Other (specify in notes)"),
]


def _esc(value: str) -> str:
    return saxutils.escape(value, {"'": "&apos;", '"': "&quot;"})


def _render_choice(node: dict, depth: int) -> list[str]:
    pad = _INDENT * depth
    label = _esc(str(node["label"]))
    desc = node.get("description")
    hint = f' hint="{_esc(str(desc))}"' if desc else ""
    children = [c for c in (node.get("children") or []) if isinstance(c, dict)]

    if not children:
        return [f'{pad}<Choice value="{label}"{hint}/>']

    lines = [f'{pad}<Choice value="{label}"{hint}>']
    for child in children:
        lines.extend(_render_choice(child, depth + 1))
    lines.append(f"{pad}</Choice>")
    return lines


def build_validation_config(taxonomy: dict[str, object]) -> str:
    mothers = taxonomy.get("children")
    if not isinstance(mothers, list):
        raise ValueError("Taxonomy root must have a 'children' list.")

    lines: list[str] = [
        "<View>",
        "",
        "  <Style>",
        "    .paper-col{overflow-y:auto;padding:16px;border-right:2px solid #e8e8e8}",
        "    .annot-col{overflow-y:auto;padding:16px;background:#fafafa}",
        "    .section{margin-top:12px;padding-top:10px;border-top:1px solid #e0e0e0}",
        "  </Style>",
        "",
        '  <View style="display:flex;height:calc(100vh - 120px);">',
        "",
        "    <!-- LEFT: paper + current metadata to validate -->",
        '    <View className="paper-col" style="flex:1;min-width:0;">',
        '      <Header value="$bibkey"/>',
        '      <HyperText name="pdf_link" value="&lt;a href=&quot;$pdf_url&quot; target=&quot;_blank&quot;&gt;Open PDF&lt;/a&gt;"/>',
        '      <Header value="Title"/>',
        '      <Text name="title" value="$title"/>',
        '      <Header value="Abstract"/>',
        '      <Text name="abstract" value="$abstract"/>',
        '      <Header value="Task description (auto-extracted)"/>',
        '      <Text name="paraphrase_task" value="$paraphrase_task"/>',
        "    </View>",
        "",
        "    <!-- RIGHT: validation controls -->",
        '    <View className="annot-col" style="width:460px;flex-shrink:0;">',
        "",
        "      <!-- ── TASK ──────────────────────────────────────── -->",
        '      <Header value="Task category"/>',
        '      <Taxonomy name="task" toName="abstract"',
        '                placeholder="Type to filter or expand categories..."',
        '                leafOnly="true" filter="true" maxUsages="1">',
    ]

    for mother in mothers:
        if not isinstance(mother, dict):
            continue
        lines.extend(_render_choice(mother, depth=4))

    lines.extend([
        '        <Choice value="other" hint="No taxonomy category fits"/>',
        "      </Taxonomy>",
        "",
        "      <!-- ── DATASET LINK ──────────────────────────────── -->",
        '      <View className="section">',
        '        <Header value="Dataset link"/>',
        # Show the actual URL from data so annotator can click it
        '        <HyperText name="ds_link" value="&lt;a href=&quot;$dataset_url&quot; target=&quot;_blank&quot;&gt;$dataset_url&lt;/a&gt;"/>',
        '        <Choices name="link_status" toName="abstract" choice="single" required="true" layout="vertical">',
        '          <Choice value="valid"    hint="Link works and points to the correct dataset"/>',
        '          <Choice value="broken"   hint="Link is dead or returns 404"/>',
        '          <Choice value="wrong"    hint="Link resolves but points to the wrong resource"/>',
        '          <Choice value="none"     hint="No dataset link was found"/>',
        "        </Choices>",
        "      </View>",
        "",
        "      <!-- ── LANGUAGE ──────────────────────────────────── -->",
        '      <View className="section">',
        '        <Header value="Language"/>',
        '        <Text name="detected_lang" value="Detected: $benchmark_languages"/>',
        "        <!-- Always shown — many entries have 'und'; select the actual language -->",
        '        <Choices name="language" toName="abstract" choice="multiple" required="true"',
        '                 layout="select" placeholder="Select correct language...">',
    ])

    for code, name in _LANGUAGES:
        lines.append(f'          <Choice value="{_esc(code)}" hint="{_esc(name)}"/>')

    lines.extend([
        "        </Choices>",
        "      </View>",
        "",
        "      <!-- ── DOMAIN ────────────────────────────────────── -->",
        '      <View className="section">',
        '        <Header value="Domain"/>',
        '        <Text name="detected_domain" value="Current: $thematic_domain"/>',
        '        <Choices name="domain_ok" toName="abstract" choice="single" required="true" layout="vertical">',
        '          <Choice value="correct"   hint="Domain label is accurate"/>',
        '          <Choice value="incorrect" hint="Domain label is wrong or too coarse"/>',
        "        </Choices>",
        "        <!-- Correction field — visible only when domain is wrong -->",
        '        <View visibleWhen="choice-selected" whenTagName="domain_ok" whenChoiceValue="incorrect">',
        '          <TextArea name="domain_correction" toName="abstract" rows="1"',
        '                    placeholder="Correct domain (e.g. biomedical, legal, social media…)"/>',
        "        </View>",
        "      </View>",
        "",
        "      <!-- ── SKIP / NOTES ──────────────────────────────── -->",
        '      <View className="section">',
        '        <Choices name="skip" toName="abstract" choice="single" layout="vertical">',
        '          <Choice value="not_classif_bench" hint="Not a classification benchmark — skip all fields"/>',
        "        </Choices>",
        '        <Header value="Notes (optional)"/>',
        '        <TextArea name="notes" toName="abstract" rows="3"',
        '                  placeholder="Corrections, edge cases, ambiguities..."/>',
        "      </View>",
        "",
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
    config = build_validation_config(taxonomy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(config, encoding="utf-8")
    logger.info("Wrote Label Studio validation config to %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Label Studio validation config (task + link + language + domain).",
    )
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    generate_config(args.taxonomy, args.output)
