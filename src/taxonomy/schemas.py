import json
import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Label(str, enum.Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNSURE = "UNSURE"


class PaperBenchmarkEligibility(BaseModel):
    """Schema for benchmark eligibility (is this a classification benchmark paper?).

    Reasoning is generated first (chain-of-thought) so the LLM commits to a
    label only after structured analysis. Enum-like fields use plain strings
    to tolerate LLM variation.
    """

    reasoning: str = Field(
        description=(
            "Brief chain-of-thought: "
            "1) Does it introduce a new distinct dataset? "
            "2) Is the task text classification (discrete labels)? "
            "3) Any exclusion criteria triggered? "
            "Keep under 100 words."
        ),
    )
    label: Label
    justification: str = Field(
        description="One-sentence summary of the decision. Max 20 words.",
    )
    benchmark_names: list[str] | None = Field(
        description="Names of introduced benchmarks/datasets. Null if NEGATIVE.",
        default=None,
    )
    task_type: str | None = Field(
        description="Short snake_case classification task label. Null if NEGATIVE.",
        default=None,
    )
    input_type: str | None = Field(
        description="One of: sentence, sentence_pair, document. Null if NEGATIVE.",
        default=None,
    )
    label_type: str | None = Field(
        description="One of: binary, multi_class, multi_label. Null if NEGATIVE.",
        default=None,
    )
    domain: str | None = Field(
        description="Short snake_case application domain (e.g. biomedical, social_media). Null if NEGATIVE.",
        default=None,
    )
    languages: list[str] | None = Field(
        description="ISO 639-1 codes or ['multilingual']. Null if NEGATIVE.",
        default=None,
    )
    data_source: str | None = Field(
        description="One of: crowdsourced, expert_annotated, automatic, existing_corpus, web_scraped. Null if NEGATIVE.",
        default=None,
    )
    benchmark_novelty: str | None = Field(
        description="One of: new, extension, adaptation. Null if NEGATIVE.",
        default=None,
    )
    is_multitask: bool | None = Field(default=None)


class InputCardinality(str, enum.Enum):
    single_text = "single_text"
    pair_of_texts = "pair_of_texts"
    text_with_target = "text_with_target"
    sequence_of_turns = "sequence_of_turns"
    text_with_context = "text_with_context"


class InputUnit(str, enum.Enum):
    token = "token"
    span = "span"
    sentence = "sentence"
    short_text = "short_text"
    paragraph = "paragraph"
    document = "document"
    dialog_turn = "dialog_turn"
    dialog = "dialog"


class OutputCardinality(str, enum.Enum):
    single_label = "single_label"
    multi_label = "multi_label"
    ordinal = "ordinal"
    hierarchical = "hierarchical"
    span_plus_label = "span_plus_label"


class Confidence(str, enum.Enum):
    high = "high"
    medium = "medium"
    low = "low"


class AnnotationSource(str, enum.Enum):
    expert = "expert"
    crowd = "crowd"
    distant = "distant"
    silver = "silver"
    automatic = "automatic"
    unknown = "unknown"


class BenchmarkRecord(BaseModel):
    id: str = Field(description="snake_case, unique per (dataset, task) tuple")

    # TASK facet - ONLY field used as clustering signal for task-taxonomy
    paraphrase_task: str = Field(
        min_length=30,
        max_length=400,
        description="Short description of the judgment mechanism. See prompt rules.",
    )

    judgment_type: str = Field(
        description="snake_case, 2-6 words. LLM-proposed label naming the judgment family. "
        "Held out from clustering; used only for post-hoc cluster validation."
    )
    judgment_target: str = Field(
        description="free text, 2-6 words. What the judgment is made about. "
        "Held out from clustering; used only for post-hoc validation."
    )

    # INPUT facet (for future input-taxonomy, NOT for task clustering)
    input_cardinality: InputCardinality
    input_unit: InputUnit

    # OUTPUT facet (for future output-taxonomy, NOT for task clustering)
    output_cardinality: OutputCardinality
    label_space_size: int | Literal["open", "unknown"]
    num_classes: str = "unknown"
    annotation_source: AnnotationSource = AnnotationSource.unknown

    # DOMAIN facet (for future domain-taxonomy, NOT for task clustering)
    domain: str = Field(
        description="Broad snake_case domain label, 1-3 words (e.g. biomedical, social_media, news_media). 'general' if no specific domain.",
        default="general",
    )
    paraphrase_domain: str = Field(
        min_length=20,
        max_length=300,
        description="Concrete description of the text source and data nature without naming the domain. Clustering signal for domain-taxonomy.",
        default="",
    )
    languages: list[str] = Field(description="ISO 639-1 codes, or ['unknown']")

    # Resource dependency
    requires_external_knowledge: bool

    # Multi-task flag (independent of subtasks_not_inferable)
    is_multitask: bool = False

    # Grounding + audit
    evidence_passages: list[str] = Field(min_length=1, max_length=5)
    extraction_confidence: Confidence = Confidence.medium

    @field_validator("label_space_size", mode="before")
    @classmethod
    def _coerce_label_space_size(cls, v: object) -> object:
        if isinstance(v, str) and v not in ("open", "unknown"):
            return "unknown"
        return v

    @field_validator("num_classes", mode="before")
    @classmethod
    def _coerce_num_classes(cls, v: object) -> str:
        return str(v) if v is not None else "unknown"

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("id must be snake_case/kebab-case alphanumeric")
        return v.lower()


class ExtractionResult(BaseModel):
    reasoning: str = Field(
        min_length=100,
        description="CoT: record count decision; leakage risks avoided; confidence justification.",
    )
    records: list[BenchmarkRecord] = Field(min_length=1)
    subtasks_not_inferable: bool = False


class CoarseTask(str, enum.Enum):
    sentiment_opinion = "sentiment_opinion"
    emotion = "emotion"
    topic_subject = "topic_subject"
    intent = "intent"
    dialogue_act = "dialogue_act"
    nli = "nli"
    paraphrase_equivalence = "paraphrase_equivalence"
    qa_classification = "qa_classification"
    fact_claim_verification = "fact_claim_verification"
    stance_detection = "stance_detection"
    abusive_language = "abusive_language"
    argument_mining = "argument_mining"
    relation_classification = "relation_classification"
    figurative_language = "figurative_language"
    other = "other"


class CoarseTaskClassification(BaseModel):
    reasoning: str = Field(
        min_length=30,
        max_length=600,
        description=(
            "2-3 sentences: identify the benchmark's input, the label space, "
            "and which boundary rule applies if any."
        ),
    )
    coarse_task: CoarseTask


THEMATIC_DOMAINS: tuple[str, ...] = (
    "general",
    "news",
    "biomedical",
    "mental_health",
    "legal",
    "scientific",
    "education",
    "finance",
    "e_commerce",
    "literary",
    "politics",
    "linguistics",
    "other",
)


class DomainAssignment(BaseModel):
    reasoning: str = Field(
        min_length=20,
        description=(
            "2-3 short sentences: real-world area the text is drawn from (not the task); "
            "which boundary rule applies. Under 500 characters preferred."
        ),
    )
    thematic_domain: str = Field(
        description="Exactly one label from THEMATIC_DOMAINS.",
    )

    @field_validator("thematic_domain", mode="before")
    @classmethod
    def _coerce_thematic_domain(cls, v: object) -> str:
        normalized = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        return normalized if normalized in THEMATIC_DOMAINS else "other"


class LeafAssignment(BaseModel):
    reasoning: str = Field(
        min_length=40,
        max_length=3500,
        description=(
            "Follow PROCEDURE 1-5: the benchmark's decision and label space; "
            "the candidate mothers; the boundary rule applied; the chosen leaf."
        ),
    )
    mother: str = Field(description="Exactly one top-category label, or 'other'.")
    leaf: str | None = Field(
        default=None,
        description="Terminal node under the chosen mother; null only if mother='other'.",
    )

    @model_validator(mode="after")
    def _leaf_required_except_other(self) -> "LeafAssignment":
        if self.mother != "other" and self.leaf is None:
            raise ValueError("leaf must be set unless mother is 'other'")
        if self.mother == "other" and self.leaf is not None:
            raise ValueError("leaf must be null when mother is 'other'")
        return self


DEFAULT_MANUAL_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "taxonomy"
    / "manual_taxonomy_tree.json"
)


def load_manual_taxo(path: str | Path | None = None) -> dict[str, object]:
    taxo_path = Path(path) if path is not None else DEFAULT_MANUAL_TAXONOMY_PATH
    with taxo_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Manual taxonomy must be a JSON object.")
    return data


def _taxo_children(node: dict[str, object]) -> list[dict[str, object]]:
    children = node.get("children") or []
    if not isinstance(children, list):
        return []
    return [child for child in children if isinstance(child, dict)]


def _taxo_label(node: dict[str, object]) -> str:
    return str(node.get("label") or node.get("id") or "unknown").strip()


def _format_taxo_node(node: dict[str, object]) -> str:
    label = _taxo_label(node)
    description = str(node.get("description") or "").strip()
    return f"{label}: {description}" if description else label


def _build_taxo_block(taxonomy: dict[str, object]) -> str:
    mothers = _taxo_children(taxonomy) or [taxonomy]
    lines: list[str] = []
    for mother in mothers:
        lines.append(f"- {_format_taxo_node(mother)}")
        leaves = _taxo_children(mother) or [mother]
        for leaf in leaves:
            lines.append(f"  - {_format_taxo_node(leaf)}")
    return "\n".join(lines)


def _build_valid_taxo(taxonomy: dict[str, object]) -> dict[str, set[str]]:
    mothers = _taxo_children(taxonomy) or [taxonomy]
    valid: dict[str, set[str]] = {}
    for mother in mothers:
        mother_label = _taxo_label(mother)
        leaves = _taxo_children(mother) or [mother]
        valid[mother_label] = {_taxo_label(leaf) for leaf in leaves}
    return valid


def is_valid_leaf_assignment(
    mother: object,
    leaf: object,
    valid_taxonomy: dict[str, set[str]] | None = None,
) -> bool:
    if mother == "other":
        return leaf is None
    if not isinstance(mother, str):
        return False
    if leaf is not None and not isinstance(leaf, str):
        return False

    valid = valid_taxonomy if valid_taxonomy is not None else VALID_TAXONOMY
    return mother in valid and leaf in valid[mother]


try:
    _MANUAL_TAXO = load_manual_taxo()
except FileNotFoundError:
    _MANUAL_TAXO = {"id": "forest_root", "label": "forest_root", "children": []}

TAXO_BLOCK = _build_taxo_block(_MANUAL_TAXO)
VALID_TAXONOMY = _build_valid_taxo(_MANUAL_TAXO)


_TAXONOMY_PROMPT = """You are an expert NLP researcher classifying academic papers.
Your task is to determine if the paper explicitly INTRODUCES a NEW benchmark or dataset specifically for TEXT CLASSIFICATION.

Use the `reasoning` field to think step-by-step BEFORE committing to a label.

### STEP 1: CORE ELIGIBILITY (Must meet BOTH criteria)
1. **New Resource:** The paper must INTRODUCE, RELEASE, or CREATE a new dataset. It must be identifiable as a distinct resource (e.g., named, or described as new labeled data that can be distinguished as a resource). The paper does not need to use the word "benchmark" or claim availability.
   - *Fail:* Papers that only use, evaluate on, or compare against existing datasets. Shared task overview papers are NEGATIVE unless they release a novel dataset.
2. **Text Classification Task:** The dataset must be for text classification (predicting a discrete, fixed-set label for a given text or text pair). Examples: sentiment analysis, NLI, topic classification, stance detection, fact verification, relation classification (when entities are given), paraphrase detection, metaphor detection.
   - *Note on Multi-task:* If it's a multi-task benchmark, 75% of the sub-tasks must be text classification.

### STEP 2: EXCLUSION CRITERIA (If ANY apply, the paper is NEGATIVE)
- **Non-Classification Outputs:** The task requires extracting spans, token-level labels (NER, POS, code-switching at token level), targeted sentiment requiring span extraction, rankings, continuous scores (e.g. semantic similarity on a continuous scale), structured outputs, or generation (summarization, translation, QA, dialogue).
- **Diagnostic/Meta-Evaluation:** The dataset is a diagnostic test set designed to evaluate a non-classification system (like MT or embeddings), even if some sub-tasks involve classification.
- **Intermediate Step:** The data collection/annotation is just an intermediate step to train a method, without being presented as a distinct resource.
- **Ambiguity:** The task definition is ambiguous between classification and a non-classification formulation.

### STEP 3: DECISION RULE
- If it passes STEP 1 and triggers NONE of the criteria in STEP 2 -> POSITIVE.
- If it fails STEP 1 OR triggers ANY criteria in STEP 2 -> NEGATIVE.
- When in doubt -> NEGATIVE.

### EXTRACTION RULES (Only if POSITIVE)
If POSITIVE, extract the requested taxonomy fields.
- Only use information explicitly stated or strongly implied by the title and abstract.
- Do NOT hallucinate values. Set fields to null if the information is absent.
- For `task_type` and `domain`, use short snake_case labels.
- For `languages`, use ISO 639-1 codes or "multilingual".
If NEGATIVE, all extraction fields must be strictly set to null.
"""


_TAXONOMY_EXTRACTION_PROMPT = """You extract structured records from an NLP paper that introduces text-classification benchmarks. Scope has been validated upstream — do not re-check.

# UNIT
One record per (dataset × task × output-setup). Emit multiple records when:
- same dataset, semantically distinct tasks → one per task
- same task, distinct output setups (e.g. binary vs ordinal over the same judgment) → one per setup, paraphrase_task (near-)identical, output fields differ
- multiple datasets → one per dataset
- If the abstract does not provide enough information to decompose subtasks,
  emit a single aggregated record and set subtasks_not_inferable: true.

# paraphrase_task — the clustering signal (critical)

This is the ONLY field used to cluster the task-taxonomy. Clustering must DISCOVER families; you must NOT pre-label them.

Describe the judgment MECHANISM concretely: what property/relation is determined, what the decision is about, what kind of reasoning is involved.

Hard exclusions (each leaks into another facet or pre-labels the cluster):
- Task-family names: sentiment analysis, NLI, fact-checking, topic classification, stance detection, toxicity detection, intent detection, emotion classification, sarcasm detection, paraphrase detection, etc.
- Output label names: positive/negative, entailment/contradiction, supported/refuted, toxic, etc.
- Cardinality words: binary, ternary, multi-class, multi-label, 3-way, 5-class.
- Specific domains: medical, legal, clinical, twitter, youtube, reddit, movie review, etc.
- Dataset/benchmark names.
- Annotation-schema rationale ("designed to emphasize X", "fine-grained to capture Y").
- Method terms (prompting, zero-shot, fine-tuned).
- Input/output cardinality phrasing ("the input is a single text", "outputs a single label", "a pair of sentences") — these belong in their own fields.

Style: 1-2 sentences, 30-80 tokens, present tense. Use mechanism vocabulary, not task-name vocabulary.

Self-check: if a reader could recover the specific domain, labels, number of classes, or input/output cardinality from your paraphrase alone → leak, rewrite. If you wrote "the task is X" → pre-label, rewrite.

# OTHER FACET FIELDS

Fill from paper text. These are held out from clustering and used for post-hoc validation — be specific here:
- input_cardinality, input_unit, output_cardinality, label_space_size
- num_classes: exact int if stated, else binary (<2 classes) / few (3–10) / many (11–100) / extreme (>100) / unknown
- annotation_source: expert / crowd / distant / silver / automatic / unknown — how gold labels were produced
- domain: broad snake_case label (1-3 words, e.g. biomedical, social_media, news_media, legal, finance, general)
- paraphrase_domain: 1-2 sentences describing the text source and data nature concretely — what kind of text, from where, what makes it characteristic. Do NOT name the domain, platform, or genre directly (same exclusion principle as paraphrase_task). 20-60 tokens.
- requires_external_knowledge (true iff the task definition structurally requires consulting external information beyond the input)
- judgment_type, judgment_target (you MAY name the family here — these fields exist for validation and are not clustered)
- is_multitask: true iff the paper introduces multiple semantically distinct tasks (set independently of whether subtasks are inferable from the abstract)

# GROUNDING

Each record: 1-5 evidence_passages, verbatim quotes from the paper. They must jointly justify the task exists, labels are discrete, and any specific claim (e.g. label_space_size). Never invent numbers, class counts, or languages — use "open" / ["unknown"].

Set extraction_confidence: high (all fields verbatim-supported) / medium (1-2 inferred) / low (several inferred, needs full text).

subtasks_not_inferable: set to true when the paper is multi-dataset or multi-task but the abstract does not describe individual tasks with sufficient detail to emit separate records. In this case, describe the aggregate judgment in paraphrase_task and set extraction_confidence: low.

# REASONING (first, mandatory)

4-6 short sentences:
1. How many records and why.
2. Which abstract terms you EXCLUDED from paraphrase_task (name them).
3. What mechanism wording you used instead.
4. Confidence and what the full paper would resolve.

# OUTPUT

Single JSON conforming to ExtractionResult. No text outside.
"""


USER_PROMPT_TEMPLATE = """\
PAPER TEXT:
<
{paper_text}
>>>
"""

_TAXONOMY_EXTRACTION_USER_PROMPT_TEMPLATE = """PAPER TEXT:
<
{paper_text}
>>>
"""

_TAXONOMY_COARSE_LABEL_EXTRACTION_PROMPT = """ You are classifying an NLP paper that introduces a text classification benchmark into one of 13 predefined coarse task categories, based on its title and abstract.

# Coarse task categories (choose exactly ONE)

- **sentiment_opinion**: Predict sentiment polarity, valence, or opinion toward a target/aspect. Includes review rating, ABSA, opinion mining.
- **emotion**: Predict discrete emotions (anger, joy, fear...) or emotion dimensions. Distinct from sentiment polarity.
- **topic_subject**: Assign topic, subject, genre, or thematic category to a text. Includes news categorization, document classification by subject.
- **intent**: Predict user intent or goal in an utterance/query. Typically conversational or search (ATIS, SNIPS, CLINC style).
- **dialogue_act**: Classify the communicative function of an utterance in dialogue (question, statement, acknowledgment...). Structure of conversation, not content.
- **nli**: Determine entailment / contradiction / neutrality between a premise and a hypothesis.
- **paraphrase_equivalence**: Decide whether two texts are semantically equivalent or paraphrases (MRPC, PAWS, QQP style).
- **qa_classification**: Question answering framed as classification: yes/no QA, answer selection, answer triggering. EXCLUDES multiple-choice QA and span extraction.
- **fact_claim_verification**: Verify a claim against evidence (supported / refuted / not enough info). FEVER, SciFact style.
- **stance_detection**: Predict stance (favor / against / neutral) of a text toward an EXPLICIT target (entity, topic, proposition).
- **abusive_language**: Detect toxic, hateful, offensive, or abusive content. Umbrella covering toxicity, hate speech, offensive language, cyberbullying.
- **argument_mining**: Classify argumentative components (claim, premise, evidence) or argument quality/relation.
- **relation_classification**: Predict the relation type between two GIVEN entities in a text. Entities are provided, not extracted.
- **figurative_language**: Detect figurative uses of language — irony, sarcasm, metaphor, hyperbole. Binary or multi-class detection of non-literal meaning.

Use **other** only if the benchmark clearly does not fit any of the 13 categories.

# Boundary rules (critical)

- News categorization → `topic_subject`.
- Hate speech, toxicity, offensive → all `abusive_language`.
- Claim + evidence with factuality judgment → `fact_claim_verification`; text + target with attitude → `stance_detection`.
- Entailment-style pairs without explicit claim verification framing → `nli`.
- Relation extraction where entities must first be identified → `other` (not classification).
- Multi-choice QA → `other`.
- Irony, sarcasm, metaphor, figurative uses → figurative_language (NOT sentiment_opinion, even when sarcasm flips polarity).
- If the paper introduces several benchmarks spanning different categories, pick the one that is the paper's **primary contribution** (usually the one named in the title or most developed in the abstract). Use `other` only if they are truly equally central.
- Ignore methodological contributions (new model, new loss) — classify by the **benchmark task**, not the method.

# Output format

Respond with ONLY a JSON object, no preamble.
"""
_COARSE_USER_PROMPT_TEMPLATE = "{paper_text}"


_TAXONOMY_LEAF_ASSIGNMENT_PROMPT = """You assign a text-classification benchmark task paraphrase to one terminal node of a FIXED, hand-curated task taxonomy. Scope is validated upstream — the task IS a classification benchmark, do not re-check.

You make ONE decision: the terminal taxonomy leaf whose task mechanism matches the provided paraphrase. The taxonomy has top categories (mothers) and terminal leaves. Always assign to a LEAF. If no terminal leaf fits, set mother to "other" and leaf to null.

# TAXONOMY (mother → leaves, with operational definitions)

{taxonomy_block}

# PROCEDURE (use the reasoning field, in this order)
1. State the judgment mechanism described by the task paraphrase.
2. Name the 2-3 candidate MOTHERS it could fall under. If only one is plausible, say so.
3. Resolve between candidates using the BOUNDARY RULES below. Name the rule you applied.
4. Within the chosen mother, pick the terminal LEAF whose definition matches.
5. If no terminal leaf fits, set mother to "other" and leaf to null.

# BOUNDARY RULES (the adjacencies where assignment fails — apply explicitly)
- Attitude. Sentiment & Opinion = polarity/opinion the text itself expresses, no external target needed. Stance = position toward an EXPLICIT target given alongside the text, not inferred from polarity. Stereotype & Social Bias = generalizing attribution to a social group, even when phrased without hostility. A text can be positive in sentiment yet against a target in stance — classify by WHAT is judged, not surface polarity.
- Veracity. Fact & Claim = truth of a claim assessed against evidence or world knowledge. Event Factuality = whether an event is presented as actually holding (factuality/modality of the event mention), with no external evidence check. Deception = intent to mislead inferred from the text's own properties, no ground-truth check. Machine-Generated Text = provenance (machine vs human), not truth.
- Harm. Abusive Language = the text attacks or demeans a person/group (harmful content). Stereotype & Social Bias = group generalization, not necessarily abusive. Social Norm = whether described behavior conforms to a social/moral norm (the norm is the target, not a victim). Mental Health / Wellbeing = the author's psychological state, not harm directed at others.
- Inference. NLI = entailment/contradiction/neutral between a given premise and hypothesis. Logical Reasoning = validity of a deductive/formal inference. Commonsense Plausibility = plausibility of a situation under world knowledge. Classify by which property is judged: semantic entailment vs formal validity vs plausibility.
- Non-literal vs Sentiment. Irony, sarcasm, humor, satire, metaphor → Non-literal, even when sarcasm flips sentiment polarity.
- Conversational. Intent = the goal behind an utterance (what the speaker wants). Dialogue Act = the communicative function of the turn (question, acknowledgment...), independent of content.

Always classify by the JUDGMENT MECHANISM described in the paraphrase, never by the domain, platform, benchmark name, or paper branding.

# OUTPUT
A single JSON conforming to the schema. No text outside it. reasoning first.
"""


_TAXONOMY_LEAF_ASSIGNMENT_USER_PROMPT_TEMPLATE = """TASK PARAPHRASE:
{paper_text}
"""

_THEMATIC_DOMAIN_ASSIGNMENT_PROMPT = """You assign the APPLICATION DOMAIN of a text-classification
benchmark, from its title and abstract, using a fixed controlled vocabulary. Choose exactly ONE
thematic domain.

The thematic domain is the SUBJECT-MATTER AREA the annotated text is about. It is independent of:
- the TASK (what is predicted),
- the LANGUAGE of the text,
- the PLATFORM/channel where the text was posted (that is register, handled separately),
- the textual GENRE.

### CORE PRINCIPLE
Decide from what the text content is ABOUT. Then apply the arbitration rules below — they resolve
the recurring cases where subject-matter is unclear or only carried by the task.

### Controlled vocabulary (choose exactly ONE)
- general: the text is about no specific subject-matter area. This is the DEFAULT and is common.
- news: journalistic reporting — the annotated text is press articles or headlines written by
  journalists. A news article that happens to be about politics stays `news` if the text is
  journalistic reporting (reserve `politics` for political discourse itself; see below).
- biomedical: medicine, clinical records, health, pharmacology, public health, neurology.
- mental_health: psychological distress, depression, anxiety, addiction, suicidality, counseling.
- legal: law, regulation, contracts, court rulings, government administration, public policy.
- scientific: assign ONLY when the ANNOTATED TEXT itself is scholarly/technical content
  (research paper text, mathematical statements). Do NOT assign scientific merely because the
  benchmark studies language models or is framed as a research evaluation — that describes
  almost every benchmark. Verb-bias / numeracy probes on ordinary sentences are general.
- education: pedagogy, educational content, learner/essay assessment.
- finance: finance, corporate, business, marketing.
- e_commerce: product reviews, hospitality, tourism, consumer goods.
- literary: fiction, literature, cultural heritage, historical texts.
- politics: assign ONLY when the text is PRIMARILY political discourse or directly about the
  political process — parliamentary records, political speeches, party manifestos, electoral
  campaign material, or text explicitly about named elections/parties/legislation/government.
  A topic being publicly debated or societally significant (climate, COVID, military, gender)
  does NOT by itself make it `politics`; judge the actual dominant subject instead.
- other: a genuine specific area not covered above (military, gaming, cybersecurity, HR).
  Use sparingly; never as a substitute for general.

### Arbitration rules (general — apply in order)
1. TASK IS NOT A DOMAIN. A benchmark whose only specificity is what is predicted — sentiment,
   stance, emotion, hate, bias, sarcasm, metaphor, irony, entailment, acceptability — does NOT
   take that phenomenon as its domain. Judge the underlying text's subject matter instead.
2. CONSTRUCTED-PROBE RULE. If the text is synthetic or crowdsourced specifically to probe a
   phenomenon and has no real subject matter of its own (e.g. bias/stereotype minimal pairs,
   diagnostic test items), the domain is general — even if the task targets a social attribute
   (gender, race, religion). The attribute is the task, not the subject of the text.
3. LANGUAGE IS NOT A DOMAIN. A benchmark in a non-English language or about a dialect/variety
   (NLI in Arabic, sentiment in Bangla, dialect data) takes the subject-matter domain of its text
   (usually general). The language/variety is recorded by a separate facet, not here.
4. PLATFORM IS NOT A DOMAIN. The posting channel (Twitter, Reddit, Wikipedia, forum) never sets
   the domain. Judge the subject matter: "depression posts on Reddit" -> mental_health;
   "tweets with no specific subject" -> general.
5. MIXED SOURCES -> GENERAL. If the text is drawn from several unrelated areas (e.g. a corpus
   sampling news + Wikipedia + reviews + political discourse), do NOT pick one of them. A
   multi-source corpus with no single dominant area is `general`. Assign a specific domain only
   when ONE area clearly dominates the text content.
6. SUBJECT MATTER, NOT MENTION. Assign a specific domain only when the text content is genuinely
   and primarily ABOUT that area, not merely because it mentions a related term or includes it
   among several sources. When two real areas apply, pick the one the benchmark is primarily
   built around (named in the title / most developed in the abstract). Only ONE domain is allowed.
7. NO GUESSING. If the abstract does not establish a specific subject-matter area, choose general.
   Do not infer a domain from weak cues (dataset name, the task alone, a single keyword).
8. Reason first in `reasoning` (2-3 short sentences, under 500 characters), then commit.

### Output
Respond with ONLY a JSON object conforming to the schema. No preamble.
"""

_THEMATIC_DOMAIN_USER_PROMPT_TEMPLATE = "{title}\n\n{abstract}"


LANGUAGE_EVIDENCE = frozenset(
    {"explicit_mention", "inferred_from_name", "default_assumed"}
)


class LanguageExtraction(BaseModel):
    reasoning: str = Field(
        min_length=20,
        description=(
            "1-2 short sentences: language(s) of the annotated text and the cue used "
            "(explicit mention, proper name, script). Under 500 characters preferred."
        ),
    )
    languages: list[str] = Field(
        description=(
            "ISO 639-3 codes for the annotated benchmark text, ['mul'] if massively "
            "multilingual and not enumerable, or ['und'] if no language signal."
        ),
    )
    evidence: str = Field(
        default="default_assumed",
        description=(
            "How the language was determined: explicit_mention | inferred_from_name | "
            "default_assumed."
        ),
    )

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, v: object) -> str:
        normalized = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        return normalized if normalized in LANGUAGE_EVIDENCE else "default_assumed"

    @field_validator("languages", mode="before")
    @classmethod
    def _normalize_languages(cls, v: object) -> list[str]:
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return ["und"]
        codes = [str(c).strip().lower() for c in v if str(c).strip()]
        return codes if codes else ["und"]


_LANGUAGE_ASSIGNMENT_PROMPT = """You extract the LANGUAGE(S) of the annotated text in a
text-classification benchmark, from its title and abstract. Output ISO 639-3 codes.

The language is that of the BENCHMARK'S TEXT DATA (the labeled documents/sentences/posts),
NOT the language the paper is written in, nor a language merely mentioned or compared against.
Example: an English-written paper introducing a Bangla corpus -> ben.

### RULES
1. List one 639-3 code per language of the labeled text (eng, fra, arb, swh, lao...).
   Small named set (<=8) -> list each. Massively multilingual, not enumerable (XNLI/XTREME) -> ['mul'].
2. Use a code ONLY if you are certain it is the real 639-3 code for that language/variety.
   If unsure of the exact code, back off to the macrolanguage (any Arabic dialect you can't code -> arb)
   or to ['und']. NEVER invent or guess a code to fit a variety.
3. Dialect/variety -> closest specific code you are sure of (Levantine Arabic -> apc), else arb.
4. Code-switched/translated data -> list EVERY language present (['eng','hin']); never collapse to one.
5. `default_assumed` and ['und'] are locked together: if you cannot point to an explicit
   mention or a proper-name/script cue, you MUST output ['und'] with evidence='default_assumed'.
   Do NOT output ['eng'] (or any code) as a default guess — "most NLP benchmarks are English",
   "AI-generated text is usually English", or domain/topic cues are NOT signals. A real code
   requires evidence='explicit_mention' or 'inferred_from_name'. When in doubt -> ['und'].
6. Set evidence: explicit_mention | inferred_from_name | default_assumed.
7. Reason in 1-2 sentences, then commit.

Respond with ONLY a JSON object conforming to the schema. No preamble."""

_LANGUAGE_ASSIGNMENT_USER_PROMPT_TEMPLATE = "{title}\n\n{abstract}"


class ArbitratedLink(BaseModel):
    url: str | None = Field(
        default=None,
        description="Must be a URL appearing verbatim among the candidates; null otherwise.",
    )
    host_type: str | None = None
    is_official: bool = Field(
        default=False,
        description="True if introduced or released by this paper.",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LinkArbitration(BaseModel):
    reasoning: str = Field(
        min_length=20,
        description=(
            "Chosen candidate, deciding context verb, and why others were rejected or accepted "
            "as official. Under 500 characters preferred."
        ),
    )
    dataset_link: list[ArbitratedLink] = Field(
        default_factory=list,
        description="Every official URL of the introduced dataset (e.g. a HF + GitHub mirror). [] if none.",
    )
    code_link: list[ArbitratedLink] = Field(
        default_factory=list,
        description="Every official URL of the released code. [] if none.",
    )
    data_availability: Literal["open", "on_request", "restricted", "none"] = "none"

    @field_validator("dataset_link", "code_link", mode="before")
    @classmethod
    def _coerce_link_list(cls, v: object) -> object:
        # Tolère l'ancien format (objet unique ou null) + le LLM qui renvoie un seul objet -> liste.
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        return v


_LINK_ARBITER_PROMPT = """You arbitrate which candidate URL is the RESOURCE OFFICIALLY INTRODUCED
by THIS paper — separately for dataset and code. Input: title, abstract, and numbered candidates,
each with a zone (abstract/body/footnote/back/ref/pdf_annot/metadata) and its context sentence.

### CORE PRINCIPLE
Decide on the VERB of the context, not the host or URL shape. A link is official only if THIS paper
presents the resource as its own. A valid GitHub/dataset URL is still REJECTED if it is a dependency,
a compared resource, or a third-party tool.
EXCEPTION — no verb needed: a clickable link to a resource host (github, huggingface, zenodo, osf,
codalab, gitlab) in the ABSTRACT/TITLE or a PAGE-1 clickable footnote (zone `pdf_annot` whose context
says "page 1", or zone `abstract`) whose URL PATH echoes the paper's own benchmark/dataset name (e.g.
title "TAAROFBench" <-> huggingface.co/.../TAAROFBENCH) IS the authors' official resource — they post
it as a logo/shortcut with no sentence. Otherwise: no release verb -> reject.

### OFFICIAL vs REJECT
- OFFICIAL: author-release language — "we release/introduce/present/build/gather/collect/annotate",
  or "our {data|corpus|code} is (publicly) available at", "data and models available at".
- REJECT: "we use/used/based on/trained with" (dependency, tool, framework); "we compare against/
  following/hosted by/taken from" (compared or prior resource); license/doc/homepage links; funding
  or compute (HPC/grant) acknowledgements; publisher DOIs; cited references. Zone `ref` is a citation
  context -> reject unless it is an explicit release of THIS paper's resource.

### RULES (in order)
1. VERB OVER HOST. Institutional/unknown host + release verb = official; github.com + "we used" = not.
2. VERBATIM ONLY. Each URL must appear exactly as in a candidate. Never invent, repair, complete, or
   merge. If none qualifies, return [] for that field.
3. dataset_link and code_link are LISTS — include EVERY official URL. If the dataset is mirrored on
   several hosts (e.g. Hugging Face AND GitHub both hosting the data), put ALL of them in dataset_link.
   Same for code. A repo that holds both data and code goes in both lists. Empty list if none.
4. FOOTNOTE: context is "{calling sentence} [note] {note text}" — judge on the calling sentence; the
   note often holds only the URL.
5. SAME RESOURCE, different path depths -> return the most specific non-garbled one.
6. Multiple authored releases -> pick the one the paper is built around (title / most of abstract).
7. AVAILABILITY IS SEPARATE FROM THE LINK. If this paper's own resource has a candidate URL, RETURN it
   (is_official=true) EVEN under restriction — restriction never nulls the URL. Then set
   data_availability: "open" = public/direct download; "on_request" = "upon request"/contact authors;
   "restricted" = agreement/license/IRB/sensitive. "none" only when the paper releases no dataset of its
   own. Leave the list empty only when no candidate is this paper's resource (the access note alone, no link).
8. NO GUESSING. A bare URL with no verb is insufficient -> empty list, EXCEPT the abstract/title/page-1
   named-resource case in CORE PRINCIPLE.
9. Reason first in `reasoning` (chosen candidate, deciding verb, why others rejected; <500 chars).

### EXAMPLES
- [footnote] "we gather a custom dataset [note] https://grouplens.org/datasets/..." -> dataset_link,
  open. Release verb makes it official despite a non-github host.
- [back] "we used Pyphen (github.com/Kozea/Pyphen)" -> REJECT (dependency).
- [body] "we compare against CEFRLex (...)" -> REJECT (third-party).
- [ref] "...trained using fairseq (github.com/facebookresearch/fairseq)" -> REJECT (cited tool).
- [body] "AfroLID is publicly available at github.com/UBC-NLP/afrolid" -> code_link (+dataset_link if
  the repo hosts the data), open.
- [pdf_annot] "[clickable link, page 1]" huggingface.co/datasets/smksaha/apt-eval, title "APT-Eval..."
  -> dataset_link, is_official, open. Page-1 logo whose path matches the paper's name = authors' release.
- "we release APT-Eval on HF and GitHub: huggingface.co/datasets/.../apt-eval, github.com/.../apt-eval"
  -> dataset_link = BOTH urls (mirror); open.
- "our corpus is available to researchers upon signing an agreement (github.com/x/y)" -> dataset_link =
  [github.com/x/y], restricted. Keep the URL; restriction is NOT empty.
- "the dataset is available upon request" with no URL candidate -> dataset_link = [], on_request.

Respond with ONLY a JSON object conforming to the schema. No preamble."""


@dataclass(frozen=True)
class TaskConfig:
    """Bundles prompt, schema, and tool metadata for a labelling task."""

    system_prompt: str
    response_model: type[BaseModel]
    tool_name: str
    tool_description: str
    max_tokens: int = 256
    user_prompt_template: str = "{paper_text}"

    def render_user_prompt(self, paper_text: str | None = None, **kwargs) -> str:
        if paper_text is not None:
            kwargs["paper_text"] = paper_text
        return self.user_prompt_template.format(**kwargs)

    @property
    def tool_schema(self) -> dict:
        return {
            "name": self.tool_name,
            "description": self.tool_description,
            "input_schema": self.response_model.model_json_schema(),
        }

    @property
    def result_columns(self) -> list[str]:
        return list(self.response_model.model_fields.keys())


TASKS: dict[str, TaskConfig] = {
    # Canonical task names
    "benchmark_eligibility": TaskConfig(
        system_prompt=_TAXONOMY_PROMPT,
        response_model=PaperBenchmarkEligibility,
        tool_name="classify_paper",
        tool_description="Screen a paper for classification-benchmark eligibility.",
        max_tokens=1024,
        user_prompt_template="{paper_text}",
    ),
    "benchmark_record_extraction": TaskConfig(
        system_prompt=_TAXONOMY_EXTRACTION_PROMPT,
        response_model=ExtractionResult,
        tool_name="extract_taxonomy_records",
        tool_description=(
            "Extract grounded benchmark records for task-taxonomy and HAC clustering preparation."
        ),
        max_tokens=2048,
        user_prompt_template=_TAXONOMY_EXTRACTION_USER_PROMPT_TEMPLATE,
    ),
    "coarse_task_classification": TaskConfig(
        system_prompt=_TAXONOMY_COARSE_LABEL_EXTRACTION_PROMPT,
        response_model=CoarseTaskClassification,
        tool_name="extract_taxonomy_records_coarse",
        tool_description=(
            "Extract coarse-grained benchmark records for task-taxonomy and HAC clustering preparation."
        ),
        max_tokens=2048,
        user_prompt_template=_COARSE_USER_PROMPT_TEMPLATE,
    ),
    "taxonomy_leaf_assignment": TaskConfig(
        system_prompt=_TAXONOMY_LEAF_ASSIGNMENT_PROMPT.format(
            taxonomy_block=TAXO_BLOCK
        ),
        response_model=LeafAssignment,
        tool_name="assign_leaf",
        tool_description=(
            "Assign a classification-benchmark paper to a fixed taxonomy mother/leaf node."
        ),
        max_tokens=1024,
        user_prompt_template=_TAXONOMY_LEAF_ASSIGNMENT_USER_PROMPT_TEMPLATE,
    ),
    "thematic_domain_assignment": TaskConfig(
        system_prompt=_THEMATIC_DOMAIN_ASSIGNMENT_PROMPT,
        response_model=DomainAssignment,
        tool_name="assign_thematic_domain",
        tool_description=(
            "Assign a classification-benchmark paper to one controlled application domain."
        ),
        max_tokens=512,
        user_prompt_template=_THEMATIC_DOMAIN_USER_PROMPT_TEMPLATE,
    ),
    "link_arbiter": TaskConfig(
        system_prompt=_LINK_ARBITER_PROMPT,
        response_model=LinkArbitration,
        tool_name="arbitrate_resource_links",
        tool_description=(
            "Choose official dataset/code URLs among PDF-extracted link candidates."
        ),
        max_tokens=1024,
        user_prompt_template="{paper_text}",
    ),
    "language_assignment": TaskConfig(
        system_prompt=_LANGUAGE_ASSIGNMENT_PROMPT,
        response_model=LanguageExtraction,
        tool_name="extract_benchmark_languages",
        tool_description=(
            "Extract ISO 639-3 language codes for the annotated benchmark text from title and abstract."
        ),
        max_tokens=256,
        user_prompt_template=_LANGUAGE_ASSIGNMENT_USER_PROMPT_TEMPLATE,
    ),
}

DEFAULT_TASK = "benchmark_eligibility"
