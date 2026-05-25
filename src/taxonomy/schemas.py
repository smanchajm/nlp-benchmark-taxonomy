import enum
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator


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


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy LLM-merge: schemas + prompts (consumed by src/taxonomy/llm_merge.py)
# ─────────────────────────────────────────────────────────────────────────────


def _validate_snake_case(v: str) -> str:
    v = v.strip().lower().replace(" ", "_").replace("-", "_")
    if not v.replace("_", "").isalnum():
        raise ValueError(f"label must be snake_case alphanumeric, got: {v!r}")
    return v


class LeafLabelGen(BaseModel):
    """LLM output for the per-leaf labelling step."""

    reasoning: str = Field(
        min_length=20,
        description=(
            "Think step by step: (1) identify the shared cognitive operation across "
            "all paraphrases, (2) flag any paraphrase that describes a different "
            "mechanism and name it explicitly, (3) identify the nearest adjacent task "
            "this cluster could be confused with and state what distinguishes it."
        ),
    )
    label: str = Field(
        min_length=3,
        max_length=60,
        description=(
            "snake_case cognitive judgment name, 2-5 words. "
            "Forbidden as sole content-bearing suffix: classification, detection, assessment, "
            "verification, prediction, processing, task, evaluation ."
        ),
    )
    description: str = Field(
        min_length=30,
        description=(
            "1-2 sentences naming the cluster's specific judgment mechanism. "
            "MUST start with 'Heterogeneous cluster: ' if the paraphrases cover "
            "more than one distinct judgment mechanism."
        ),
    )

    @field_validator("label")
    @classmethod
    def _label_snake_case(cls, v: str) -> str:
        return _validate_snake_case(v)


class LeafCluster(BaseModel):
    """A taxonomy leaf: an HAC cluster enriched with LLM-generated label/description."""

    id: str
    label: str
    description: str
    paraphrases: list[str] = Field(min_length=1, max_length=10)
    size: int = Field(ge=1)
    member_indices: list[int]
    coherence: float
    hac_id: int


class MergeProposal(BaseModel):
    """One proposed N-ary merge of pool nodes into a parent."""

    children_ids: list[str] = Field(min_length=2, max_length=5)
    parent_label: str = Field(min_length=3, max_length=40)
    parent_description: str = Field(min_length=20, max_length=400)
    justification: str = Field(
        min_length=20,
        description="1-2 sentences citing concrete paraphrases that justify the merge.",
    )
    confidence: float = Field(ge=0.6, le=1.0)

    @field_validator("parent_label")
    @classmethod
    def _label_snake_case(cls, v: str) -> str:
        v = _validate_snake_case(v)
        # Anti-conjonctif structurel
        tokens = v.split("_")
        if "and" in tokens or "or" in tokens:
            raise ValueError(
                f"label '{v}' contains conjunction '_and_'/'_or_' — refactor as a single concept"
            )
        # Anti-générique pur (déjà dans prompt mais on durcit)
        forbidden_solo = {
            "classification",
            "judgment",
            "identification",
            "evaluation",
            "interpretation",
            "analysis",
            "assessment",
            "detection",
        }
        if len(tokens) == 1 and v in forbidden_solo:
            raise ValueError(f"label '{v}' is too generic on its own")
        return v


class MergeRound(BaseModel):
    """Output of one merge-proposal LLM call. Empty merges → stop signal."""

    reasoning: str = Field(
        min_length=80,
        description="Overall reasoning: which thematic axes you considered, which nodes you set aside as niches.",
    )
    merges: list[MergeProposal] = Field(default_factory=list)


class MergeNode(BaseModel):
    """A node in the LLM-built taxonomy tree (leaf or internal)."""

    id: str
    label: str
    description: str
    paraphrases: list[str]
    size: int
    children: list[str] = Field(default_factory=list)
    iteration_created: int  # 0 for HAC leaves
    confidence: float | None = None  # confidence of the merge that created this node


_LEAF_LABEL_SYSTEM_PROMPT = """You are an NLP taxonomy expert. You receive a sample of paraphrases \
drawn from a single cluster of NLP benchmark task descriptions (the most central members of an HAC cluster).
Your job: produce a specific snake_case label and a short description for this cluster.

### PRIORITY CHECK (before writing anything else)
Read all paraphrases and ask: do they all describe the same cognitive operation?
If they cover more than one distinct judgment mechanism (e.g. some assess complexity, \
others assess grammatical correctness, others assess pragmatic appropriateness), \
you MUST start the description with "Heterogeneous cluster: " and explicitly enumerate \
what varies — do not abstract over the differences.
This check is not optional — when in doubt, apply it.

### Rules for the label
- Name the COGNITIVE JUDGMENT performed by the model, not the domain of application.
  Good: `stance_toward_target`, `numerical_claim_verification`, `implicit_hate_speech_detection`
  Bad:  `legal_text_classification`, `clinical_nlp`, `arabic_dialect_task`
- Exception: domain IS part of the label only when it defines the task itself \
(e.g. `dialect_identification`, `code_switching_detection`).
- 2 to 5 words, snake_case, lowercase.
- Forbidden generic suffixes used as the sole content-bearing word: \
`_classification`, `_detection`, `_assessment`, `_verification`, `_prediction`, \
`_processing`, `_task`, `_evaluation`. A suffix is allowed only when the preceding words \
are already specific enough to stand alone (e.g. `implicit_hate_speech_detection` is valid \
because `implicit_hate_speech` is specific; `safety_compliance_assessment` is not).
- When two plausible labels would differ only by a modifier (e.g. `deductive_X` vs \
`textual_X`), name the distinguishing mechanism explicitly rather than adding an adjective.
- Do NOT use `X_or_Y` patterns. If the cluster is heterogeneous, name the dominant \
mechanism only in the label and describe what varies in the description.

### Rules for the description
- 2-3 sentences maximum.
- Sentence 1: name the core judgment mechanism — what cognitive operation is performed \
on what input to produce what output.
- Sentence 2: name the key feature that distinguishes this cluster from adjacent tasks \
(e.g. if the task resembles sentiment analysis but is target-dependent, say so explicitly; \
if it resembles hate speech detection but focuses on implicit stereotypes rather than \
explicit hostility, say so). This sentence is mandatory.
- Do not copy any paraphrase verbatim.

Output: a single JSON conforming to the LeafLabelGen schema. No prose outside the JSON."""

_LEAF_LABEL_USER_PROMPT_TEMPLATE = """Cluster size: {n} papers.
Representative paraphrases (most central first):
{paraphrases_block}
Produce label + description."""


_MERGE_PROPOSAL_SYSTEM_PROMPT = """You are an NLP taxonomist building a hierarchical taxonomy \
of NLP classification benchmarks via iterative agglomerative merging.

Each round you receive a LOCAL WINDOW of semantically similar nodes (not the full pool). \
Window similarity creates implicit pressure to merge — resist it. Empty output (`merges: []`) \
is valid and often correct; there is no quota.

Use the `reasoning` field to think step-by-step BEFORE committing to any merge proposal.

### STEP 1 — IDENTIFY CANDIDATE GROUPINGS
For each potential subset of 2-5 nodes, ask: do they share a SINGLE unified concept that \
goes beyond surface lexical overlap or "all are NLP/classification"?
- Inspect paraphrases, not just labels.
- A node's best parent may live outside this window — overlapping windows give every node \
multiple chances. When unsure, set the candidate aside.

### STEP 2 — APPLY EXCLUSION GATES (any HIT → reject the candidate)
Reject any candidate merge if ANY of these conditions hold:
- **Conjunction gate**: the parent label or description enumerates TRULY DIFFERENT \
FAMILIES (e.g., "harm_and_deception", "emotion_or_stance"). \
EXCEPTION: same family at different specificities/contexts is fine — \
`emotion_classification` covering general + dialogue emotion is one concept; \
a subcase merged with its broader family (gender bias under social bias) is one concept.
- **Generic-abstraction gate**: the most precise label you can assign uses banned terms as \
standalone or compound: `classification`, `judgment`, `identification`, `evaluation`, \
`interpretation`, `analysis`, `assessment`, `text_analysis`, `nlp_task`.
- **Confidence gate**: your honest confidence in the merge is below 0.7 (i.e., not even \
"defensible reading"). Forced merges score below 0.7 — DO NOT propose them.

### STEP 3 — NAME AND JUSTIFY (only candidates that passed STEP 2)
- Prefer ESTABLISHED community names: `nli`, `fact_verification`, `stance_detection`, \
`aspect_based_sentiment`, `wsd`, `hate_speech_detection`, `emotion_classification`, \
`dialogue_act_classification`, `ai_text_detection`, `paraphrase_detection`. \
Invent a descriptive label only when no canonical term fits. snake_case, 1-3 tokens.
- Cite specific paraphrase ids in the justification (e.g., "a3.p2, a7.p1 both target stance").
- Each node may appear in AT MOST ONE merge per round.

### STEP 4 — DECISION RULE
- A candidate that passes STEP 2 → include in `merges`.
- A candidate that fails any gate → DO NOT propose. In `reasoning`, list AT MOST 5 \
candidates you genuinely considered (close calls), with one-line analyses (≤25 words each) \
and the gate that killed each. Do not enumerate obviously unrelated pairs.
- If no candidate survives all gates → return `merges: []`.

When in doubt → DO NOT MERGE.

Output: a single JSON conforming to MergeRound. No prose outside.
"""

_MERGE_PROPOSAL_USER_PROMPT_TEMPLATE = """Iteration {iteration}. Window size: {pool_size} nodes \
(local neighborhood, not the full pool).

WINDOW NODES:
{pool_block}

Identify defensible merges (2-5 nodes each). Leave niches and out-of-place nodes unmerged — \
they will be re-evaluated in other windows. If no defensible merge exists in this window, \
return an empty merges list with reasoning.

Note on late iterations: as iteration grows, nodes are themselves merged subtrees. Resist \
grouping them under abstract umbrellas (`judgment`, `evaluation`, `analysis`, `interpretation`); \
a defensible late merge must point to a specific recognized sub-family, not an invented chapeau."""


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
    "taxonomy_leaf_label": TaskConfig(
        system_prompt=_LEAF_LABEL_SYSTEM_PROMPT,
        response_model=LeafLabelGen,
        tool_name="generate_leaf_label",
        tool_description="Generate a specific snake_case label and description for an HAC leaf cluster.",
        max_tokens=1024,
        user_prompt_template=_LEAF_LABEL_USER_PROMPT_TEMPLATE,
    ),
    "taxonomy_merge_proposal": TaskConfig(
        system_prompt=_MERGE_PROPOSAL_SYSTEM_PROMPT,
        response_model=MergeRound,
        tool_name="propose_merges",
        tool_description="Propose N-ary merges of the current pool into parent nodes.",
        max_tokens=2048,
        user_prompt_template=_MERGE_PROPOSAL_USER_PROMPT_TEMPLATE,
    ),
}

DEFAULT_TASK = "benchmark_eligibility"
