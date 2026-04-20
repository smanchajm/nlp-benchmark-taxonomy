import enum
from dataclasses import dataclass

from pydantic import BaseModel, Field


class Label(str, enum.Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNSURE = "UNSURE"


class PaperTaxonomy(BaseModel):
    """Rich schema for the 'taxonomy' task.

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


@dataclass(frozen=True)
class TaskConfig:
    """Bundles prompt, schema, and tool metadata for a labelling task."""

    system_prompt: str
    response_model: type[BaseModel]
    tool_name: str
    tool_description: str
    max_tokens: int = 256

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
    "taxonomy": TaskConfig(
        system_prompt=_TAXONOMY_PROMPT,
        response_model=PaperTaxonomy,
        tool_name="classify_paper",
        tool_description="Classify an NLP paper and extract taxonomy metadata.",
        max_tokens=1024,
    ),
}

DEFAULT_TASK = "taxonomy"
