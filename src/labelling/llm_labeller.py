import enum
import json
import logging
import os
from dataclasses import dataclass

import instructor
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MODEL_MAP: dict[str, str] = {
    "mistral": "mistral-small-latest",
    "claude": "claude-3-haiku-20240307",
    "gemini": "gemini-1.5-flash-latest",
}


class Label(str, enum.Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNSURE = "UNSURE"


# ── Filter task: simple yes/no classification ────────────────────────────


class PaperClassification(BaseModel):
    """Lightweight schema for the 'filter' task."""

    label: Label
    justification: str = Field(description="Max 20 words")
    task_family: str | None = Field(
        description='Short task label (e.g. "sentiment", "NLI", "NER"). null if no new dataset.',
        default=None,
    )
    is_multitask: bool | None = Field(
        description="True if multi-task benchmark (e.g. GLUE). null for NEGATIVE.",
        default=None,
    )


_FILTER_PROMPT = """You classify NLP papers. Determine if the paper INTRODUCES a benchmark/dataset for evaluating TEXT CLASSIFICATION.

POSITIVE: Introduces a benchmark/dataset where a model predicts a discrete label from a fixed set for each text input (or text pair), e.g., sentiment analysis, NLI, topic classification, fact verification, relation classification. Multi-task benchmarks are POSITIVE if ≥75% of sub-tasks are classification.

NEGATIVE if either:
- The paper does NOT introduce a new evaluation dataset (method/model papers, surveys, training-only datasets, shared tasks without new data).
- The paper introduces a dataset for a task whose output is not a single discrete label per input: spans, token-level labels, rankings, structured outputs (e.g., NER, parsing, MCQA, extractive QA, generation, alignment, translation, summarization, dialogue, multimodal benchmarks requiring non-textual inputs).
- The paper introduces a diagnostic test set or meta-evaluation benchmark designed to evaluate a non-classification system (e.g., MT, embeddings, summarization), even if some sub-tasks involve classification.

UNSURE: The paper describes collecting or annotating data but does not clearly frame it as a reusable benchmark, OR it is a multi-task benchmark near the 75% classification boundary or where the abstract does not provide enough detail to quantify the proportion of classification sub-tasks, OR the task definition is ambiguous between classification and a non-classification formulation.

task_family: short label for the task type (e.g., "sentiment", "NLI", "NER", "summarization"). Required when the paper introduces a dataset. Set null when NEGATIVE because no new dataset.

is_multitask: true if the benchmark bundles several sub-tasks into one evaluation suite (e.g. GLUE, SuperGLUE). false otherwise. Set null for NEGATIVE.

Justification: name the task type and explain why it is or is not classification. If NEGATIVE because no new dataset, state that. Max 20 words."""


# ── Taxonomy task: rich metadata extraction ──────────────────────────────


class PaperTaxonomy(BaseModel):
    """Rich schema for the 'taxonomy' task.

    Enum-like fields use plain strings to tolerate LLM variation.
    Expected values are documented in field descriptions for schema guidance.
    """

    label: Label
    justification: str = Field(
        description="Max 20 words. Name the task type and explain why it is or is not classification.",
    )
    benchmark_names: list[str] | None = Field(
        description="Names of introduced benchmarks/datasets. Null if NEGATIVE.",
        default=None,
    )
    task_type: str | None = Field(
        description="Short snake_case classification task label (e.g. sentiment_analysis). Null if NEGATIVE.",
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


_TAXONOMY_PROMPT = """You classify NLP papers. Determine if the paper INTRODUCES a benchmark/dataset for evaluating TEXT CLASSIFICATION.

POSITIVE: Introduces a benchmark/dataset where a model predicts a discrete label from a fixed set for each text input (or text pair), e.g., sentiment analysis, NLI, topic classification, fact verification, relation classification. Multi-task benchmarks are POSITIVE if ≥75% of sub-tasks are classification.
NEGATIVE if either:
- The paper does NOT introduce a new evaluation dataset (method/model papers, surveys, training-only datasets, shared tasks without new data).
- The paper introduces a dataset for a task whose output is not a single discrete label per input: spans, token-level labels, rankings, structured outputs (e.g., NER, parsing, MCQA, extractive QA, generation, alignment, translation, summarization, dialogue, multimodal benchmarks requiring non-textual inputs).
- The paper introduces a diagnostic test set or meta-evaluation benchmark designed to evaluate a non-classification system (e.g., MT, embeddings, summarization), even if some sub-tasks involve classification.
UNSURE: The paper describes collecting or annotating data but does not clearly frame it as a reusable benchmark, OR it is a multi-task benchmark near the 75% classification boundary, OR the task definition is ambiguous between classification and a non-classification formulation.

When POSITIVE or UNSURE, fill all taxonomy fields. When NEGATIVE, set all taxonomy fields to null.
For task_type and domain, use your best judgment with a short snake_case label — do not constrain to a fixed list.
For languages, use ISO 639-1 codes or "multilingual"."""


# ── Task configuration registry ──────────────────────────────────────────


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
    "filter": TaskConfig(
        system_prompt=_FILTER_PROMPT,
        response_model=PaperClassification,
        tool_name="classify_paper",
        tool_description="Return the classification for the paper.",
    ),
    "taxonomy": TaskConfig(
        system_prompt=_TAXONOMY_PROMPT,
        response_model=PaperTaxonomy,
        tool_name="classify_paper",
        tool_description="Classify an NLP paper and extract taxonomy metadata.",
        max_tokens=512,
    ),
}

DEFAULT_TASK = "taxonomy"


# ── Core classification ──────────────────────────────────────────────────


def _classify_paper(
    client,
    provider: str,
    title: str,
    abstract: str,
    task: str = DEFAULT_TASK,
) -> BaseModel:
    """Classify a single paper."""
    cfg = TASKS[task]
    try:
        return client.chat.completions.create(
            model=MODEL_MAP[provider],
            response_model=cfg.response_model,
            messages=[
                {"role": "system", "content": cfg.system_prompt},
                {
                    "role": "user",
                    "content": f"Title: {title}\nAbstract: {abstract}",
                },
            ],
            max_retries=3,
        )
    except Exception as e:
        logger.error("Classification failed for '%s': %s", title, e)
        return cfg.response_model(
            label=Label.UNSURE,
            justification="API Error",
        )


def _create_client(provider: str):
    """Create an instructor-wrapped client for the given provider.

    Provider-specific SDKs are imported lazily so users only need the SDK
    for the provider they actually use.
    """
    if provider == "mistral":
        from mistralai import Mistral

        return instructor.from_mistral(Mistral(api_key=os.getenv("MISTRAL_API_KEY")))
    if provider == "claude":
        from anthropic import AsyncAnthropic

        return instructor.from_anthropic(
            AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
    if provider == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return instructor.from_gemini(genai.GenerativeModel("gemini-1.5-flash-latest"))
    raise ValueError(f"Unknown provider: {provider!r}")


def label_papers(
    df: pd.DataFrame,
    provider: str = "mistral",
    task: str = DEFAULT_TASK,
) -> pd.DataFrame:
    """Classify all papers in *df* sequentially and return the augmented DataFrame."""
    client = _create_client(provider)

    logger.info("Classifying %d papers via %s (task=%s)...", len(df), provider, task)
    results = [
        _classify_paper(client, provider, r.title, str(r.abstract), task=task)
        for r in df.itertuples()
    ]

    res_df = pd.DataFrame([r.model_dump() for r in results]).rename(
        columns=lambda c: f"llm_{c}"
    )
    return pd.concat([df.reset_index(drop=True), res_df], axis=1)


# ── Mistral Batch API ────────────────────────────────────────────────────


def mistral_batch_submit(
    df: pd.DataFrame,
    model: str = "mistral-small-latest",
    task: str = DEFAULT_TASK,
) -> str:
    """Submit a Mistral Batch API job with auto-generated JSON schema."""
    from mistralai import Mistral
    from mistralai.models import BatchRequest

    cfg = TASKS[task]
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "PaperClass",
            "schema": cfg.response_model.model_json_schema(),
            "strict": True,
        },
    }

    requests = [
        BatchRequest(
            custom_id=str(row["bibkey"]),
            body={
                "model": model,
                "messages": [
                    {"role": "system", "content": cfg.system_prompt},
                    {
                        "role": "user",
                        "content": f"Title: {row['title']}\nAbstract: {row['abstract']}",
                    },
                ],
                "response_format": schema,
                "temperature": 0.0,
            },
        )
        for _, row in df.iterrows()
    ]

    job = client.batch.jobs.create(
        endpoint="/v1/chat/completions", model=model, requests=requests
    )
    logger.info("Batch job submitted: %s", job.id)
    return job.id


def mistral_batch_results(
    job_id: str,
    df: pd.DataFrame,
    task: str = DEFAULT_TASK,
) -> pd.DataFrame:
    """Fetch and merge results from a completed Mistral Batch job."""
    from mistralai import Mistral

    cfg = TASKS[task]
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    job = client.batch.jobs.get(job_id=job_id)
    if job.status != "SUCCESS":
        raise RuntimeError(f"Batch job not finished (status: {job.status})")

    output_file = client.files.download(file_id=job.output_file)
    results = []
    for line in output_file.read().decode("utf-8").strip().split("\n"):
        entry = json.loads(line)
        content = entry["response"]["body"]["choices"][0]["message"]["content"]
        parsed = cfg.response_model.model_validate_json(content)
        data = parsed.model_dump()
        data["bibkey"] = entry["custom_id"]
        results.append(data)

    res_df = pd.DataFrame(results).rename(
        columns={c: f"mistral_{c}" for c in cfg.result_columns}
    )
    return df.merge(res_df, on="bibkey", how="left")


# ── Claude Batch API ─────────────────────────────────────────────────────


def claude_batch_submit(
    df: pd.DataFrame,
    model: str = "claude-haiku-4-5-20251001",
    task: str = DEFAULT_TASK,
) -> str:
    """Submit an Anthropic Message Batches job."""
    from anthropic import Anthropic

    cfg = TASKS[task]
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    requests = [
        {
            "custom_id": str(row["bibkey"]),
            "params": {
                "model": model,
                "max_tokens": cfg.max_tokens,
                "system": cfg.system_prompt,
                "tools": [cfg.tool_schema],
                "tool_choice": {"type": "tool", "name": cfg.tool_name},
                "messages": [
                    {
                        "role": "user",
                        "content": f"Title: {row['title']}\nAbstract: {row['abstract']}",
                    }
                ],
            },
        }
        for _, row in df.iterrows()
    ]

    batch = client.messages.batches.create(requests=requests)
    logger.info("Claude batch submitted: %s", batch.id)
    return batch.id


def claude_batch_results(
    batch_id: str,
    df: pd.DataFrame,
    task: str = DEFAULT_TASK,
) -> pd.DataFrame:
    """Fetch and merge results from a completed Anthropic Message Batch."""
    from anthropic import Anthropic

    cfg = TASKS[task]
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        raise RuntimeError(f"Batch not finished (status: {batch.processing_status})")

    results = []
    errors = 0
    for entry in client.messages.batches.results(batch_id):
        if entry.result.type != "succeeded":
            logger.error("Failed for %s: %s", entry.custom_id, entry.result.type)
            errors += 1
            continue
        tool_input = entry.result.message.content[0].input
        try:
            parsed = cfg.response_model.model_validate(tool_input)
        except Exception as e:
            logger.warning("Validation error for %s: %s", entry.custom_id, e)
            errors += 1
            continue
        data = parsed.model_dump()
        data["bibkey"] = entry.custom_id
        results.append(data)

    if errors:
        logger.warning(
            "Skipped %d entries due to errors (out of %d)",
            errors,
            errors + len(results),
        )

    res_df = pd.DataFrame(results).rename(
        columns={c: f"claude_{c}" for c in cfg.result_columns}
    )
    return df.merge(res_df, on="bibkey", how="left")
