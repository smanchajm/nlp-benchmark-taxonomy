import enum
import json
import logging
import os
from dataclasses import dataclass

import instructor
import pandas as pd
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

MODEL_MAP: dict[str, str] = {
    "mistral": "mistral-large-latest",
    "claude": "claude-3-haiku-20240307",
    "gemini": "gemini-1.5-flash-latest",
    "deepseek": "deepseek-chat",
    "openrouter": "google/gemma-4-31b-it",
}


class Label(str, enum.Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNSURE = "UNSURE"


# Taxonomy task


class PaperTaxonomy(BaseModel):
    """Rich schema for the 'taxonomy' task.

    Reasoning is generated first (chain-of-thought) so the LLM commits to a
    label only after structured analysis.  Enum-like fields use plain strings
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


_TAXONOMY_PROMPT = """You are an expert NLP researcher classifying academic papers.
Your task is to determine if the paper explicitly INTRODUCES a NEW benchmark or dataset specifically for TEXT CLASSIFICATION.

Use the `reasoning` field to think step-by-step BEFORE committing to a label.

### STEP 1: CORE ELIGIBILITY (Must meet BOTH criteria)
1. **New Resource:** The paper must INTRODUCE, RELEASE, or CREATE a new dataset. It must be identifiable as a distinct resource (e.g., named, or described as new labeled data that can be distinguished as a resource). The paper does not need to use the word "benchmark" or claim availability.
   - *Fail:* Papers that only use, evaluate on, or compare against existing datasets. Shared task overview papers are NEGATIVE unless they release a novel dataset.
2. **Text Classification Task:** The dataset must be for text classification (predicting a discrete, fixed-set label for a given text or text pair). Examples: sentiment analysis, NLI, topic classification, stance detection, fact verification, relation classification (when entities are given), paraphrase detection, metaphor detection.
   - *Note on Multi-task:* If it's a multi-task benchmark, ≥75% of the sub-tasks must be text classification.

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

#  Task configuration registry


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


# Core classification


def _create_client(provider: str, *, async_: bool = False):
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
    if provider == "deepseek":
        if async_:
            from openai import AsyncOpenAI

            return instructor.from_openai(
                AsyncOpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com",
                )
            )
        from openai import OpenAI

        return instructor.from_openai(
            OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            )
        )
    if provider == "openrouter":
        if async_:
            from openai import AsyncOpenAI

            return instructor.from_openai(
                AsyncOpenAI(
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                )
            )
        from openai import OpenAI

        return instructor.from_openai(
            OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        )
    raise ValueError(f"Unknown provider: {provider!r}")


async def _classify_paper_async(
    client,
    provider: str,
    title: str,
    abstract: str,
    task: str = DEFAULT_TASK,
) -> BaseModel:
    """Classify a single paper (async)."""
    cfg = TASKS[task]
    try:
        return await client.chat.completions.create(
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


async def label_papers_async(
    df: pd.DataFrame,
    provider: str = "deepseek",
    task: str = DEFAULT_TASK,
    max_concurrent: int = 20,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    """Classify papers concurrently with a semaphore-based pool.

    Args:
        max_concurrent: Max parallel API calls (default: 20).
        checkpoint_path: If set, save intermediate results periodically.
        checkpoint_every: How often to save checkpoints (default: 50).
    """
    import asyncio

    client = _create_client(provider, async_=True)
    sem = asyncio.Semaphore(max_concurrent)
    results: list[tuple[int, dict]] = []
    pbar = tqdm(total=len(df), desc=f"{provider}/{task}")

    async def _process(idx: int, title: str, abstract: str):
        async with sem:
            res = await _classify_paper_async(
                client, provider, title, abstract, task=task
            )
            results.append((idx, res.model_dump()))
            pbar.update(1)
            if checkpoint_path and len(results) % checkpoint_every == 0:
                _save_checkpoint_indexed(df, results, checkpoint_path, prefix=provider)

    tasks = [
        _process(i, r.title, str(r.abstract)) for i, r in enumerate(df.itertuples())
    ]
    await asyncio.gather(*tasks)
    pbar.close()

    # Sort by original order
    results.sort(key=lambda x: x[0])
    res_df = pd.DataFrame([r for _, r in results]).rename(
        columns=lambda c: f"{provider}_{c}"
    )

    if checkpoint_path:
        _save_checkpoint_indexed(df, results, checkpoint_path, prefix=provider)

    return pd.concat([df.reset_index(drop=True), res_df], axis=1)


def _save_checkpoint_indexed(
    df: pd.DataFrame, results: list[tuple[int, dict]], path: str, prefix: str = "llm"
) -> None:
    sorted_results = sorted(results, key=lambda x: x[0])
    res_df = pd.DataFrame([r for _, r in sorted_results]).rename(
        columns=lambda c: f"{prefix}_{c}"
    )
    idxs = [i for i, _ in sorted_results]
    out = pd.concat([df.iloc[idxs].reset_index(drop=True), res_df], axis=1)
    out.to_parquet(path, index=False)
    logger.info("Checkpoint saved: %d/%d papers → %s", len(results), len(df), path)


# Mistral Batch API


def mistral_batch_submit(
    df: pd.DataFrame,
    model: str = "mistral-large-latest",
    task: str = DEFAULT_TASK,
) -> str:
    """Submit a Mistral Batch API job via file upload (handles large batches)."""
    import tempfile

    from mistralai import Mistral

    cfg = TASKS[task]
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "PaperClass",
            "schema": cfg.response_model.model_json_schema(),
        },
    }

    lines = []
    for _, row in df.iterrows():
        entry = {
            "custom_id": str(row["bibkey"]),
            "body": {
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
        }
        lines.append(json.dumps(entry, ensure_ascii=False))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(lines))
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as fh:
            batch_data = client.files.upload(
                file={"file_name": "batch.jsonl", "content": fh},
                purpose="batch",
            )
    finally:
        os.unlink(tmp_path)

    job = client.batch.jobs.create(
        input_files=[batch_data.id],
        model=model,
        endpoint="/v1/chat/completions",
    )
    logger.info("Batch job submitted: %s (file: %s)", job.id, batch_data.id)
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


# Claude Batch API


def claude_batch_submit(
    df: pd.DataFrame,
    model: str = "claude-haiku-4-5-20251001",
    task: str = DEFAULT_TASK,
) -> str:
    """Submit an Anthropic Message Batches job."""
    from anthropic import Anthropic

    cfg = TASKS[task]
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    tool_schema = cfg.tool_schema
    tool_schema["cache_control"] = {"type": "ephemeral"}

    requests = [
        {
            "custom_id": str(row["bibkey"]),
            "params": {
                "model": model,
                "max_tokens": cfg.max_tokens,
                "system": [
                    {
                        "type": "text",
                        "text": cfg.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "tools": [tool_schema],
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


# Google Batch API


def _flatten_google_schema(schema: dict) -> dict:
    """Convert a Pydantic JSON Schema to a Google-compatible schema.

    Resolves $defs/$ref, replaces anyOf nullable patterns with nullable flag,
    and strips unsupported keys (title, description, default, $defs).
    """
    defs = schema.get("$defs", {})

    def _resolve(node: dict) -> dict:
        if "$ref" in node:
            ref_name = node["$ref"].rsplit("/", 1)[-1]
            return _resolve(defs[ref_name])

        out = {}

        # Handle anyOf nullable pattern: anyOf: [{type: X}, {type: null}]
        if "anyOf" in node:
            variants = [v for v in node["anyOf"] if v.get("type") != "null"]
            has_null = any(v.get("type") == "null" for v in node["anyOf"])
            if len(variants) == 1:
                out = _resolve(variants[0])
                if has_null:
                    out["nullable"] = True
                return out

        if "type" in node:
            out["type"] = node["type"].upper() if node["type"] != "null" else "STRING"

        if "enum" in node:
            out["enum"] = node["enum"]

        if "properties" in node:
            out["properties"] = {k: _resolve(v) for k, v in node["properties"].items()}

        if "required" in node:
            out["required"] = node["required"]

        if "items" in node:
            out["items"] = _resolve(node["items"])

        if "description" in node:
            out["description"] = node["description"]

        return out

    return _resolve(schema)


def google_batch_submit(
    df: pd.DataFrame,
    model: str = "gemini-3-flash-preview",
    task: str = DEFAULT_TASK,
) -> str:
    """Submit a Google GenAI Batch job via file upload."""
    import tempfile

    from google import genai
    from google.genai import types

    cfg = TASKS[task]
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    schema = _flatten_google_schema(cfg.response_model.model_json_schema())
    lines = []
    for _, row in df.iterrows():
        entry = {
            "key": str(row["bibkey"]),
            "request": {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": f"Title: {row['title']}\nAbstract: {row['abstract']}"
                            }
                        ]
                    },
                ],
                "systemInstruction": {
                    "parts": [{"text": cfg.system_prompt}],
                },
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": schema,
                    "temperature": 0.0,
                },
            },
        }
        lines.append(json.dumps(entry, ensure_ascii=False))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(lines))
        tmp_path = f.name

    try:
        uploaded = client.files.upload(
            file=tmp_path,
            config=types.UploadFileConfig(
                display_name=f"batch-{task}",
                mime_type="jsonl",
            ),
        )
    finally:
        os.unlink(tmp_path)

    job = client.batches.create(
        model=model,
        src=uploaded.name,
        config=types.CreateBatchJobConfig(
            display_name=f"nlp-taxonomy-{task}",
        ),
    )
    logger.info("Google batch submitted: %s (file: %s)", job.name, uploaded.name)
    return job.name


def google_batch_results(
    job_name: str,
    df: pd.DataFrame,
    task: str = DEFAULT_TASK,
) -> pd.DataFrame:
    """Fetch and merge results from a completed Google GenAI Batch job."""
    from google import genai
    from google.genai import types

    cfg = TASKS[task]
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    job = client.batches.get(name=job_name)
    if job.state != types.JobState.JOB_STATE_SUCCEEDED:
        raise RuntimeError(f"Batch job not finished (state: {job.state})")

    raw = client.files.download(file=job.dest.file_name)
    results = []
    errors = 0
    for line in raw.decode("utf-8").strip().split("\n"):
        entry = json.loads(line)
        bibkey = entry.get("key")
        if entry.get("error"):
            logger.error("Failed for %s: %s", bibkey, entry["error"])
            errors += 1
            continue
        try:
            text = entry["response"]["candidates"][0]["content"]["parts"][0]["text"]
            parsed = cfg.response_model.model_validate_json(text)
        except Exception as e:
            logger.warning("Validation error for %s: %s", bibkey, e)
            errors += 1
            continue
        data = parsed.model_dump()
        data["bibkey"] = bibkey
        results.append(data)

    if errors:
        logger.warning(
            "Skipped %d entries due to errors (out of %d)",
            errors,
            errors + len(results),
        )

    res_df = pd.DataFrame(results).rename(
        columns={c: f"google_{c}" for c in cfg.result_columns}
    )
    return df.merge(res_df, on="bibkey", how="left")
