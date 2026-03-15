import enum
import json
import logging
import os

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


class PaperClassification(BaseModel):
    label: Label
    justification: str = Field(description="Max 20 words", max_length=150)
    task_family: str | None = Field(
        description='Short task label (e.g. "sentiment", "NLI", "NER"). null if no new dataset.',
        default=None,
    )
    is_multitask: bool | None = Field(
        description="True if multi-task benchmark (e.g. GLUE). null for NEGATIVE.",
        default=None,
    )


SYSTEM_PROMPT = """You classify NLP papers. Determine if the paper INTRODUCES a benchmark/dataset for TEXT CLASSIFICATION.

POSITIVE: Introduces a benchmark/dataset where a model predicts a discrete label from a fixed set for each text input (or text pair), e.g., sentiment analysis, NLI, topic classification, fact verification, relation classification. Multi-task benchmarks are POSITIVE if ≥75% of sub-tasks are classification.

NEGATIVE if either:
- The paper does NOT introduce a new dataset (method/model papers, surveys, shared tasks without new data).
- The paper introduces a dataset for a task whose output is not a single discrete label per input: spans, token-level labels, rankings, structured outputs (e.g., NER, parsing, extractive QA, generation, translation, summarization, dialogue, multiple-choice QA, multimodal benchmarks requiring non-textual inputs).

UNSURE: The paper describes collecting or annotating data but does not clearly frame it as a reusable benchmark, OR it is a multi-task benchmark near the 75% classification boundary, OR the task definition is ambiguous between classification and a non-classification formulation.

task_family: short label for the task type (e.g., "sentiment", "NLI", "NER", "summarization", "MCQA"). Required when the paper introduces a dataset. Set null when NEGATIVE because no new dataset.

is_multitask: true if the benchmark bundles several sub-tasks into one evaluation suite (e.g. GLUE, SuperGLUE). false otherwise. Set null for NEGATIVE.

Justification: name the task type and explain why it is or is not classification. If NEGATIVE because no new dataset, state that. Max 20 words."""


def _classify_paper(
    client,
    provider: str,
    title: str,
    abstract: str,
) -> PaperClassification:
    """Classify a single paper."""
    try:
        return client.chat.completions.create(
            model=MODEL_MAP[provider],
            response_model=PaperClassification,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Title: {title}\nAbstract: {abstract}",
                },
            ],
            max_retries=3,
        )
    except Exception as e:
        logger.error("Classification failed for '%s': %s", title, e)
        return PaperClassification(
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
) -> pd.DataFrame:
    """Classify all papers in *df* sequentially and return the augmented DataFrame."""
    client = _create_client(provider)

    logger.info("Classifying %d papers via %s...", len(df), provider)
    results = [
        _classify_paper(client, provider, r.title, str(r.abstract))
        for r in df.itertuples()
    ]

    res_df = pd.DataFrame([r.model_dump() for r in results]).rename(
        columns=lambda c: f"llm_{c}"
    )
    return pd.concat([df.reset_index(drop=True), res_df], axis=1)


def mistral_batch_submit(df: pd.DataFrame, model: str = "mistral-small-latest") -> str:
    """Submit a Mistral Batch API job with auto-generated JSON schema."""
    from mistralai import Mistral
    from mistralai.models import BatchRequest

    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "PaperClass",
            "schema": PaperClassification.model_json_schema(),
            "strict": True,
        },
    }

    requests = [
        BatchRequest(
            custom_id=str(row["bibkey"]),
            body={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
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


def mistral_batch_results(job_id: str, df: pd.DataFrame) -> pd.DataFrame:
    """Fetch and merge results from a completed Mistral Batch job."""
    from mistralai import Mistral

    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    job = client.batch.jobs.get(job_id=job_id)
    if job.status != "SUCCESS":
        raise RuntimeError(f"Batch job not finished (status: {job.status})")

    output_file = client.files.download(file_id=job.output_file)
    results = []
    for line in output_file.read().decode("utf-8").strip().split("\n"):
        entry = json.loads(line)
        content = entry["response"]["body"]["choices"][0]["message"]["content"]
        parsed = PaperClassification.model_validate_json(content)
        data = parsed.model_dump()
        data["bibkey"] = entry["custom_id"]
        results.append(data)

    res_df = pd.DataFrame(results).rename(
        columns={
            c: f"llm_{c}"
            for c in [
                "label",
                "justification",
                "task_family",
                "is_multitask",
            ]
        }
    )
    return df.merge(res_df, on="bibkey", how="left")
