import asyncio
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
    justification: str = Field(description="Max 15 words", max_length=100)
    confidence: float = Field(description="0.0 to 1.0", ge=0.0, le=1.0)


SYSTEM_PROMPT = """You classify NLP papers. Determine if the paper INTRODUCES a benchmark/dataset for TEXT CLASSIFICATION.

POSITIVE: introduces a benchmark for sentiment analysis, NLI, topic classification, intent detection, hate speech detection, stance detection, emotion classification, fact checking, paraphrase detection, language identification. Also POSITIVE if it introduces an NLU benchmark where the majority of tasks are classification.

NEGATIVE: method paper using existing benchmarks, benchmark for non-classification tasks (QA, generation, NER, parsing, MT), survey, dataset without classification task.

UNSURE: multi-task NLU benchmark with a significant mix of classification and non-classification tasks.

Keep justification under 15 words."""


async def _classify_paper(
    client,
    provider: str,
    title: str,
    abstract: str,
    sem: asyncio.Semaphore,
) -> PaperClassification:
    """Classify a single paper with semaphore-based rate limiting."""
    async with sem:
        try:
            return await client.chat.completions.create(
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
                label=Label.UNSURE, justification="API Error", confidence=0.0
            )


def _create_client(provider: str):
    """Create an instructor-wrapped client for the given provider.

    Provider-specific SDKs are imported lazily so users only need the SDK
    for the provider they actually use.
    """
    if provider == "mistral":
        from mistralai import Mistral

        return instructor.from_mistral(
            Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        )
    if provider == "claude":
        from anthropic import AsyncAnthropic

        return instructor.from_anthropic(
            AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
    if provider == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return instructor.from_gemini(
            genai.GenerativeModel("gemini-1.5-flash-latest")
        )
    raise ValueError(f"Unknown provider: {provider!r}")


async def label_papers(
    df: pd.DataFrame, provider: str = "mistral", concurrency: int = 10
) -> pd.DataFrame:
    """Classify all papers in *df* concurrently and return the augmented DataFrame."""
    client = _create_client(provider)
    sem = asyncio.Semaphore(concurrency)

    tasks = [
        _classify_paper(client, provider, r.title, str(r.abstract), sem)
        for r in df.itertuples()
    ]

    logger.info("Classifying %d papers via %s...", len(tasks), provider)
    results = await asyncio.gather(*tasks)

    res_df = pd.DataFrame([r.model_dump() for r in results]).rename(
        columns=lambda c: f"llm_{c}"
    )
    return pd.concat([df.reset_index(drop=True), res_df], axis=1)


def mistral_batch_submit(
    df: pd.DataFrame, model: str = "mistral-small-latest"
) -> str:
    """Submit a Mistral Batch API job with auto-generated JSON schema."""
    from mistralai import Mistral
    from mistralai.models import BatchRequestInput

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
        BatchRequestInput(
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
        content = entry["response"]["choices"][0]["message"]["content"]
        parsed = PaperClassification.model_validate_json(content)
        data = parsed.model_dump()
        data["bibkey"] = entry["custom_id"]
        results.append(data)

    res_df = pd.DataFrame(results).rename(
        columns={c: f"llm_{c}" for c in ["label", "justification", "confidence"]}
    )
    return df.merge(res_df, on="bibkey", how="left")
