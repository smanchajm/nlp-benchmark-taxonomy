import asyncio
import json
import logging
import os
import tempfile

import instructor
import pandas as pd
from tqdm.auto import tqdm

from src.taxonomy.schemas import DEFAULT_TASK, Label, TASKS

logger = logging.getLogger(__name__)

MODEL_MAP: dict[str, str] = {
    "mistral": "mistral-large-latest",
    "claude": "claude-3-haiku-20240307",
    "gemini": "gemini-1.5-flash-latest",
    "deepseek": "deepseek-chat",
    "openrouter": "google/gemma-4-31b-it",
}


def create_client(provider: str, *, async_: bool = False):
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
            OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
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
            OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
        )
    raise ValueError(f"Unknown provider: {provider!r}")



async def _classify_paper_async(client, provider: str, title: str, abstract: str, task: str = DEFAULT_TASK):
    cfg = TASKS[task]
    try:
        return await client.chat.completions.create(
            model=MODEL_MAP[provider],
            response_model=cfg.response_model,
            messages=[
                {"role": "system", "content": cfg.system_prompt},
                {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"},
            ],
            max_retries=3,
        )
    except Exception as e:
        logger.error("Classification failed for '%s': %s", title, e)
        return cfg.response_model(label=Label.UNSURE, justification="API Error")


def _save_checkpoint_indexed(df: pd.DataFrame, results: list[tuple[int, dict]], path: str, prefix: str = "llm") -> None:
    sorted_results = sorted(results, key=lambda x: x[0])
    res_df = pd.DataFrame([r for _, r in sorted_results]).rename(columns=lambda c: f"{prefix}_{c}")
    idxs = [i for i, _ in sorted_results]
    out = pd.concat([df.iloc[idxs].reset_index(drop=True), res_df], axis=1)
    out.to_parquet(path, index=False)
    logger.info("Checkpoint saved: %d/%d papers → %s", len(results), len(df), path)


async def label_papers_async(
    df: pd.DataFrame,
    provider: str = "deepseek",
    task: str = DEFAULT_TASK,
    max_concurrent: int = 20,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    """Classify papers concurrently with a semaphore-based pool."""
    client = create_client(provider, async_=True)
    sem = asyncio.Semaphore(max_concurrent)
    results: list[tuple[int, dict]] = []
    pbar = tqdm(total=len(df), desc=f"{provider}/{task}")

    async def _process(idx: int, title: str, abstract: str):
        async with sem:
            res = await _classify_paper_async(client, provider, title, abstract, task=task)
            results.append((idx, res.model_dump()))
            pbar.update(1)
            if checkpoint_path and len(results) % checkpoint_every == 0:
                _save_checkpoint_indexed(df, results, checkpoint_path, prefix=provider)

    await asyncio.gather(*[_process(i, r.title, str(r.abstract)) for i, r in enumerate(df.itertuples())])
    pbar.close()

    results.sort(key=lambda x: x[0])
    res_df = pd.DataFrame([r for _, r in results]).rename(columns=lambda c: f"{provider}_{c}")
    if checkpoint_path:
        _save_checkpoint_indexed(df, results, checkpoint_path, prefix=provider)
    return pd.concat([df.reset_index(drop=True), res_df], axis=1)



def mistral_batch_submit(df: pd.DataFrame, model: str = "mistral-large-latest", task: str = DEFAULT_TASK) -> str:
    """Submit a Mistral Batch API job via file upload."""
    from mistralai import Mistral

    cfg = TASKS[task]
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    schema = {
        "type": "json_schema",
        "json_schema": {"name": "PaperClass", "schema": cfg.response_model.model_json_schema()},
    }
    lines = [
        json.dumps({
            "custom_id": str(row["bibkey"]),
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": cfg.system_prompt},
                    {"role": "user", "content": f"Title: {row['title']}\nAbstract: {row['abstract']}"},
                ],
                "response_format": schema,
                "temperature": 0.0,
            },
        }, ensure_ascii=False)
        for _, row in df.iterrows()
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines))
        tmp_path = f.name
    try:
        with open(tmp_path, "rb") as fh:
            batch_data = client.files.upload(file={"file_name": "batch.jsonl", "content": fh}, purpose="batch")
    finally:
        os.unlink(tmp_path)
    job = client.batch.jobs.create(input_files=[batch_data.id], model=model, endpoint="/v1/chat/completions")
    logger.info("Mistral batch submitted: %s (file: %s)", job.id, batch_data.id)
    return job.id


def mistral_batch_results(job_id: str, df: pd.DataFrame, task: str = DEFAULT_TASK) -> pd.DataFrame:
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
    res_df = pd.DataFrame(results).rename(columns={c: f"mistral_{c}" for c in cfg.result_columns})
    return df.merge(res_df, on="bibkey", how="left")



def claude_batch_submit(df: pd.DataFrame, model: str = "claude-haiku-4-5-20251001", task: str = DEFAULT_TASK) -> str:
    """Submit an Anthropic Message Batches job."""
    from anthropic import Anthropic

    cfg = TASKS[task]
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    tool_schema = {**cfg.tool_schema, "cache_control": {"type": "ephemeral"}}
    requests = [
        {
            "custom_id": str(row["bibkey"]),
            "params": {
                "model": model,
                "max_tokens": cfg.max_tokens,
                "system": [{"type": "text", "text": cfg.system_prompt, "cache_control": {"type": "ephemeral"}}],
                "tools": [tool_schema],
                "tool_choice": {"type": "tool", "name": cfg.tool_name},
                "messages": [{"role": "user", "content": f"Title: {row['title']}\nAbstract: {row['abstract']}"}],
            },
        }
        for _, row in df.iterrows()
    ]
    batch = client.messages.batches.create(requests=requests)
    logger.info("Claude batch submitted: %s", batch.id)
    return batch.id


def claude_batch_results(batch_id: str, df: pd.DataFrame, task: str = DEFAULT_TASK) -> pd.DataFrame:
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
        try:
            parsed = cfg.response_model.model_validate(entry.result.message.content[0].input)
        except Exception as e:
            logger.warning("Validation error for %s: %s", entry.custom_id, e)
            errors += 1
            continue
        data = parsed.model_dump()
        data["bibkey"] = entry.custom_id
        results.append(data)
    if errors:
        logger.warning("Skipped %d entries due to errors (out of %d)", errors, errors + len(results))
    res_df = pd.DataFrame(results).rename(columns={c: f"claude_{c}" for c in cfg.result_columns})
    return df.merge(res_df, on="bibkey", how="left")



def _flatten_google_schema(schema: dict) -> dict:
    """Convert a Pydantic JSON Schema to a Google-compatible schema.

    Resolves $defs/$ref, replaces anyOf nullable patterns with nullable flag,
    and strips unsupported keys (title, description, default, $defs).
    """
    defs = schema.get("$defs", {})

    def _resolve(node: dict) -> dict:
        if "$ref" in node:
            return _resolve(defs[node["$ref"].rsplit("/", 1)[-1]])
        out = {}
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


def google_batch_submit(df: pd.DataFrame, model: str = "gemini-3-flash-preview", task: str = DEFAULT_TASK) -> str:
    """Submit a Google GenAI Batch job via file upload."""
    from google import genai
    from google.genai import types

    cfg = TASKS[task]
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    schema = _flatten_google_schema(cfg.response_model.model_json_schema())
    lines = [
        json.dumps({
            "key": str(row["bibkey"]),
            "request": {
                "contents": [{"parts": [{"text": f"Title: {row['title']}\nAbstract: {row['abstract']}"}]}],
                "systemInstruction": {"parts": [{"text": cfg.system_prompt}]},
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": schema,
                    "temperature": 0.0,
                },
            },
        }, ensure_ascii=False)
        for _, row in df.iterrows()
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines))
        tmp_path = f.name
    try:
        uploaded = client.files.upload(
            file=tmp_path,
            config=types.UploadFileConfig(display_name=f"batch-{task}", mime_type="jsonl"),
        )
    finally:
        os.unlink(tmp_path)
    job = client.batches.create(
        model=model,
        src=uploaded.name,
        config=types.CreateBatchJobConfig(display_name=f"nlp-taxonomy-{task}"),
    )
    logger.info("Google batch submitted: %s (file: %s)", job.name, uploaded.name)
    return job.name


def google_batch_results(job_name: str, df: pd.DataFrame, task: str = DEFAULT_TASK) -> pd.DataFrame:
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
        logger.warning("Skipped %d entries due to errors (out of %d)", errors, errors + len(results))
    res_df = pd.DataFrame(results).rename(columns={c: f"google_{c}" for c in cfg.result_columns})
    return df.merge(res_df, on="bibkey", how="left")
