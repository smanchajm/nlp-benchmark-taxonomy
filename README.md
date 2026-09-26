# nlp-benchmark-taxonomy

Mapping and taxonomizing NLP classification benchmarks from the ACL Anthology. Research project conducted at RALI, DIRO, Université de Montréal.

## Pipeline

Identifying benchmark papers is a **two-stage filter**: a fine-tuned SciBERT classifier
narrows the full Anthology, then an LLM vote refines its positives.

```
                    data/raw/anthology_enriched.parquet
                                   │
   ┌───────────────────────────────┴──────────── stage 0: training data ─────┐
   │  regex bucketing (src/corpus/regex_buckets.py, notebooks/1)            │
   │      → anthology_enriched_with_bucket.parquet, sampled splits          │
   │  LLM vote on splits (Claude / Mistral) → data/classifier/ready/        │
   └───────────────────────────────┬───────────────────────────────────────┘
                                   │
   ┌───────────────────────────────┴──────────── stage 1: SciBERT ──────────┐
   │  src/classifier/{train,infer}.py + configs/ + slurm/ (cluster jobs)    │
   │      → data/classifier/predictions/inference.parquet (3,738 positives) │
   └───────────────────────────────┬───────────────────────────────────────┘
                                   │
   ┌───────────────────────────────┴──────────── stage 2: LLM extraction ───┐
   │  notebooks/2 — taxonomy prompt on SciBERT positives, multi-provider    │
   │      → data/taxonomy/per_llm/*.parquet → merged.parquet                │
   └───────────────────────────────┬───────────────────────────────────────┘
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        │  taxonomy construction                              │
        │    embeddings + HAC/UMAP (src/taxonomy, notebooks/3)│
        │    manual tree (tools/taxonomy_builder)             │
        │      → data/taxonomy/manual_taxonomy_tree.json      │
        │    Mistral leaf assignment (notebooks/4)            │
        │  resource enrichment                                │
        │    PDF/GROBID link extraction (src/extraction, nb 5)│
        │    hub coverage (src/coverage, notebooks/6)         │
        └──────────────────────────┬──────────────────────────┘
                                   │
                 src/corpus/build_single_task_benchmark_paper.py
                   → data/corpus/single_task_benchmark_paper.parquet
                                   │
   ┌───────────────────────────────┴──────────── evaluation ────────────────┐
   │  src/evaluation/ — Label Studio configs, task export, gold sampling    │
   │  annotation_guide.typ — annotator guide (French)                       │
   │  notebooks/8 — accuracy / F1 against the gold set                      │
   └────────────────────────────────────────────────────────────────────────┘
```

`data/` holds every runtime output and is not versioned (3.7 GB); see `src/paths.py`.

## Layout

| Path | Role |
| --- | --- |
| `src/corpus/` | Anthology fetch, clean, enrich, regex buckets, final corpus build |
| `src/classifier/` | SciBERT stage-1 filter — train, infer, model, plus `configs/` (YAML) and `slurm/` (cluster jobs) |
| `src/taxonomy/` | Embeddings, HAC utilities, LLM providers and structured schemas |
| `src/extraction/` | PDF / GROBID-TEI dataset and code link extraction |
| `src/coverage/` | Hub matching (PwC, HuggingFace) and arXiv id resolution |
| `src/evaluation/` | Label Studio annotation pipeline and gold-set sampling |
| `notebooks/` | Stage orchestration, numbered 1–8 in execution order |
| `tools/taxonomy_builder/` | Offline single-page app to build the taxonomy by hand |
| `scripts/` | Local cluster sync helpers (untracked — they carry login details) |

## Setup

This project uses **uv**; Python 3.13+ is required.

```bash
uv sync
uv run python src/corpus/fetch_anthology.py
```
