import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from logging_config import setup_logging

logger = logging.getLogger(__name__)

SIGNALS: dict[str, list[str]] = {
    "introduces": [
        # --- span-limited: max 120 chars between verb and object ---
        r"\bwe\s+(?:introduce|present|propose|release|create|construct|compile|curate|build)\b.{0,120}\b(?:benchmark|dataset|corpus|evaluation\s+suite|test\s+suite)\b",
        r"\bnew\s+(?:benchmark|dataset|evaluation|corpus)\b",
        r"\bnovel\s+(?:benchmark|dataset)\b",
        r"\bcontribut(?:e|ion)\b.{0,80}\b(?:benchmark|dataset)\b",
    ],
    "evaluates_on": [
        r"\b(?:evaluate|test|assess|validate|experiment)(?:s|ed|ing)?\s+(?:on|across|over)\b",
        # --- removed "results on" / "performance on": fires on any empirical paper ---
        r"\bfine[\-\s]?tun(?:e|ed|ing)\s+on\b",
    ],
    "benchmark_vocab": [
        r"(?<!existing )(?<!a existing )(?<!standard )(?<!a standard )(?<!current )(?<!a current )(?<!popular )(?<!a popular )(?<!the standard )\bbenchmark(?:s|ing)?\b",
        r"\bleaderboard\b",
        # --- removed "state-of-the-art", "SOTA", "baseline(s)": too noisy ---
    ],
    # --- weak benchmark signals: only count when strong signals are present ---
    "benchmark_weak": [
        r"\bstate[\-\s]of[\-\s]the[\-\s]art\b",
        r"\bSOTA\b",
        r"\bbaseline(?:s)?\b",
        r"\bresults?\s+on\b",
        r"\bperformanc(?:e|es)\s+on\b",
    ],
    "classif_explicit": [
        r"\btext\s+classif(?:ication|ier|iers)\b",
        r"\b(?:sentence|document|review|news|tweet|comment|message|post)\s+classif(?:ication|ier)\b",
        r"\bclassif(?:ication|y)\s+(?:task|benchmark|dataset|problem)\b",
        r"\btext\s+categori[sz]ation\b",
        # --- removed "label prediction": matches NER / sequence labeling ---
    ],
    "classif_tasks": [
        # Sentiment / opinion
        r"\bsentiment\s+(?:analysis|classification|detection|prediction)\b",
        r"\bopinion\s+(?:mining|classification|detection)\b",
        r"\baspect[\-\s](?:based|level)\b.*\bsentiment\b",
        # --- tightened: require "polarity" with sentiment/opinion/classification context ---
        r"\b(?:sentiment|opinion)\s+polarity\b",
        r"\bpolarity\s+(?:classification|detection)\b",
        # NLI / entailment
        r"\bnatural\s+language\s+inference\b",
        r"\btextual\s+entailment\b",
        # --- NLI: exclude "native language identification" context ---
        r"(?<!native language )\bNLI\b",
        r"\brecogniz(?:e|ing)\s+textual\s+entailment\b",
        # Topic
        r"\btopic\s+(?:classification|detection|categorization|labeling)\b",
        # Intent / dialogue
        r"\bintent\s+(?:detection|classification|recognition)\b",
        r"\bdialogue?\s+act\b",
        # Detection (binary/multiclass)
        r"\bhate\s+speech\b",
        r"\btoxic(?:ity)?\s+(?:detection|classification)\b",
        r"\boffensive\s+language\b",
        r"\babusive\s+language\b",
        r"\bspam\s+(?:detection|classification|filtering)\b",
        r"\bfake\s+news\s+(?:detection|classification)\b",
        r"\bmisinformation\s+detection\b",
        r"\brumou?r\s+(?:detection|verification)\b",
        r"\bstance\s+(?:detection|classification)\b",
        r"\birony\s+detection\b",
        r"\bsarcasm\s+detection\b",
        r"\bemotion\s+(?:detection|classification|recognition)\b",
        # Paraphrase (keep — this IS classification)
        r"\bparaphrase\s+(?:detection|identification)\b",
        r"\bduplicate\s+(?:detection|question)\b",
        # --- semantic similarity moved to classif_weak ---
        # Language / domain
        # --- "language identification" restricted to exclude "native language identification" ---
        r"(?<!native )\blanguage\s+identification\b",
        r"\bgenre\s+classification\b",
        r"\bdocument\s+classif",
        # Fact checking
        r"\bfact[\-\s]?check(?:ing)?\b",
        r"\bclaim\s+(?:verification|detection)\b",
    ],
    "classif_weak": [
        r"\bnatural\s+language\s+understanding\b",
        r"\bNLU\b",
        # --- moved from classif_tasks: often regression/ranking, not classification ---
        r"\bsemantic\s+(?:similarity|equivalence)\b",
    ],
    "known_classif_benchmarks": [
        r"\bGLUE\b",
        r"\bSuperGLUE\b",
        r"\bSST[\-\s]?[25]\b",
        r"\bMNLI\b",
        r"\bSNLI\b",
        r"\bXNLI\b",
        r"\bANLI\b",
        r"\bRTE\b",
        r"\bMRPC\b",
        r"\bCoLA\b",
        r"\bQNLI\b",
        r"\bWNLI\b",
        r"\bPAWS\b",
        r"\bPAWS[\-\s]X\b",
        r"\bIMDb\b",
        r"\bAG[\s\']?s?\s*News\b",
        # --- removed DBpedia (entity linking noise) and TREC (venue, not a benchmark) ---
        r"\b20[\s]?News(?:groups)?\b",
        r"\bYahoo[\s!]?\s*Answers?\b",
        r"\bYelp\b.*\b(?:review|polarity)\b",
        r"\bAmazon\b.*\breview\b",
        # --- SemEval removed: it's a venue, not a benchmark ---
        # --- TREC removed (see above) ---
        r"\bSogou\b",
        r"\bHatEval\b",
        r"\bOffComBR\b",
        r"\bOLID\b",
    ],
}

# --- Post-filter: detect "we propose a METHOD" pattern ---
_METHOD_RE = re.compile(
    r"\bwe\s+(?:introduce|present|propose|release|create|construct|compile|curate|build)\b"
    r".{0,60}\b(?:method|model|approach|framework|system|technique|architecture|algorithm|pipeline|tool|mechanism|set\s+of\s+experiments)\b",
    re.IGNORECASE,
)


def compute_scores(title: str, abstract: str) -> dict[str, int]:
    """Compute regex-based signal scores from a paper's title and abstract."""
    text = f"{title} {abstract}"
    scores: dict[str, int] = {}
    for cat, patterns in SIGNALS.items():
        scores[cat] = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))

    scores["benchmark_total"] = (
        scores["introduces"] + scores["evaluates_on"] + scores["benchmark_vocab"]
    )
    # benchmark_weak only counts if there's already a strong benchmark signal
    if scores["benchmark_total"] >= 1:
        scores["benchmark_total"] += scores["benchmark_weak"]

    strong_classif = (
        scores["classif_explicit"]
        + scores["classif_tasks"]
        + scores["known_classif_benchmarks"]
    )
    scores["classif_total"] = strong_classif + (
        scores["classif_weak"] if strong_classif >= 1 else 0
    )

    # Flag papers that introduce a method/model (not a benchmark)
    scores["introduces_method"] = 1 if _METHOD_RE.search(text) else 0

    return scores


def assign_bucket(row: dict[str, int]) -> str:
    """Assign a paper to bucket A, B, or C based on precomputed scores.

    Bucket A requires a genuine *introduces benchmark/dataset* signal,
    not just "we propose a method ... evaluated on a benchmark".
    """
    introduces = row["introduces"]
    classif = row["classif_total"]
    known = row["known_classif_benchmarks"]
    introduces_method = row.get("introduces_method", 0)

    # Downgrade: if the paper introduces a method, not a benchmark
    if introduces >= 1 and introduces_method >= 1:
        introduces = 0  # treat as not introducing a benchmark

    # A: paper introduces a benchmark AND has a classification signal
    if introduces >= 1 and classif >= 1:
        return "A"
    if introduces >= 1 and known >= 1:
        return "A"

    # B: weaker / partial evidence — needs LLM adjudication
    if known >= 1:
        return "B"
    if row["introduces"] >= 1:  # original score, before downgrade
        return "B"
    if classif >= 1 and row["benchmark_vocab"] >= 1:
        return "B"

    # C: no relevant signal
    return "C"


def stratified_bucket_sample(
    df: "pd.DataFrame",
    bucket_col: str = "bucket",
    n_a: int = 400,
    n_b: int = 400,
    n_c: int = 300,
    seed: int = 42,
) -> "pd.DataFrame":
    """Sample papers stratified by bucket and year."""
    import pandas as pd

    samples = []
    for bucket, n in [("A", n_a), ("B", n_b), ("C", n_c)]:
        bucket_df = df[df[bucket_col] == bucket]
        n_actual = min(n, len(bucket_df))

        sample = (
            bucket_df.groupby("year", group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=max(1, round(n_actual * len(x) / len(bucket_df))),
                    random_state=seed,
                )
            )
            .head(n_actual)
        )
        samples.append(sample)

    result = pd.concat(samples)
    logger.info("Total sampled: %d", len(result))
    logger.info("Per bucket:\n%s", result[bucket_col].value_counts())
    return result


if __name__ == "__main__":
    setup_logging()
