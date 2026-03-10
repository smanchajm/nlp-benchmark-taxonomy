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
        r"\bwe\s+(?:introduce|present|propose|release|create|construct|compile|curate|build)\b.*\b(?:benchmark|dataset|corpus|evaluation\s+suite|test\s+suite)\b",
        r"\bnew\s+(?:benchmark|dataset|evaluation|corpus)\b",
        r"\bnovel\s+(?:benchmark|dataset)\b",
        r"\bcontribut(?:e|ion)\b.*\b(?:benchmark|dataset)\b",
    ],
    "evaluates_on": [
        r"\b(?:evaluate|test|assess|validate|experiment)(?:s|ed|ing)?\s+(?:on|across|over)\b",
        r"\bresults?\s+on\b",
        r"\bperformanc(?:e|es)\s+on\b",
        r"\bfine[\-\s]?tun(?:e|ed|ing)\s+on\b",
    ],
    "benchmark_vocab": [
        r"(?<!existing )(?<!a existing )(?<!standard )(?<!a standard )(?<!current )(?<!a current )(?<!popular )(?<!a popular )(?<!the standard )\bbenchmark(?:s|ing)?\b",
        r"\bleaderboard\b",
        r"\bstate[\-\s]of[\-\s]the[\-\s]art\b",
        r"\bSOTA\b",
        r"\bbaseline(?:s)?\b",
    ],
    "classif_explicit": [
        r"\btext\s+classif(?:ication|ier|iers)\b",
        r"\b(?:sentence|document|review|news|tweet|comment|message|post)\s+classif(?:ication|ier)\b",
        r"\bclassif(?:ication|y)\s+(?:task|benchmark|dataset|problem)\b",
        r"\btext\s+categori[sz]ation\b",
        r"\blabel\s+prediction\b",
    ],
    "classif_tasks": [
        # Sentiment / opinion
        r"\bsentiment\s+(?:analysis|classification|detection|prediction)\b",
        r"\bopinion\s+(?:mining|classification|detection)\b",
        r"\baspect[\-\s](?:based|level)\b.*\bsentiment\b",
        r"\bpolarity\b",
        # NLI / entailment
        r"\bnatural\s+language\s+inference\b",
        r"\btextual\s+entailment\b",
        r"\bNLI\b",
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
        # Paraphrase / similarity (binary classification)
        r"\bparaphrase\s+(?:detection|identification)\b",
        r"\bsemantic\s+(?:similarity|equivalence)\b",
        r"\bduplicate\s+(?:detection|question)\b",
        # Language / domain
        r"\blanguage\s+identification\b",
        r"\bgenre\s+classification\b",
        r"\bdocument\s+classif",
        # Fact checking
        r"\bfact[\-\s]?check(?:ing)?\b",
        r"\bclaim\s+(?:verification|detection)\b",
    ],
    "classif_weak": [
        r"\bnatural\s+language\s+understanding\b",
        r"\bNLU\b",
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
        r"\bDBpedia\b",
        r"\b20[\s]?News(?:groups)?\b",
        r"\bYahoo[\s!]?\s*Answers?\b",
        r"\bYelp\b.*\b(?:review|polarity)\b",
        r"\bAmazon\b.*\breview\b",
        r"\bSemEval\b",
        r"\bTREC\b",
        r"\bSogou\b",
        r"\bHatEval\b",
        r"\bOffComBR\b",
        r"\bOLID\b",
    ],
}


def compute_scores(title: str, abstract: str) -> dict[str, int]:
    """Compute regex-based signal scores from a paper's title and abstract."""
    text = f"{title} {abstract}"
    scores: dict[str, int] = {}
    for cat, patterns in SIGNALS.items():
        scores[cat] = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))

    scores["benchmark_total"] = (
        scores["introduces"] + scores["evaluates_on"] + scores["benchmark_vocab"]
    )

    strong_classif = (
        scores["classif_explicit"]
        + scores["classif_tasks"]
        + scores["known_classif_benchmarks"]
    )
    scores["classif_total"] = strong_classif + (
        scores["classif_weak"] if strong_classif >= 1 else 0
    )
    return scores


def assign_bucket(row: dict[str, int]) -> str:
    """Assign a paper to bucket A, B, or C based on precomputed scores."""
    bench = row["benchmark_total"]
    classif = row["classif_total"]
    known = row["known_classif_benchmarks"]

    # A: both signals present, or a known classification benchmark
    if known >= 1:
        return "A"
    if bench >= 1 and classif >= 1:
        return "A"

    # B: one axis only — needs LLM adjudication
    if classif >= 1:
        return "B"
    if row["introduces"] >= 1:
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
