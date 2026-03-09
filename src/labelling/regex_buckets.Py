import logging

import re

from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

SIGNALS = {
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
        r"\bbenchmark(?:s|ing)?\b",
        r"\bleaderboard\b",
        r"\bstate[\-\s]of[\-\s]the[\-\s]art\b",
        r"\bSOTA\b",
        r"\bbaseline(?:s)?\b",
    ],
    # === AXE CLASSIFICATION : le papier traite d'une tâche de classification ===
    # Terme direct
    "classif_explicit": [
        r"\bclassif(?:y|ication|ier|iers)\b",
        r"\btext\s+categori[sz]ation\b",
        r"\blabel\s+prediction\b",
    ],
    # Tâches qui SONT de la classification (même sans le mot)
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
        # Détection (binaire/multiclass)
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
        # Paraphrase / similarité (classification binaire)
        r"\bparaphrase\s+(?:detection|identification)\b",
        r"\bsemantic\s+(?:similarity|equivalence)\b",
        r"\bduplicate\s+(?:detection|question)\b",
        # Langue / domaine
        r"\blanguage\s+identification\b",
        r"\bgenre\s+classification\b",
        r"\bdocument\s+classif",
        # Fact checking
        r"\bfact[\-\s]?check(?:ing)?\b",
        r"\bclaim\s+(?:verification|detection)\b",
        r"\bnatural\s+language\s+understanding\b",
        r"\bNLU\b",
    ],
    # Benchmarks connus qui SONT de la classification
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


def compute_scores(title: str, abstract: str) -> dict:
    text = f"{title} {abstract}"
    scores = {}
    for cat, patterns in SIGNALS.items():
        scores[cat] = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
    # Scores agrégés
    scores["benchmark_total"] = (
        scores["introduces"] + scores["evaluates_on"] + scores["benchmark_vocab"]
    )
    scores["classif_total"] = (
        scores["classif_explicit"]
        + scores["classif_tasks"]
        + scores["known_classif_benchmarks"]
    )
    return scores


def assign_bucket(row) -> str:
    bench = row["benchmark_total"]
    classif = row["classif_total"]
    known = row["known_classif_benchmarks"]

    # A: les deux signaux présents, ou benchmark connu de classification
    if known >= 1:
        return "A"
    if bench >= 1 and classif >= 1:
        return "A"

    # B: un seul axe — le LLM doit trancher
    if classif >= 1:  # parle de classif mais pas de benchmark
        return "B"
    if bench >= 2:  # parle beaucoup de benchmark, peut-être classif implicite
        return "B"

    # C: rien
    return "C"


if __name__ == "__main__":
    pass
