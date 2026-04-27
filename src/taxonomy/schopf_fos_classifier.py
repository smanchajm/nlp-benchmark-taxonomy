from transformers import pipeline
import pandas as pd


def classify_fos(texts: pd.DataFrame) -> list[str]:
    classifier = pipeline(
        "text-classification",
        model="TimSchopf/nlp_taxonomy_classifier",
        device=0,  # Use GPU if available
    )
    tokenizer = classifier.tokenizer

    texts = [
        text["title"] + tokenizer.sep_token + (text.get("abstract") or "")
        for text in texts.to_dict(orient="records")
    ]
    results = classifier(texts, truncation=True)
    return [res["label"] for res in results]
