import pandas as pd
import seaborn as sns
import umap
from sentence_transformers import SentenceTransformer


def compute_embeddings(
    texts: list[str], model_name: str = "sentence-transformers/all-mpnet-base-v2"
) -> list[list[float]]:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()


def compute_embeddings_instruct(
    texts: list[str],
    model_name: str = "intfloat/multilingual-e5-large-instruct",
    device: str = "cpu",
) -> list[list[float]]:
    task_desc = (
        "Identify the type of NLP classification task by the prediction mechanism, "
        "ignoring the domain or genre of the input text"
    )
    texts = [f"Instruct: {task_desc}\nQuery: {p}" for p in texts]
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def umap_projection(
    embeddings: list[list[float]],
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> list[list[float]]:
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    projected = reducer.fit_transform(embeddings)
    return projected.tolist()


def plot_umap(points, labels=None):
    df = pd.DataFrame(points, columns=["x", "y"])
    if labels is not None:
        df["label"] = labels
        sns.scatterplot(data=df, x="x", y="y", hue="label")
    else:
        sns.scatterplot(data=df, x="x", y="y")
