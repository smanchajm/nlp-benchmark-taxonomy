from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import LabelEncoder

EXCLUDED_LABELS: frozenset[str] = frozenset({"other", "topic_subject"})


def build_linkage(embeddings: np.ndarray, method: str = "average") -> np.ndarray:
    if method == "ward":
        return linkage(embeddings, method="ward", metric="euclidean")
    dist = squareform(cosine_distances(embeddings), checks=False)
    return linkage(dist, method=method)


def make_color_map(
    coarse_values: pd.Series,
    palette: list[str] | None = None,
) -> dict[str, str]:
    if palette is None:
        palette = pc.qualitative.Plotly
    unique = sorted(coarse_values.dropna().unique())
    return {label: palette[i % len(palette)] for i, label in enumerate(unique)}


def plot_dendrogram(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    coarse_col: str,
    label_col: str,
    title: str,
    color_map: dict[str, str] | None = None,
    color_threshold: float = 0.4,
) -> ff.Figure:
    coarse_values = df[coarse_col].fillna("Unknown")
    if color_map is None:
        color_map = make_color_map(coarse_values)

    leaf_labels = [
        f"{i:02d} | {bk[:30]} | [{tt[:35]}]"
        for i, (bk, tt) in enumerate(zip(df["bibkey"], df[label_col].fillna("?")))
    ]
    label_to_coarse = dict(zip(leaf_labels, coarse_values))

    fig = ff.create_dendrogram(
        embeddings,
        orientation="right",
        labels=leaf_labels,
        distfun=lambda x: squareform(cosine_distances(x), checks=False),
        linkagefun=lambda d: linkage(d, method="average"),
        color_threshold=color_threshold,
    )

    ticktext = (
        list(fig.layout.yaxis.ticktext) if fig.layout.yaxis.ticktext is not None else []
    )
    tickvals = (
        list(fig.layout.yaxis.tickvals) if fig.layout.yaxis.tickvals is not None else []
    )

    leaf_annotations = [
        dict(
            x=1.002,
            y=val,
            xref="paper",
            yref="y",
            text=text,
            showarrow=False,
            font=dict(
                color=color_map.get(label_to_coarse.get(text, "Unknown"), "black"),
                size=10,
                family="monospace",
            ),
            xanchor="left",
            align="left",
        )
        for text, val in zip(ticktext, tickvals)
    ]
    legend_annotations = [
        dict(
            x=0.01,
            y=1.0 - i * 0.04,
            xref="paper",
            yref="paper",
            text=f"■ {label}",
            showarrow=False,
            font=dict(color=color, size=11),
            xanchor="left",
        )
        for i, (label, color) in enumerate(color_map.items())
    ]

    fig.update_layout(
        title=title,
        height=max(800, 22 * len(leaf_labels)),
        font=dict(size=11, family="monospace"),
        plot_bgcolor="white",
        autosize=True,
        annotations=leaf_annotations + legend_annotations,
        xaxis=dict(
            title="Cosine Distance",
            showgrid=True,
            gridcolor="lightgray",
            side="bottom",
            fixedrange=False,
            rangeslider=dict(visible=True, thickness=0.05),
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            side="right",
            automargin=True,
            fixedrange=False,
        ),
        margin=dict(l=20, r=450, t=80, b=20),
    )
    return fig


def plot_dendrogram_adaptive(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    root: TaxoNode,
    coarse_col: str,
    label_col: str,
    title: str,
    color_threshold: float = 0.4,
) -> ff.Figure:
    """
    Dendrogram where leaves are coloured by their adaptive cluster (from cut_tree_adaptive).

    Each cluster gets a distinct colour; the legend shows the majority coarse label
    and size of each cluster. Papers not assigned to any leaf (outliers skipped by
    the degenerate-split rule) are shown in grey.
    """
    n = len(df)
    flat = root.flat_labels(n=n)  # -1 for unassigned outliers

    # Build cluster display names: "C03 · NER (n=25)"
    leaf_nodes = root._leaf_nodes()
    cluster_names: dict[int, str] = {}
    for cid, node in enumerate(leaf_nodes):
        top = df.iloc[node.leaves][coarse_col].value_counts()
        label = top.index[0] if len(top) else "?"
        cluster_names[cid] = f"C{cid:02d} · {label} (n={len(node.leaves)})"

    coarse_values = pd.Series(
        [cluster_names.get(flat[i], "outlier") for i in range(n)],
        index=df.index,
    )

    palette = pc.qualitative.Alphabet + pc.qualitative.Dark24
    unique_clusters = sorted(cluster_names.values()) + ["outlier"]
    color_map = {
        lbl: (palette[i % len(palette)] if lbl != "outlier" else "#cccccc")
        for i, lbl in enumerate(unique_clusters)
    }

    leaf_labels = [
        f"{i:02d} | {bk[:30]} | [{tt[:35]}]"
        for i, (bk, tt) in enumerate(zip(df["bibkey"], df[label_col].fillna("?")))
    ]
    label_to_coarse = dict(zip(leaf_labels, coarse_values))

    fig = ff.create_dendrogram(
        embeddings,
        orientation="right",
        labels=leaf_labels,
        distfun=lambda x: squareform(cosine_distances(x), checks=False),
        linkagefun=lambda d: linkage(d, method="average"),
        color_threshold=color_threshold,
    )

    ticktext = (
        list(fig.layout.yaxis.ticktext) if fig.layout.yaxis.ticktext is not None else []
    )
    tickvals = (
        list(fig.layout.yaxis.tickvals) if fig.layout.yaxis.tickvals is not None else []
    )

    leaf_annotations = [
        dict(
            x=1.002,
            y=val,
            xref="paper",
            yref="y",
            text=text,
            showarrow=False,
            font=dict(
                color=color_map.get(label_to_coarse.get(text, "outlier"), "#cccccc"),
                size=10,
                family="monospace",
            ),
            xanchor="left",
            align="left",
        )
        for text, val in zip(ticktext, tickvals)
    ]
    legend_annotations = [
        dict(
            x=0.01,
            y=1.0 - i * 0.03,
            xref="paper",
            yref="paper",
            text=f"■ {lbl}",
            showarrow=False,
            font=dict(color=color_map[lbl], size=10),
            xanchor="left",
        )
        for i, lbl in enumerate(sorted(cluster_names.values()))
    ]

    fig.update_layout(
        title=title,
        height=max(800, 22 * len(leaf_labels)),
        font=dict(size=11, family="monospace"),
        plot_bgcolor="white",
        autosize=True,
        annotations=leaf_annotations + legend_annotations,
        xaxis=dict(
            title="Cosine Distance",
            showgrid=True,
            gridcolor="lightgray",
            side="bottom",
            fixedrange=False,
            rangeslider=dict(visible=True, thickness=0.05),
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            side="right",
            automargin=True,
            fixedrange=False,
        ),
        margin=dict(l=20, r=500, t=80, b=20),
    )
    return fig


def clustering_metrics(
    embeddings: np.ndarray,
    Z: np.ndarray,
    coarse_values: pd.Series,
    n_clusters: int = 15,
) -> dict:
    hac_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    le = LabelEncoder()
    coarse_encoded = le.fit_transform(coarse_values)

    ari = adjusted_rand_score(coarse_encoded, hac_labels)
    nmi = normalized_mutual_info_score(coarse_encoded, hac_labels)
    sil = silhouette_score(embeddings, hac_labels, metric="cosine")

    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, np.nan)
    intra, inter = [], []
    n = len(hac_labels)
    for i in range(n):
        for j in range(i + 1, n):
            s = sim_matrix[i, j]
            (intra if hac_labels[i] == hac_labels[j] else inter).append(s)

    intra_mean = float(np.mean(intra))
    inter_mean = float(np.mean(inter))
    return {
        "n_clusters": len(set(hac_labels)),
        "n_clusters_requested": n_clusters,
        "ari": ari,
        "nmi": nmi,
        "silhouette": sil,
        "cosine_intra": intra_mean,
        "cosine_inter": inter_mean,
        "cosine_ratio": intra_mean / inter_mean,
        "hac_labels": hac_labels,
    }


def dendrogram_purity(
    Z: np.ndarray,
    labels: np.ndarray,
    excluded: frozenset[str] = EXCLUDED_LABELS,
) -> float:
    n = len(labels)
    node_leaves: dict[int, set[int]] = {i: {i} for i in range(n)}
    for k, (c1, c2, _, _) in enumerate(Z):
        c1, c2 = int(c1), int(c2)
        node_leaves[n + k] = node_leaves[c1] | node_leaves[c2]

    parent: dict[int, int] = {}
    for k, (c1, c2, _, _) in enumerate(Z):
        c1, c2 = int(c1), int(c2)
        parent[c1] = n + k
        parent[c2] = n + k

    def lca(i: int, j: int) -> int:
        ancestors_i: set[int] = set()
        curr = i
        while curr in parent:
            curr = parent[curr]
            ancestors_i.add(curr)
        curr = j
        while curr in parent:
            curr = parent[curr]
            if curr in ancestors_i:
                return curr
        return n + len(Z) - 1

    label_groups: dict[str, list[int]] = {}
    for i, lbl in enumerate(labels):
        if lbl not in excluded:
            label_groups.setdefault(lbl, []).append(i)

    total, count = 0.0, 0
    for lbl, group in label_groups.items():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                node = lca(group[a], group[b])
                leaves = node_leaves[node]
                purity = sum(labels[l] == lbl for l in leaves) / len(leaves)
                total += purity
                count += 1

    return total / count if count > 0 else 0.0


@dataclass
class TaxoNode:
    hac_id: int
    leaves: list[int]
    coherence: float
    children: list[TaxoNode] = field(default_factory=list)
    is_outlier: bool = False  # True for asymmetric-split orphan leaves

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def flat_labels(self, n: int | None = None) -> np.ndarray:
        """
        Flat cluster labels (0..K-1) for the n original papers.

        Must be called on the root. `n` should be the total number of input
        papers; if omitted, inferred from max leaf index (unsafe if the last
        paper is missing — pass `n` explicitly when possible).
        """
        if n is None:
            n = max(self.leaves) + 1
        labels = np.full(n, -1, dtype=int)
        for cluster_id, node in enumerate(self._leaf_nodes()):
            for idx in node.leaves:
                labels[idx] = cluster_id

        missing = int((labels == -1).sum())
        if missing:
            raise AssertionError(
                f"flat_labels: {missing}/{n} papers not assigned to any leaf — "
                f"cut_tree_adaptive lost papers, check force_split branch."
            )
        return labels

    def _leaf_nodes(self) -> list[TaxoNode]:
        if self.is_leaf:
            return [self]
        result = []
        for child in self.children:
            result.extend(child._leaf_nodes())
        return result

    def all_papers(self) -> set[int]:
        """All paper indices reachable under this node (debugging / audit)."""
        return set(self.leaves)

    def outlier_leaves(self) -> list[TaxoNode]:
        """Convenience: list terminal leaves marked as outliers."""
        return [n for n in self._leaf_nodes() if n.is_outlier]


def cut_tree_adaptive(
    Z: np.ndarray,
    embeddings: np.ndarray,
    hard_max: int = 50,
    min_gain: float = 0.05,
    min_cluster_size: int = 3,
) -> TaxoNode:
    """
    Three-regime adaptive cut of a HAC dendrogram, returned as a TaxoNode tree.

    Regimes (evaluated top-down on each node):
      - size > hard_max                                           → force split
      - size > 2*min_cluster_size, gain >= min_gain, both children large enough → coherence split
      - otherwise                                                 → leaf

    Coherence gain = weighted_mean(coh_children) - coh_parent.  A split is taken
    only when the gain is positive enough (relative criterion) AND both children
    have at least min_cluster_size leaves.  The minimum splittable size is
    2*min_cluster_size (the smallest size that can yield two viable children).

    Force splits are always taken when size > hard_max, even if the split is
    asymmetric: the smaller side is materialised as an outlier leaf so no paper
    is silently lost.

    Invariant guaranteed: every input paper appears in exactly one terminal leaf.
    """
    n = len(embeddings)

    # Pre-compute leaves and children for every internal node
    node_leaves: dict[int, list[int]] = {i: [i] for i in range(n)}
    for k, (c1, c2, _, _) in enumerate(Z):
        node_leaves[n + k] = node_leaves[int(c1)] + node_leaves[int(c2)]
    node_children: dict[int, tuple[int, int]] = {
        n + k: (int(c1), int(c2)) for k, (c1, c2, _, _) in enumerate(Z)
    }

    # Normalize once; cosine similarity = dot product on unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embeddings / norms

    def _coherence(leaves: list[int]) -> float:
        m = len(leaves)
        if m < 2:
            return 1.0
        sub = unit[leaves]
        sims = sub @ sub.T
        iu = np.triu_indices(m, k=1)
        return float(sims[iu].mean())

    def _make_leaf(
        node: int, leaves: list[int], coh: float, is_outlier: bool = False
    ) -> TaxoNode:
        return TaxoNode(
            hac_id=node,
            leaves=leaves,
            coherence=coh,
            is_outlier=is_outlier,
        )

    def _traverse(node: int) -> TaxoNode:
        leaves = node_leaves[node]
        size = len(leaves)
        coh = _coherence(leaves)
        c1, c2 = node_children.get(node, (None, None))

        # Terminal cases: HAC leaf, or cluster small enough to keep whole
        if c1 is None or size <= min_cluster_size:
            return _make_leaf(node, leaves, coh)

        size_c1, size_c2 = len(node_leaves[c1]), len(node_leaves[c2])
        both_large_enough = min(size_c1, size_c2) >= min_cluster_size

        # Safety split: cluster too broad to keep as a leaf
        if size > hard_max:
            if both_large_enough:
                return TaxoNode(
                    hac_id=node,
                    leaves=leaves,
                    coherence=coh,
                    children=[_traverse(c1), _traverse(c2)],
                )
            # Asymmetric: materialise the smaller side as an outlier leaf
            small, large = (c1, c2) if size_c1 < size_c2 else (c2, c1)
            small_leaves = node_leaves[small]
            small_node = _make_leaf(
                small,
                small_leaves,
                _coherence(small_leaves),
                is_outlier=True,
            )
            return TaxoNode(
                hac_id=node,
                leaves=leaves,
                coherence=coh,
                children=[small_node, _traverse(large)],
            )

        # Coherence-gain split: worthwhile only when both children are viable
        if size > 2 * min_cluster_size and both_large_enough:
            coh_c1 = _coherence(node_leaves[c1])
            coh_c2 = _coherence(node_leaves[c2])
            coh_children = (coh_c1 * size_c1 + coh_c2 * size_c2) / (size_c1 + size_c2)
            if coh_children - coh >= min_gain:
                return TaxoNode(
                    hac_id=node,
                    leaves=leaves,
                    coherence=coh,
                    children=[_traverse(c1), _traverse(c2)],
                )

        # Default: leaf
        return _make_leaf(node, leaves, coh)

    return _traverse(n + len(Z) - 1)


def flat_labels(root: TaxoNode, n: int) -> np.ndarray:
    """
    Assign each of the `n` input papers a terminal-leaf id.

    Walks the tree, collecting only nodes with no children. Asserts every
    paper is covered exactly once (catches regressions of the silent-loss bug).
    """
    labels = np.full(n, -1, dtype=np.int64)

    def _walk(node: TaxoNode) -> None:
        if not node.children:
            for paper_idx in node.leaves:
                labels[paper_idx] = node.hac_id
            return
        for child in node.children:
            _walk(child)

    _walk(root)

    missing = int((labels == -1).sum())
    if missing:
        raise AssertionError(
            f"flat_labels: {missing}/{n} papers not assigned to any leaf — "
            f"cut_tree_adaptive lost papers, check force_split branch."
        )
    return labels


def plot_taxo_tree(
    root: TaxoNode,
    df: pd.DataFrame,
    label_col: str,
    title: str = "Adaptive Taxonomy Tree",
) -> go.Figure:
    """
    Tree diagram of a TaxoNode tree (horizontal, root on left).

    Leaf nodes show their dominant label. Internal nodes show coherence + size.
    Node colour = coherence (RdYlGn). `df` must be indexed 0..n-1.
    """
    # ── 1. Compute layout positions ──────────────────────────────────────────
    # x = depth, y = vertical position (leaves spaced 1 apart)
    positions: dict[int, tuple[float, float]] = {}
    leaf_counter: list[float] = [0.0]

    def _layout(node: TaxoNode, depth: int) -> None:
        if node.is_leaf:
            positions[node.hac_id] = (depth, leaf_counter[0])
            leaf_counter[0] += 1.0
        else:
            for child in node.children:
                _layout(child, depth + 1)
            ys = [positions[c.hac_id][1] for c in node.children]
            positions[node.hac_id] = (depth, sum(ys) / len(ys))

    _layout(root, 0)

    # ── 2. Build edge traces ──────────────────────────────────────────────────
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []

    def _add_edges(node: TaxoNode) -> None:
        px, py = positions[node.hac_id]
        for child in node.children:
            cx, cy = positions[child.hac_id]
            # Elbow: horizontal then vertical
            edge_x.extend([px, cx, cx, None])
            edge_y.extend([py, py, cy, None])
            _add_edges(child)

    _add_edges(root)

    # ── 3. Build node traces ──────────────────────────────────────────────────
    def _top_label(leaves: list[int]) -> str:
        counts = df.iloc[leaves][label_col].value_counts()
        return counts.index[0] if len(counts) else "?"

    node_x, node_y, node_color, node_size = [], [], [], []
    text_x, text_y, text_labels = [], [], []
    hover_texts = []

    def _add_nodes(node: TaxoNode) -> None:
        x, y = positions[node.hac_id]
        top = _top_label(node.leaves)
        n = len(node.leaves)
        node_x.append(x)
        node_y.append(y)
        node_color.append(node.coherence)
        node_size.append(max(6, int(np.sqrt(n) * 3)))
        hover_texts.append(f"<b>{top}</b><br>n={n}  coh={node.coherence:.2f}")
        if node.is_leaf:
            text_x.append(x)
            text_y.append(y)
            text_labels.append(f"  {top} (n={n})")
        for child in node.children:
            _add_nodes(child)

    _add_nodes(root)

    # ── 4. Assemble figure ────────────────────────────────────────────────────
    n_leaves = int(leaf_counter[0])
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="lightgray", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale="RdYlGn",
                cmin=0.0,
                cmax=1.0,
                showscale=True,
                colorbar=dict(title="Coherence", thickness=12, len=0.5),
                line=dict(color="gray", width=0.5),
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=text_x,
            y=text_y,
            mode="text",
            text=text_labels,
            textposition="middle right",
            textfont=dict(size=10, family="monospace"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    max_depth = max(x for x, _ in positions.values())
    fig.update_layout(
        title=title,
        height=max(600, 18 * n_leaves),
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.3, max_depth + 8],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1, n_leaves],
        ),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def cophenetic_table(
    Z: np.ndarray,
    labels: np.ndarray,
    excluded: frozenset[str] = EXCLUDED_LABELS,
) -> pd.DataFrame:
    coph_matrix = squareform(cophenet(Z))
    unique_labels = [l for l in np.unique(labels) if l not in excluded]

    rows = []
    for lbl in unique_labels:
        idx_in = np.where(labels == lbl)[0]
        idx_out = np.where((labels != lbl) & ~np.isin(labels, list(excluded)))[0]
        if len(idx_in) < 2 or len(idx_out) == 0:
            continue
        intra_vals = coph_matrix[np.ix_(idx_in, idx_in)][
            np.triu_indices(len(idx_in), k=1)
        ]
        inter_vals = coph_matrix[np.ix_(idx_in, idx_out)].ravel()
        intra_m = float(intra_vals.mean())
        inter_m = float(inter_vals.mean())
        rows.append(
            {
                "family": lbl,
                "intra": round(intra_m, 3),
                "inter": round(inter_m, 3),
                "ratio": round(intra_m / inter_m, 3),
                "n": len(idx_in),
            }
        )

    return pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)
